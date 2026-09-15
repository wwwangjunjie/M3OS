"""LLM factory for creating language model instances.

This module provides a unified interface for creating LLM instances
from different providers (Claude, Gemini, Kimi) with appropriate middleware.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from m3os.agents_v5.core.config import LLMConfig


class KimiChatOpenAI(ChatOpenAI):
    """ChatOpenAI adapter for Kimi-specific reasoning and tool-call fields.

    Kimi K3 always reasons and accepts ``tool_choice="required"``, but it does
    not accept OpenAI's named-function ``tool_choice`` form. LangChain's
    function-calling structured-output path normally uses that named form.
    Since each structured-output invocation binds exactly one schema tool,
    ``required`` has the same effect while remaining compatible with K3.

    K3 also requires its returned ``reasoning_content`` to be preserved across
    multi-turn tool calls. ChatOpenAI currently drops this provider extension,
    so retain it in ``AIMessage.additional_kwargs`` and restore it in the next
    request payload.
    """

    @property
    def is_kimi_k3(self) -> bool:
        return str(self.model_name or "").strip().lower().startswith("kimi-k3")

    def with_structured_output(
        self,
        schema: Any = None,
        *,
        method: str = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        tools: Optional[list] = None,
        **kwargs: Any,
    ) -> Any:
        if self.is_kimi_k3 and method == "function_calling":
            kwargs["tool_choice"] = "required"
        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            tools=tools,
            **kwargs,
        )

    def _create_chat_result(
        self,
        response: Any,
        generation_info: Optional[dict] = None,
    ) -> Any:
        response_dict = response if isinstance(response, dict) else response.model_dump()
        result = super()._create_chat_result(response, generation_info)
        choices = response_dict.get("choices") or []
        for generation, choice in zip(result.generations, choices):
            raw_message = choice.get("message") or {}
            if (
                isinstance(generation.message, AIMessage)
                and "reasoning_content" in raw_message
            ):
                generation.message.additional_kwargs["reasoning_content"] = (
                    raw_message.get("reasoning_content")
                )
        return result

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        source_messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload_messages = payload.get("messages")
        if isinstance(payload_messages, list):
            for source, target in zip(source_messages, payload_messages):
                if not isinstance(source, AIMessage) or not isinstance(target, dict):
                    continue
                if "reasoning_content" in source.additional_kwargs:
                    target["reasoning_content"] = source.additional_kwargs[
                        "reasoning_content"
                    ]
        return payload


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def create_model(self, config: LLMConfig):
        """Create and return the language model instance."""
        pass
    
    @abstractmethod
    def get_middleware(self, config: LLMConfig) -> List[Any]:
        """Get provider-specific middleware list."""
        pass


class ClaudeProvider(LLMProvider):
    """Anthropic Claude LLM provider."""
    
    def create_model(self, config: LLMConfig):
        """Create a Claude model instance."""
        from langchain_anthropic import ChatAnthropic
        
        if not config.claude_api_key:
            raise ValueError("Claude API key not configured")
        
        return ChatAnthropic(
            model=config.claude_model,
            base_url=config.claude_base_url,
            api_key=config.claude_api_key,
            timeout=config.request_timeout_seconds,
            stream_usage=False
        )
    
    def get_middleware(self, config: LLMConfig) -> List[Any]:
        """Get Claude-specific middleware.

        Deep Agents already injects `AnthropicPromptCachingMiddleware`
        internally, so only add retry middleware here to avoid duplicate
        middleware names during `create_deep_agent()`.
        """
        return [
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
                max_delay=60.0,
                jitter=True,
            ),
        ]


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""
    
    def create_model(self, config: LLMConfig):
        """Create a Gemini model instance."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not config.gemini_api_key:
            raise ValueError("Gemini API key not configured")
        
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            base_url=config.gemini_base_url,
            api_key=config.gemini_api_key,
            request_timeout=config.request_timeout_seconds,
            disable_streaming=True,
        )
    
    def get_middleware(self, config: LLMConfig) -> List[Any]:
        """Get Gemini-specific middleware."""
        # Note: Gemini doesn't have prompt caching middleware like Claude
        # but we still need retry middleware
        return [
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
                max_delay=300.0,  # Higher max delay for Gemini
                jitter=True,
            ),
        ]


class KimiProvider(LLMProvider):
    """Moonshot Kimi provider using its OpenAI-compatible API."""

    def create_model(self, config: LLMConfig):
        """Create a Kimi model instance."""
        if not config.kimi_api_key:
            raise ValueError("Kimi API key not configured")

        model_kwargs: Dict[str, Any] = {}
        is_kimi_k3 = str(config.kimi_model or "").strip().lower().startswith("kimi-k3")
        if is_kimi_k3:
            reasoning_effort = str(config.kimi_reasoning_effort or "max").strip().lower()
            if reasoning_effort not in {"low", "high", "max"}:
                raise ValueError(
                    "KIMI_REASONING_EFFORT must be one of: low, high, max"
                )
            model_kwargs["reasoning_effort"] = reasoning_effort
        else:
            # Kimi K2.5/K2.6 can disable thinking and require it for named
            # structured-output tool choice. K3 always reasons and rejects this
            # parameter, so never send it to K3.
            model_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        return KimiChatOpenAI(
            model=config.kimi_model,
            base_url=config.kimi_base_url,
            api_key=config.kimi_api_key,
            timeout=config.kimi_request_timeout_seconds,
            max_retries=config.kimi_max_retries,
            disable_streaming=True,
            stream_usage=False,
            **model_kwargs,
        )

    def get_middleware(self, config: LLMConfig) -> List[Any]:
        """Get Kimi-specific middleware."""
        return [
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
                max_delay=60.0,
                jitter=True,
            ),
        ]


class LLMFactory:
    """Factory for creating LLM instances.
    
    This factory provides a unified way to create LLM instances
    from different providers with appropriate configuration.
    """
    
    _providers: Dict[str, LLMProvider] = {
        "claude": ClaudeProvider(),
        "gemini": GeminiProvider(),
        "kimi": KimiProvider(),
    }
    
    @classmethod
    def create_model(cls, config: LLMConfig) -> Any:
        """Create an LLM model based on configuration.
        
        Args:
            config: LLM configuration
            
        Returns:
            LangChain chat model instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = cls._providers.get(config.provider.lower())
        if not provider:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        
        return provider.create_model(config)
    
    @classmethod
    def get_middleware(cls, config: LLMConfig) -> List[Any]:
        """Get middleware for the configured provider.
        
        Args:
            config: LLM configuration
            
        Returns:
            List of middleware instances
        """
        provider = cls._providers.get(config.provider.lower())
        if not provider:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        
        return provider.get_middleware(config)
    
    @classmethod
    def register_provider(cls, name: str, provider: LLMProvider) -> None:
        """Register a new LLM provider.
        
        Args:
            name: Provider name
            provider: Provider instance
        """
        cls._providers[name.lower()] = provider


def create_base_model(config: LLMConfig) -> Any:
    """Convenience function to create a base LLM model.
    
    Args:
        config: LLM configuration
        
    Returns:
        LangChain chat model instance
    """
    return LLMFactory.create_model(config)


def get_middleware(config: LLMConfig) -> List[Any]:
    """Convenience function to get middleware for the configured provider.
    
    Args:
        config: LLM configuration
        
    Returns:
        List of middleware instances
    """
    return LLMFactory.get_middleware(config)
