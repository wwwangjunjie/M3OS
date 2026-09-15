"""Retry utilities for LLM invocations.

This module provides retry logic with exponential backoff for handling
transient failures in LLM API calls.
"""

import asyncio
import random
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class LLMInvocationError(Exception):
    """Raised when LLM invocation fails after all retries."""
    pass


class InvalidResponseError(Exception):
    """Raised when LLM returns an invalid or empty response."""
    pass


async def invoke_with_retry(
    callable_fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_ratio: float = 0.5,
    validate_fn: Optional[Callable[[T], bool]] = None,
    operation_name: str = "LLM call",
) -> T:
    """Execute a callable with retry logic and exponential backoff.

    Args:
        callable_fn: The async or sync callable to execute
        max_retries: Maximum number of total attempts (default: 3)
        base_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        jitter_ratio: Random jitter ratio added to delay. Set to 0 for fixed waits.
        validate_fn: Optional function to validate the result
        operation_name: Name of the operation for logging purposes

    Returns:
        The result from the callable

    Raises:
        LLMInvocationError: If all retries are exhausted
        InvalidResponseError: If validation fails and all retries are exhausted
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            # Handle both sync and async callables
            if asyncio.iscoroutinefunction(callable_fn):
                result = await callable_fn()
            else:
                result = callable_fn()

            # Validate result if validation function provided
            if validate_fn is not None:
                if not validate_fn(result):
                    raise InvalidResponseError(
                        f"Validation failed for {operation_name}"
                    )

            # Success - return result
            if attempt > 0:
                print(f"[Retry] {operation_name} succeeded on attempt {attempt + 1}")
            return result

        except Exception as e:
            last_exception = e
            attempt_num = attempt + 1

            if attempt_num >= max_retries:
                # All retries exhausted
                break

            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * jitter_ratio) if jitter_ratio > 0 else 0
            total_delay = delay + jitter

            print(
                f"[Retry] {operation_name} failed (attempt {attempt_num}/{max_retries}): "
                f"{type(e).__name__}: {e}. Retrying in {total_delay:.2f}s..."
            )

            await asyncio.sleep(total_delay)

    # All retries exhausted
    raise LLMInvocationError(
        f"{operation_name} failed after {max_retries} attempts. "
        f"Last error: {type(last_exception).__name__}: {last_exception}"
    )


def validate_structured_response(result: Any) -> bool:
    """Validate that a structured LLM response contains valid parsed data.

    Args:
        result: The response from structured LLM invocation

    Returns:
        True if response is valid, False otherwise
    """
    if result is None:
        return False

    def _has_valid_payload(payload: Any) -> bool:
        if payload is None:
            return False

        if isinstance(payload, dict):
            if not payload:
                return False
            if "optimizations" in payload:
                optimizations = payload.get("optimizations")
                return isinstance(optimizations, list) and len(optimizations) > 0
            return True

        optimizations = getattr(payload, "optimizations", None)
        if optimizations is not None:
            return isinstance(optimizations, list) and len(optimizations) > 0

        if hasattr(payload, "__dict__"):
            return bool(payload.__dict__)

        return True

    # Handle LangChain structured output format (dict with 'parsed' key)
    if isinstance(result, dict):
        if "parsed" in result and _has_valid_payload(result["parsed"]):
            return True
        if "structured_response" in result:
            return _has_valid_payload(result["structured_response"])
        if not result:
            return False

    # Handle direct Pydantic model output
    if hasattr(result, "__dict__") and result.__dict__:
        return _has_valid_payload(result)

    return False
