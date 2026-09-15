"""Configuration management for agents module.

This module provides centralized configuration loading from environment variables
and config files, replacing hardcoded paths and settings.
"""

import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from configs.settings import get_setting, load_settings


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_TMP_DIR = PROJECT_ROOT / "tmp"


def _resolve_repo_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _current_tmp_dir() -> Path:
    value = os.getenv("M3OS_TMP_DIR")
    if not value or not value.strip():
        return PROJECT_TMP_DIR
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _resolve_intermediate_path(value: Optional[str], default: str) -> str:
    text = (value if value is not None else default).strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "tmp":
        return str((_current_tmp_dir().joinpath(*path.parts[1:])).resolve())
    return str((_current_tmp_dir() / path).resolve())


def _resolve_command_path(value: str) -> str:
    text = value.strip()
    if not text:
        return text
    if os.sep in text or (os.altsep and os.altsep in text) or text.startswith("."):
        resolved = _resolve_repo_path(text)
        return resolved if resolved is not None else text
    return text


def _resolve_command_argv(values: Optional[list[str]]) -> list[str]:
    """Resolve executable/script paths in a configured command argument list."""
    if not values:
        return []
    resolved: list[str] = []
    for index, value in enumerate(values):
        text = str(value).strip()
        if not text:
            continue
        if index == 0:
            resolved.append(_resolve_command_path(text))
            continue
        if (
            not text.startswith("-")
            and not text.startswith(("http://", "https://"))
            and (os.sep in text or (os.altsep and os.altsep in text) or text.startswith("."))
        ):
            path = _resolve_repo_path(text)
            resolved.append(path if path is not None else text)
            continue
        resolved.append(text)
    return resolved


def _clear_langsmith_env_cache() -> None:
    """Refresh LangSmith env lookups after loading a dotenv file.

    Some LangChain/LangSmith imports can query tracing env vars before this
    module loads the requested .env file. LangSmith caches those lookups, so
    clear the cache once dotenv has populated os.environ.
    """
    try:
        from langsmith import utils as langsmith_utils

        langsmith_utils.get_env_var.cache_clear()
    except Exception:
        pass


def _sync_langsmith_env_aliases() -> None:
    """Expose LangSmith settings through legacy LangChain env names too."""
    alias_pairs = {
        "LANGSMITH_TRACING": "LANGCHAIN_TRACING_V2",
        "LANGSMITH_ENDPOINT": "LANGCHAIN_ENDPOINT",
        "LANGSMITH_API_KEY": "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT": "LANGCHAIN_PROJECT",
    }
    for langsmith_name, langchain_name in alias_pairs.items():
        value = os.getenv(langsmith_name)
        if value is not None:
            os.environ[langchain_name] = value


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "claude"  # "claude", "gemini", or "kimi"
    request_timeout_seconds: float = 1200.0
    
    # Claude settings
    claude_model: str = "claude-3-5-sonnet-20241022"
    claude_base_url: Optional[str] = None
    claude_api_key: Optional[str] = None
    
    # Gemini settings
    gemini_model: str = "gemini-2.5-pro-exp-03-25"
    gemini_base_url: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Kimi settings (OpenAI-compatible API)
    kimi_model: str = "kimi-k2.6"
    kimi_base_url: Optional[str] = None
    kimi_api_key: Optional[str] = None
    kimi_reasoning_effort: str = "max"
    kimi_max_retries: int = 8
    kimi_request_timeout_seconds: float = 1200.0

@dataclass
class MCPConfig:
    """Configuration for MCP client connections."""
    main_server_path: Optional[str] = None
    main_server_python: str = sys.executable
    main_server_url: Optional[str] = None
    main_server_transport: str = "streamable_http"
    server_init_timeout_seconds: float = 120.0
    tool_timeout_seconds: float = 1200.0
    auto_restart_enabled: bool = True
    restart_timeout_seconds: float = 300.0
    restart_poll_interval_seconds: float = 2.0
    restart_cooldown_seconds: float = 30.0
    restart_lock_dir: str = str(PROJECT_TMP_DIR / "mcp_restart_locks")
    server_restart_commands: dict[str, list[str]] = field(default_factory=dict)
    admet_ai_url: Optional[str] = None
    nesso_url: Optional[str] = None
    transformercpi_v2_url: Optional[str] = None
    boltz_url: Optional[str] = None
    iupac_gen_url: Optional[str] = None
    reinvent_url: Optional[str] = None
    fast_medchem_url: Optional[str] = None
    fast_medchem_transport: str = "streamable_http"
    mmp_url: Optional[str] = None
    mmp_transport: str = "streamable_http"
    m3os_wiki_command: Optional[str] = None
    m3os_wiki_args: Optional[list[str]] = None
    m3os_wiki_url: Optional[str] = None
    m3os_wiki_transport: str = "stdio"
    
    def get_server_configs(self) -> dict:
        """Generate server configurations for MultiServerMCPClient."""
        configs = {}
        
        if self.main_server_url:
            transport = self.main_server_transport or "streamable_http"
            if transport == "stdio":
                transport = "streamable_http"
            configs['mcp_server'] = {
                "url": self.main_server_url,
                "transport": transport,
            }
        elif self.main_server_path:
            configs['mcp_server'] = {
                "command": self.main_server_python,
                "args": [self.main_server_path],
                "transport": "stdio",
            }
        
        if self.iupac_gen_url:
            configs['mcp_server_iupac_gen'] = {
                "url": self.iupac_gen_url,
                "transport": "streamable_http",
            }
        
        if self.admet_ai_url:
            configs['mcp_server_admet_ai'] = {
                "url": self.admet_ai_url,
                "transport": "streamable_http",
            }

        if self.nesso_url:
            configs['mcp_server_nesso_cofolding'] = {
                "url": self.nesso_url,
                "transport": "streamable_http",
            }

        if self.transformercpi_v2_url:
            configs['mcp_server_transformercpi_v2'] = {
                "url": self.transformercpi_v2_url,
                "transport": "streamable_http",
            }

        if self.reinvent_url:
            configs['mcp_server_reinvent'] = {
                "url": self.reinvent_url,
                "transport": "streamable_http",
            }

        if self.fast_medchem_url:
            transport = self.fast_medchem_transport or "streamable_http"
            if transport == "stdio":
                transport = "streamable_http"
            configs['mcp_server_fast_medchem_request'] = {
                "url": self.fast_medchem_url,
                "transport": transport,
            }

        if self.mmp_url:
            transport = self.mmp_transport or "streamable_http"
            if transport == "stdio":
                transport = "streamable_http"
            configs['mcp_server_m3os_mmp'] = {
                "url": self.mmp_url,
                "transport": transport,
            }
        
        if self.boltz_url:
            configs['mcp_server_boltz_plip'] = {
                "url": self.boltz_url,
                "transport": "streamable_http",
            }

        if self.m3os_wiki_url:
            transport = self.m3os_wiki_transport or "streamable_http"
            if transport == "stdio":
                transport = "streamable_http"
            configs['mcp_server_m3os_wiki'] = {
                "url": self.m3os_wiki_url,
                "transport": transport,
            }
        elif self.m3os_wiki_command:
            configs['mcp_server_m3os_wiki'] = {
                "command": self.m3os_wiki_command,
                "args": self.m3os_wiki_args or [],
                "transport": "stdio",
            }
        
        return configs


@dataclass
class PathsConfig:
    """Configuration for file paths."""
    property_meta_info_path: Optional[str] = None
    csv_save_path_prefix: str = str(PROJECT_TMP_DIR)
    env_file_path: Optional[str] = None
    rag_data_dir: str = str(PROJECT_TMP_DIR / "rag")
    rag_chunk_size: int = 1024
    rag_chunk_overlap: int = 200
    rag_top_k: int = 8
    rag_full_text_max_chars: int = 700000


@dataclass
class DeepAgentConfig:
    """Configuration for Deep Agents runtime and skills."""

    backend_root_dir: str = "."
    skills_root: str = "m3os/agents_v5/skills"
    enable_optimization_case_retrieval: bool = True

    def _backend_virtual_path(self, path: str) -> str:
        """Return a path usable by FilesystemBackend virtual mode."""
        text = str(path or "").strip()
        if not text:
            return text
        source_path = Path(text).expanduser()
        if not source_path.is_absolute():
            return source_path.as_posix()

        root_path = Path(self.backend_root_dir or ".").expanduser()
        try:
            root_path = root_path.resolve()
            source_path = source_path.resolve()
            relative_path = source_path.relative_to(root_path)
        except (OSError, ValueError):
            return text
        return "/" + relative_path.as_posix()

    def get_skill_sources(self, agent_role: str) -> list[str]:
        """Get skill source directories for a given role.

        Deep Agents expects each source path to be a directory whose immediate
        child directories are skills. Following the official layout, we expose a
        single flat source root and let role prompts, skill descriptions, and
        tool availability determine which skills are actually used.
        """
        supported_roles = {
            "creative",
            "rational",
            "critic",
            "report",
            "medchem_retrieval",
        }
        if agent_role.lower() not in supported_roles:
            return []
        return [self._backend_virtual_path(self.skills_root)]

    def get_allowed_skill_names(self, agent_role: str) -> list[str]:
        """Get the skills that should be visible to the given agent role."""
        role_to_skills = {
            "creative": [
                "smiles-validity-gating",
                "reinvent-candidate-generation-filtering",
                "leadopt-intersection-screening",
            ],
            "rational": [
                "smiles-validity-gating",
                "optimization-case-retrieval",
                "protein-context-check",
                "case-guided-medicinal-design",
            ],
            "critic": [
                "smiles-validity-gating",
                "candidate-quality-gating",
                "activity-consensus-evaluation",
            ],
            "medchem_retrieval": [
                "medchem-knowledge-retrieval",
                "m3os-wiki-mcp-retrieval",
            ],
            "report": [],
        }
        allowed = role_to_skills.get(agent_role.lower(), [])
        if (
            agent_role.lower() == "rational"
            and not self.enable_optimization_case_retrieval
        ):
            return [
                name for name in allowed
                if name != "optimization-case-retrieval"
            ]
        return allowed


@dataclass
class AgentConfig:
    """Complete configuration for the agents module.
    
    This configuration class centralizes all settings needed for the
    molecular optimization multi-agent system.
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    deepagent: DeepAgentConfig = field(default_factory=DeepAgentConfig)
    enable_shared_analysis: bool = True
    enable_molecular_auxiliary_context: bool = True
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "AgentConfig":
        """Load configuration from environment variables and optional TOML defaults.
        
        Args:
            env_file: Optional path to .env file to load
            
        Returns:
            AgentConfig instance with values from environment
        """
        settings = load_settings(
            env_file,
            use_default_config=env_file is not None,
            dotenv_override=True,
        )
        _sync_langsmith_env_aliases()
        _clear_langsmith_env_cache()
        
        def _setting(name: str, config_path: str, default=None):
            return get_setting(settings, name, config_path, default)

        def _setting_bool(name: str, config_path: str, default: bool) -> bool:
            value = _setting(name, config_path, None)
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

        def _setting_int(name: str, config_path: str, default: int) -> int:
            return int(_setting(name, config_path, default))

        def _setting_float(name: str, config_path: str, default: float) -> float:
            return float(_setting(name, config_path, default))

        def _setting_list(name: str, config_path: str) -> Optional[list[str]]:
            value = _setting(name, config_path, None)
            if value is None:
                return None
            if isinstance(value, list):
                return [str(item) for item in value]
            text = str(value).strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return shlex.split(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return shlex.split(text)

        def _restart_command(name: str, config_path: str) -> list[str]:
            return _resolve_command_argv(_setting_list(name, config_path))

        # LLM Configuration
        llm_config = LLMConfig(
            provider=str(_setting("LLM_PROVIDER", "llm.provider", "claude")),
            request_timeout_seconds=_setting_float(
                "LLM_REQUEST_TIMEOUT_SECONDS",
                "llm.request_timeout_seconds",
                1200.0,
            ),
            claude_model=str(_setting("CLAUDE_MODEL", "llm.claude_model", "claude-3-5-sonnet-20241022")),
            claude_base_url=_setting("CLAUDE_BASE_URL", "llm.claude_base_url"),
            claude_api_key=_setting("CLAUDE_API_KEY", "llm.claude_api_key"),
            gemini_model=str(_setting("GEMINI_MODEL", "llm.gemini_model", "gemini-2.5-pro-exp-03-25")),
            gemini_base_url=_setting("GEMINI_BASE_URL", "llm.gemini_base_url"),
            gemini_api_key=_setting("GEMINI_API_KEY", "llm.gemini_api_key"),
            kimi_model=str(_setting("KIMI_MODEL", "llm.kimi_model", "kimi-k2.6")),
            kimi_base_url=_setting("KIMI_BASE_URL", "llm.kimi_base_url"),
            kimi_api_key=_setting("KIMI_API_KEY", "llm.kimi_api_key"),
            kimi_reasoning_effort=str(
                _setting(
                    "KIMI_REASONING_EFFORT",
                    "llm.kimi_reasoning_effort",
                    "max",
                )
            ),
            kimi_max_retries=_setting_int(
                "KIMI_MAX_RETRIES",
                "llm.kimi_max_retries",
                8,
            ),
            kimi_request_timeout_seconds=_setting_float(
                "KIMI_REQUEST_TIMEOUT_SECONDS",
                "llm.kimi_request_timeout_seconds",
                1200.0,
            ),
        )
        
        # MCP Configuration
        m3os_wiki_url = _setting("M3OS_WIKI_MCP_URL", "mcp.m3os_wiki_url")
        m3os_wiki_transport = _setting("M3OS_WIKI_MCP_TRANSPORT", "mcp.m3os_wiki_transport")
        if m3os_wiki_transport is None:
            m3os_wiki_transport = "streamable_http" if m3os_wiki_url else "stdio"
        mcp_config = MCPConfig(
            main_server_path=_resolve_repo_path(_setting("MCP_SERVER_PATH", "mcp.main_server_path")),
            main_server_python=_resolve_command_path(str(_setting("MCP_SERVER_PYTHON", "mcp.main_server_python", sys.executable))),
            main_server_url=_setting("MCP_SERVER_URL", "mcp.main_server_url"),
            main_server_transport=str(
                _setting(
                    "MCP_SERVER_TRANSPORT",
                    "mcp.main_server_transport",
                    "streamable_http",
                )
            ),
            server_init_timeout_seconds=_setting_float(
                "MCP_SERVER_INIT_TIMEOUT_SECONDS",
                "mcp.server_init_timeout_seconds",
                120.0,
            ),
            tool_timeout_seconds=_setting_float("MCP_TOOL_TIMEOUT_SECONDS", "mcp.tool_timeout_seconds", 1200.0),
            auto_restart_enabled=_setting_bool(
                "MCP_AUTO_RESTART_ENABLED",
                "mcp.auto_restart_enabled",
                True,
            ),
            restart_timeout_seconds=_setting_float(
                "MCP_RESTART_TIMEOUT_SECONDS",
                "mcp.restart_timeout_seconds",
                300.0,
            ),
            restart_poll_interval_seconds=_setting_float(
                "MCP_RESTART_POLL_INTERVAL_SECONDS",
                "mcp.restart_poll_interval_seconds",
                2.0,
            ),
            restart_cooldown_seconds=_setting_float(
                "MCP_RESTART_COOLDOWN_SECONDS",
                "mcp.restart_cooldown_seconds",
                30.0,
            ),
            restart_lock_dir=_resolve_repo_path(
                _setting(
                    "MCP_RESTART_LOCK_DIR",
                    "mcp.restart_lock_dir",
                    "tmp/mcp_restart_locks",
                )
            )
            or str(PROJECT_TMP_DIR / "mcp_restart_locks"),
            server_restart_commands={
                server_name: command
                for server_name, command in {
                    "mcp_server": _restart_command(
                        "MCP_MAIN_SERVER_RESTART_COMMAND",
                        "mcp.main_server_restart_command",
                    ),
                    "mcp_server_admet_ai": _restart_command(
                        "MCP_SERVER_ADMET_AI_RESTART_COMMAND",
                        "mcp.admet_ai_restart_command",
                    ),
                    "mcp_server_nesso_cofolding": _restart_command(
                        "MCP_SERVER_NESSO_RESTART_COMMAND",
                        "mcp.nesso_restart_command",
                    ),
                    "mcp_server_transformercpi_v2": _restart_command(
                        "MCP_SERVER_TRANSFORMERCPI_V2_RESTART_COMMAND",
                        "mcp.transformercpi_v2_restart_command",
                    ),
                    "mcp_server_boltz_plip": _restart_command(
                        "MCP_SERVER_BOLTZ_RESTART_COMMAND",
                        "mcp.boltz_restart_command",
                    ),
                    "mcp_server_iupac_gen": _restart_command(
                        "MCP_SERVER_IUPAC_GEN_RESTART_COMMAND",
                        "mcp.iupac_gen_restart_command",
                    ),
                    "mcp_server_reinvent": _restart_command(
                        "MCP_SERVER_REINVENT_RESTART_COMMAND",
                        "mcp.reinvent_restart_command",
                    ),
                    "mcp_server_fast_medchem_request": _restart_command(
                        "M3OS_FAST_MEDCHEM_MCP_RESTART_COMMAND",
                        "mcp.fast_medchem_restart_command",
                    ),
                    "mcp_server_m3os_mmp": _restart_command(
                        "M3OS_MMP_MCP_RESTART_COMMAND",
                        "mcp.mmp_restart_command",
                    ),
                    "mcp_server_m3os_wiki": _restart_command(
                        "M3OS_WIKI_MCP_RESTART_COMMAND",
                        "mcp.m3os_wiki_restart_command",
                    ),
                }.items()
                if command
            },
            admet_ai_url=_setting("MCP_SERVER_ADMET_AI_URL", "mcp.admet_ai_url"),
            nesso_url=_setting("MCP_SERVER_NESSO_URL", "mcp.nesso_url"),
            transformercpi_v2_url=_setting(
                "MCP_SERVER_TRANSFORMERCPI_V2_URL",
                "mcp.transformercpi_v2_url",
            ),
            boltz_url=_setting("MCP_SERVER_BOLTZ_URL", "mcp.boltz_url"),
            iupac_gen_url=_setting("MCP_SERVER_IUPAC_GEN_URL", "mcp.iupac_gen_url"),
            reinvent_url=_setting("MCP_SERVER_REINVENT_URL", "mcp.reinvent_url"),
            fast_medchem_url=_setting("M3OS_FAST_MEDCHEM_MCP_URL", "mcp.fast_medchem_url"),
            fast_medchem_transport=_setting(
                "M3OS_FAST_MEDCHEM_MCP_TRANSPORT",
                "mcp.fast_medchem_transport",
                "streamable_http",
            ),
            mmp_url=_setting("M3OS_MMP_MCP_URL", "mcp.mmp_url"),
            mmp_transport=_setting(
                "M3OS_MMP_MCP_TRANSPORT",
                "mcp.mmp_transport",
                "streamable_http",
            ),
            m3os_wiki_command=_setting("M3OS_WIKI_MCP_COMMAND", "mcp.m3os_wiki_command"),
            m3os_wiki_args=_setting_list("M3OS_WIKI_MCP_ARGS", "mcp.m3os_wiki_args"),
            m3os_wiki_url=m3os_wiki_url,
            m3os_wiki_transport=str(m3os_wiki_transport),
        )
        
        # Paths Configuration
        paths_config = PathsConfig(
            property_meta_info_path=_resolve_repo_path(_setting("PROPERTY_META_INFO_PATH", "paths.property_meta_info_path")),
            csv_save_path_prefix=_resolve_intermediate_path(
                _setting("CSV_SAVE_PATH_PREFIX", "mcgs.csv_save_path_prefix"),
                "tmp",
            ),
            env_file_path=env_file,
            rag_data_dir=_resolve_intermediate_path(_setting("RAG_DATA_DIR", "paths.rag_data_dir"), "rag"),
            rag_chunk_size=_setting_int("RAG_CHUNK_SIZE", "rag.chunk_size", 1024),
            rag_chunk_overlap=_setting_int("RAG_CHUNK_OVERLAP", "rag.chunk_overlap", 200),
            rag_top_k=_setting_int("RAG_TOP_K", "rag.top_k", 8),
            rag_full_text_max_chars=_setting_int("RAG_FULL_TEXT_MAX_CHARS", "rag.full_text_max_chars", 700000),
        )

        deepagent_config = DeepAgentConfig(
            backend_root_dir=_resolve_repo_path(
                _setting("DEEPAGENTS_BACKEND_ROOT_DIR", "paths.deepagents_backend_root_dir", ".")
            ),
            skills_root=_resolve_repo_path(
                _setting("DEEPAGENTS_SKILLS_ROOT", "paths.deepagents_skills_root", "m3os/agents_v5/skills")
            ),
        )
        
        return cls(
            llm=llm_config,
            mcp=mcp_config,
            paths=paths_config,
            deepagent=deepagent_config,
            enable_shared_analysis=_setting_bool("ENABLE_SHARED_ANALYSIS", "runtime.enable_shared_analysis", True),
            enable_molecular_auxiliary_context=_setting_bool(
                "ENABLE_MOLECULAR_AUXILIARY_CONTEXT",
                "runtime.enable_molecular_auxiliary_context",
                True,
            ),
        )
    
    @classmethod
    def from_env_file(cls, env_file: str) -> "AgentConfig":
        """Load configuration from a specific .env file and configured TOML.
        
        Args:
            env_file: Path to .env file
            
        Returns:
            AgentConfig instance
        """
        return cls.from_env(env_file)
