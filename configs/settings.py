"""Shared TOML and dotenv configuration helpers for M3OS."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT_DIR / ".env"
DEFAULT_CONFIG_FILE = ROOT_DIR / "configs" / "m3os.default.toml"
DEFAULT_LOCAL_CONFIG_FILE = ROOT_DIR / "configs" / "m3os.local.toml"


_CONFIG_ENV_MAP: dict[str, str] = {
    "LLM_PROVIDER": "llm.provider",
    "LLM_REQUEST_TIMEOUT_SECONDS": "llm.request_timeout_seconds",
    "CLAUDE_MODEL": "llm.claude_model",
    "GEMINI_MODEL": "llm.gemini_model",
    "KIMI_MODEL": "llm.kimi_model",
    "KIMI_REASONING_EFFORT": "llm.kimi_reasoning_effort",
    "KIMI_MAX_RETRIES": "llm.kimi_max_retries",
    "KIMI_REQUEST_TIMEOUT_SECONDS": "llm.kimi_request_timeout_seconds",
    "OPENAI_MODEL": "llm.openai_model",
    "EMBEDDING_MODEL": "llm.embedding_model",
    "EXPERT_MODEL": "expert.model",
    "EXPERT_MODEL_TIMEOUT_SECONDS": "expert.timeout_seconds",
    "MCP_SERVER_PATH": "mcp.main_server_path",
    "MCP_SERVER_PYTHON": "mcp.main_server_python",
    "MCP_SERVER_URL": "mcp.main_server_url",
    "MCP_SERVER_TRANSPORT": "mcp.main_server_transport",
    "MCP_SERVER_INIT_TIMEOUT_SECONDS": "mcp.server_init_timeout_seconds",
    "MCP_TOOL_TIMEOUT_SECONDS": "mcp.tool_timeout_seconds",
    "MCP_AUTO_RESTART_ENABLED": "mcp.auto_restart_enabled",
    "MCP_RESTART_TIMEOUT_SECONDS": "mcp.restart_timeout_seconds",
    "MCP_RESTART_POLL_INTERVAL_SECONDS": "mcp.restart_poll_interval_seconds",
    "MCP_RESTART_COOLDOWN_SECONDS": "mcp.restart_cooldown_seconds",
    "MCP_RESTART_LOCK_DIR": "mcp.restart_lock_dir",
    "MCP_MAIN_SERVER_RESTART_COMMAND": "mcp.main_server_restart_command",
    "MCP_SERVER_ADMET_AI_RESTART_COMMAND": "mcp.admet_ai_restart_command",
    "MCP_SERVER_BOLTZ_RESTART_COMMAND": "mcp.boltz_restart_command",
    "MCP_SERVER_IUPAC_GEN_RESTART_COMMAND": "mcp.iupac_gen_restart_command",
    "MCP_SERVER_REINVENT_RESTART_COMMAND": "mcp.reinvent_restart_command",
    "M3OS_FAST_MEDCHEM_MCP_RESTART_COMMAND": "mcp.fast_medchem_restart_command",
    "M3OS_MMP_MCP_RESTART_COMMAND": "mcp.mmp_restart_command",
    "M3OS_WIKI_MCP_RESTART_COMMAND": "mcp.m3os_wiki_restart_command",
    "PROPERTY_META_INFO_PATH": "paths.property_meta_info_path",
    "M3OS_TMP_DIR": "paths.tmp_dir",
    "RAG_DATA_DIR": "paths.rag_data_dir",
    "DEEPAGENTS_BACKEND_ROOT_DIR": "paths.deepagents_backend_root_dir",
    "DEEPAGENTS_SKILLS_ROOT": "paths.deepagents_skills_root",
    "RAG_CHUNK_SIZE": "rag.chunk_size",
    "RAG_CHUNK_OVERLAP": "rag.chunk_overlap",
    "RAG_TOP_K": "rag.top_k",
    "RAG_FULL_TEXT_MAX_CHARS": "rag.full_text_max_chars",
    "VECTOR_DB_PATH": "rag.vector_db_path",
    "UNIPROT_NUM_IDS": "retrieval.uniprot_num_ids",
    "MAX_PAPERS": "retrieval.max_papers",
    "PAPER_DIR": "retrieval.paper_dir",
    "UNIPROT_BASE_URL": "retrieval.urls.uniprot_base_url",
    "CHEMBL_BASE_URL": "retrieval.urls.chembl_base_url",
    "SEMANTIC_SCHOLAR_SEARCH_URL": "retrieval.urls.semantic_scholar_search_url",
    "PUBMED_SEARCH_URL": "retrieval.urls.pubmed_search_url",
    "CSV_SAVE_PATH_PREFIX": "mcgs.csv_save_path_prefix",
    "CSV_SAVE_PATH_SUFFIX": "mcgs.csv_save_path_suffix",
    "M3OS_RECURSION_LIMIT": "runtime.recursion_limit",
    "M3OS_SESSION_IDLE_SECONDS": "runtime.session_idle_seconds",
    "M3OS_GENERATOR_MODE": "runtime.generator_mode",
    "M3OS_MAX_AUTO_EXPANSION_ROUNDS": "runtime.max_auto_expansion_rounds",
    "M3OS_MAX_NO_PROGRESS_ROUNDS": "runtime.max_no_progress_rounds",
    "M3OS_STALE_SECONDS": "runtime.stale_seconds",
    "ENABLE_SHARED_ANALYSIS": "runtime.enable_shared_analysis",
}


def resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def resolve_config_path(value: Any, default: str | os.PathLike[str] = "") -> str:
    text = str(value if value is not None else default).strip()
    if not text:
        return text
    return str(resolve_project_path(text).resolve())


def resolve_intermediate_path(
    settings: Mapping[str, Any],
    env_name: str,
    config_path: str,
    default: str,
) -> str:
    value = get_setting(settings, env_name, config_path, default)
    text = str(value if value is not None else default).strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    tmp_value = get_setting(settings, "M3OS_TMP_DIR", "paths.tmp_dir", "tmp")
    tmp_path = resolve_project_path(str(tmp_value or "tmp"))
    if path.parts and path.parts[0] == "tmp":
        return str(tmp_path.joinpath(*path.parts[1:]).resolve())
    return str((tmp_path / path).resolve())


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def get_config_value(settings: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = settings
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def get_setting(
    settings: Mapping[str, Any],
    env_name: str,
    config_path: str,
    default: Any = None,
) -> Any:
    value = os.getenv(env_name)
    if value is not None:
        return value
    return get_config_value(settings, config_path, default)


def _stringify_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def apply_config_to_environ(settings: Mapping[str, Any], *, overwrite: bool = False) -> None:
    for env_name, config_path in _CONFIG_ENV_MAP.items():
        if not overwrite and env_name in os.environ:
            continue
        value = get_config_value(settings, config_path)
        if value is None:
            continue
        os.environ[env_name] = _stringify_env_value(value)


def _resolve_optional_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return resolve_project_path(text)


def load_settings(
    env_file: str | os.PathLike[str] | None = None,
    *,
    load_default_env: bool = False,
    use_default_config: bool = False,
    dotenv_override: bool = True,
    apply_env: bool = True,
) -> dict[str, Any]:
    """Load dotenv and TOML settings.

    Without ``env_file`` this keeps tests lightweight: it only loads the
    process/default ``.env`` and a TOML file explicitly named by
    ``M3OS_CONFIG_FILE``. Runtime entry points can pass the repository-local
    ``.env`` file, which in turn selects the structured TOML defaults.
    """

    if env_file is not None:
        env_path = resolve_project_path(env_file)
        if env_path.exists():
            load_dotenv(env_path, override=dotenv_override)
    elif load_default_env:
        env_path = _resolve_optional_path(os.getenv("M3OS_ENV_FILE")) or DEFAULT_ENV_FILE
        if env_path.exists():
            load_dotenv(env_path, override=dotenv_override)
    else:
        load_dotenv(override=dotenv_override)

    config_path = _resolve_optional_path(os.getenv("M3OS_CONFIG_FILE"))
    if config_path is None and use_default_config:
        config_path = DEFAULT_CONFIG_FILE

    settings: dict[str, Any] = {}
    if config_path is not None:
        settings = _deep_merge(settings, _read_toml(config_path))

    local_path = _resolve_optional_path(os.getenv("M3OS_CONFIG_LOCAL_FILE"))
    if local_path is None and config_path is not None:
        local_path = DEFAULT_LOCAL_CONFIG_FILE
    if local_path is not None and local_path.exists():
        settings = _deep_merge(settings, _read_toml(local_path))

    if apply_env:
        apply_config_to_environ(settings)
    return settings
