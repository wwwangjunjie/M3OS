"""MCP client service for managing Model Context Protocol connections.

This module provides centralized management of MCP server connections
for tools like SMILES to fragments, IUPAC generation, ADMET prediction, etc.
"""

import asyncio
import os
import json
import re
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator
from urllib.parse import urlparse

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from m3os.agents_v5.core.config import MCPConfig


class MCPToolConnectionError(RuntimeError):
    """Raised when an MCP tool still fails after reconnect retry."""


def _disabled_mcp_tool_names() -> set[str]:
    """Return MCP tool names disabled for the current process."""
    raw_value = os.getenv("M3OS_DISABLED_MCP_TOOL_NAMES", "")
    return {
        name
        for name in re.split(r"[\s,]+", raw_value.strip())
        if name
    }


KNOWN_TOOL_SERVER_HINTS = {
    "generate_iupac_name": "mcp_server_iupac_gen",
    "admet_filter_by_admetai": "mcp_server_admet_ai",
    "admet_predict_by_admetai": "mcp_server_admet_ai",
    "admet_filter_by_task_constraints": "mcp_server_admet_ai",
    "nesso_filter_by_cofolding": "mcp_server_nesso_cofolding",
    "nesso_predict_by_cofolding": "mcp_server_nesso_cofolding",
    "transformercpi2_filter_by_cpi": "mcp_server_transformercpi_v2",
    "transformercpi2_filter_by_activity_constraints": "mcp_server_transformercpi_v2",
    "transformercpi2_predict_by_cpi": "mcp_server_transformercpi_v2",
    "setup_generation_Mol2Mol_LinkInvent": "mcp_server_reinvent",
    "generate_similarity_constrained_mol2mol": "mcp_server_reinvent",
    "prepare_smi_input": "mcp_server_reinvent",
    "run_generation": "mcp_server_reinvent",
    "setup_libinvent": "mcp_server_reinvent",
    "prepare_scaffold": "mcp_server_reinvent",
    "health_check": "mcp_server_reinvent",
    "generate_protein_ligand_complex": "mcp_server_boltz_plip",
    "analyze_protein_ligand_interactions": "mcp_server_boltz_plip",
    "list_medchem_sources": "mcp_server_fast_medchem_request",
    "resolve_medchem_source_paths": "mcp_server_fast_medchem_request",
    "read_medchem_sources": "mcp_server_fast_medchem_request",
    "query_evolutionary_optimization_cases": "mcp_server_m3os_mmp",
    "extract_molecule_scaffold": "mcp_server_m3os_mmp",
    "m3os_mmp_health_check": "mcp_server_m3os_mmp",
}


class MCPClientManager:
    """Manager for MCP client connections.
    
    This class provides a centralized way to manage connections to MCP servers
    and retrieve tools from them. It handles the lifecycle of connections
    and provides convenient access to specific tools.
    """
    
    def __init__(self, config: MCPConfig):
        """Initialize the MCP client manager.
        
        Args:
            config: MCP configuration with server endpoints
        """
        self.config = config
        self._client: Optional[MultiServerMCPClient] = None
        self._clients: List[MultiServerMCPClient] = []
        self._tools: Optional[List[Any]] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self.unavailable_servers: Dict[str, str] = {}
        self._reconnect_lock = asyncio.Lock()
        self._restart_lock = asyncio.Lock()
        self._tool_server_names: Dict[str, str] = {}
        self._medchem_tool_cache: Dict[str, Any] = {}
        self._medchem_memory: List[Dict[str, Any]] = []
        self.disabled_tool_names = _disabled_mcp_tool_names()
    
    async def initialize(self) -> None:
        """Initialize MCP clients for all reachable configured servers.

        Each server is probed independently. A failing optional MCP endpoint
        should not prevent the rest of the molecular optimization system from
        starting with the tools that are available.
        """
        server_configs = self.config.get_server_configs()
        if not server_configs:
            self._client = None
            self._clients = []
            self._tools = []
            self.unavailable_servers = {}
            return
        self._ensure_no_proxy_for_http_servers(server_configs)

        self._clients = []
        self._tools = []
        self._tool_server_names = {}
        self.unavailable_servers = {}
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = AsyncExitStack()

        for server_name, server_config in server_configs.items():
            try:
                client, server_stack, tools = await self._open_server_connection(
                    server_name,
                    server_config,
                )
            except Exception as exc:
                initial_message = self._server_error_message(exc)
                recovered = await self._restart_unavailable_server(
                    server_name,
                    server_config,
                    initial_message,
                )
                if recovered:
                    try:
                        client, server_stack, tools = await self._open_server_connection(
                            server_name,
                            server_config,
                        )
                    except Exception as retry_exc:
                        message = self._server_error_message(retry_exc)
                    else:
                        self._register_server_connection(
                            server_name,
                            client,
                            server_stack,
                            tools,
                        )
                        continue
                else:
                    message = initial_message

                self.unavailable_servers[server_name] = message
                print(
                    f"[MCP warning] Skipping unavailable server {server_name}: {message}",
                    flush=True,
                )
                continue

            self._register_server_connection(server_name, client, server_stack, tools)

        self._client = self._clients[0] if self._clients else None

    async def _open_server_connection(
        self,
        server_name: str,
        server_config: Dict[str, Any],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> tuple[MultiServerMCPClient, AsyncExitStack, List[Any]]:
        """Open one configured server and load its tools as an atomic operation."""
        client = MultiServerMCPClient({server_name: server_config})
        server_stack = AsyncExitStack()
        timeout = float(timeout_seconds or self.config.server_init_timeout_seconds)
        try:
            async with asyncio.timeout(timeout):
                session = await server_stack.enter_async_context(client.session(server_name))
                tools = await load_mcp_tools(session, server_name=server_name)
        except BaseException:
            await server_stack.aclose()
            raise
        return client, server_stack, list(tools)

    def _register_server_connection(
        self,
        server_name: str,
        client: MultiServerMCPClient,
        server_stack: AsyncExitStack,
        tools: List[Any],
    ) -> None:
        if self._exit_stack is None:
            raise RuntimeError("MCP exit stack is not initialized")
        self._exit_stack.push_async_callback(server_stack.aclose)
        self._clients.append(client)
        enabled_tools = [
            tool
            for tool in tools
            if str(getattr(tool, "name", "") or "") not in self.disabled_tool_names
        ]
        disabled_tools = sorted(
            {
                str(getattr(tool, "name", "") or "")
                for tool in tools
                if str(getattr(tool, "name", "") or "") in self.disabled_tool_names
            }
        )
        self._tools.extend(enabled_tools)
        self.unavailable_servers.pop(server_name, None)
        for tool in enabled_tools:
            tool_name = str(getattr(tool, "name", "") or "")
            if tool_name:
                self._tool_server_names[tool_name] = server_name
        if disabled_tools:
            print(
                f"[MCP] Disabled configured tools from {server_name}: "
                f"{', '.join(disabled_tools)}",
                flush=True,
            )

    def _server_error_message(self, exc: BaseException) -> str:
        if isinstance(exc, asyncio.TimeoutError):
            return f"initialization timed out after {self.config.server_init_timeout_seconds}s"
        return f"{type(exc).__name__}: {exc}"

    async def _restart_unavailable_server(
        self,
        server_name: str,
        server_config: Dict[str, Any],
        reason: str,
    ) -> bool:
        """Run a configured launcher once and wait for an MCP handshake to succeed."""
        command = self.config.server_restart_commands.get(server_name) or []
        if (
            not self.config.auto_restart_enabled
            or not command
            or str(server_config.get("transport") or "").lower() == "stdio"
        ):
            return False

        async with self._restart_lock:
            async with self._cross_process_restart_lock(server_name):
                probe_ok, _ = await self._probe_server(server_name, server_config)
                if probe_ok:
                    print(
                        f"[MCP recovery] {server_name} became available while waiting "
                        "for the restart lock.",
                        flush=True,
                    )
                    return True

                if await self._server_endpoint_reachable(server_config):
                    print(
                        f"[MCP recovery] {server_name} still accepts TCP connections but "
                        "MCP discovery is busy or unhealthy; waiting without launching a "
                        "duplicate service.",
                        flush=True,
                    )
                    return await self._wait_until_server_ready(
                        server_name,
                        server_config,
                        timeout_seconds=self.config.restart_timeout_seconds,
                    )

                marker = self._restart_marker_path(server_name)
                cooldown = max(0.0, float(self.config.restart_cooldown_seconds))
                if marker.exists() and cooldown:
                    age = max(0.0, time.time() - marker.stat().st_mtime)
                    remaining = max(0.0, cooldown - age)
                    if remaining:
                        print(
                            f"[MCP recovery] A recent {server_name} restart was detected; "
                            f"waiting up to {remaining:.1f}s before relaunching.",
                            flush=True,
                        )
                        if await self._wait_until_server_ready(
                            server_name,
                            server_config,
                            timeout_seconds=remaining,
                        ):
                            return True

                print(
                    f"[MCP recovery] {server_name} is unavailable ({reason}); "
                    "running its configured restart command.",
                    flush=True,
                )
                launched = await self._run_restart_command(server_name, command)
                marker.touch(exist_ok=True)
                if not launched:
                    return False

                recovered = await self._wait_until_server_ready(
                    server_name,
                    server_config,
                    timeout_seconds=self.config.restart_timeout_seconds,
                )
                if recovered:
                    print(
                        f"[MCP recovery] {server_name} restarted and passed MCP tool discovery.",
                        flush=True,
                    )
                else:
                    print(
                        f"[MCP recovery] {server_name} did not recover within "
                        f"{self.config.restart_timeout_seconds}s.",
                        flush=True,
                    )
                return recovered

    async def _run_restart_command(self, server_name: str, command: List[str]) -> bool:
        """Execute a trusted argv-style launcher without invoking a shell."""
        executable = Path(command[0])
        if (os.sep in command[0] or command[0].startswith(".")) and not executable.exists():
            print(
                f"[MCP recovery] Restart launcher for {server_name} does not exist: {executable}",
                flush=True,
            )
            return False
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                print(
                    f"[MCP recovery] Restart launcher for {server_name} is still running; "
                    "continuing with readiness checks.",
                    flush=True,
                )
                return True
        except Exception as exc:
            print(
                f"[MCP recovery] Could not launch {server_name}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return False

        if process.returncode != 0:
            detail = (stderr or stdout or b"").decode("utf-8", errors="replace").strip()
            if len(detail) > 1000:
                detail = detail[-1000:]
            print(
                f"[MCP recovery] Restart launcher for {server_name} exited with "
                f"code {process.returncode}: {detail}",
                flush=True,
            )
            return False
        return True

    async def _wait_until_server_ready(
        self,
        server_name: str,
        server_config: Dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> bool:
        timeout = max(0.0, float(timeout_seconds))
        if timeout == 0:
            ok, _ = await self._probe_server(server_name, server_config)
            return ok
        deadline = time.monotonic() + timeout
        poll_interval = max(0.1, float(self.config.restart_poll_interval_seconds))
        while True:
            ok, _ = await self._probe_server(server_name, server_config)
            if ok:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(poll_interval, remaining))

    async def _probe_server(
        self,
        server_name: str,
        server_config: Dict[str, Any],
    ) -> tuple[bool, str]:
        probe_timeout = min(
            max(1.0, float(self.config.restart_poll_interval_seconds) * 2.0),
            max(1.0, float(self.config.server_init_timeout_seconds)),
        )
        try:
            _, server_stack, _ = await self._open_server_connection(
                server_name,
                server_config,
                timeout_seconds=probe_timeout,
            )
        except Exception as exc:
            return False, self._server_error_message(exc)
        await server_stack.aclose()
        return True, ""

    async def _server_endpoint_reachable(self, server_config: Dict[str, Any]) -> bool:
        """Return whether an HTTP MCP endpoint still has a listening TCP socket."""
        url = str(server_config.get("url") or "").strip()
        if not url:
            return False
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            async with asyncio.timeout(2.0):
                _, writer = await asyncio.open_connection(host, port)
        except (OSError, asyncio.TimeoutError):
            return False
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        return True

    def _restart_marker_path(self, server_name: str) -> Path:
        lock_dir = Path(self.config.restart_lock_dir).expanduser()
        lock_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", server_name)
        return lock_dir / f"{safe_name}.last_restart"

    @asynccontextmanager
    async def _cross_process_restart_lock(self, server_name: str) -> AsyncGenerator[None, None]:
        """Serialize restarts across parallel benchmark worker processes."""
        lock_dir = Path(self.config.restart_lock_dir).expanduser()
        lock_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", server_name)
        lock_path = lock_dir / f"{safe_name}.lock"
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            try:
                import fcntl
            except ImportError:
                yield
                return
            await asyncio.to_thread(fcntl.flock, handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                await asyncio.to_thread(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _ensure_no_proxy_for_http_servers(self, server_configs: Dict[str, Dict[str, Any]]) -> None:
        """Keep private MCP endpoints from being routed through HTTP proxies."""
        hosts: List[str] = []
        for config in server_configs.values():
            url = config.get("url") if isinstance(config, dict) else None
            if not url:
                continue
            host = urlparse(str(url)).hostname
            if host:
                hosts.append(host)
        if not hosts:
            return

        values: List[str] = []
        existing: set[str] = set()
        for raw in (os.getenv("NO_PROXY", ""), os.getenv("no_proxy", "")):
            for item in raw.split(","):
                value = item.strip()
                if value and value not in existing:
                    values.append(value)
                    existing.add(value)
        if "*" in values:
            return
        changed = False
        for host in hosts:
            if host not in existing:
                values.append(host)
                existing.add(host)
                changed = True
        if changed:
            no_proxy = ",".join(values)
            os.environ["NO_PROXY"] = no_proxy
            os.environ["no_proxy"] = no_proxy
    
    async def close(self) -> None:
        """Close all MCP client connections."""
        clients = self._clients or ([self._client] if self._client else [])
        for client in clients:
            # Current `langchain_mcp_adapters.MultiServerMCPClient` opens
            # per-call sessions and does not expose a close/aclose method.
            # Keep this compatible with older/newer versions that may add one.
            close_method = getattr(client, "close", None)
            aclose_method = getattr(client, "aclose", None)
            if callable(aclose_method):
                await aclose_method()
            elif callable(close_method):
                maybe_result = close_method()
                if hasattr(maybe_result, "__await__"):
                    await maybe_result
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._client = None
        self._clients = []
        self._tools = None
        self._tool_server_names = {}
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator["MCPClientManager", None]:
        """Get a session context for the MCP client.
        
        Yields:
            Self for tool access
        """
        if not self._client:
            await self.initialize()
        
        try:
            yield self
        except Exception:
            # Don't close on exception, let the manager handle lifecycle
            raise
    
    def get_all_tools(self) -> List[Any]:
        """Get all tools from all configured MCP servers.
        
        Returns:
            List of all LangChain tools from all servers
            
        Raises:
            ValueError: If client not initialized
        """
        if self._tools is None:
            raise ValueError("MCP client not initialized")
        return self._tools
    
    def get_tool(self, tool_name: str) -> Any:
        """Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            The requested tool
            
        Raises:
            ValueError: If tool not found
        """
        if self._tools is None:
            raise ValueError("MCP client not initialized")
            
        tool = next((t for t in self._tools if t.name == tool_name), None)
        
        if tool is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        return tool

    def has_tool(self, tool_name: str) -> bool:
        """Return whether the current MCP tool registry contains a tool name."""
        if self._tools is None:
            return False
        return any(getattr(tool, "name", "") == tool_name for tool in self._tools)

    def is_connection_error(self, exc: BaseException) -> bool:
        """Best-effort detection for closed/stale MCP transports."""
        current: BaseException | None = exc
        while current is not None:
            name = type(current).__name__
            message = str(current)
            if name in {
                "ClosedResourceError",
                "BrokenResourceError",
                "EndOfStream",
                "ClosedStreamError",
                "StreamClosed",
                "ConnectError",
                "ReadError",
                "WriteError",
                "RemoteProtocolError",
                "MCPToolConnectionError",
            }:
                return True
            lowered = message.lower()
            if (
                "closedresourceerror" in lowered
                or "closed resource" in lowered
                or "stream is closed" in lowered
                or "connection closed" in lowered
                or "connection error" in lowered
                or "connection failed" in lowered
                or "connection reset" in lowered
                or "broken pipe" in lowered
            ):
                return True
            current = current.__cause__ or current.__context__
        return False

    def _server_name_for_tool(
        self,
        tool_name: str,
        server_hint: Optional[str],
    ) -> Optional[str]:
        server_configs = self.config.get_server_configs()
        if server_hint and server_hint in server_configs:
            return server_hint
        discovered = self._tool_server_names.get(tool_name)
        if discovered:
            return discovered
        known = KNOWN_TOOL_SERVER_HINTS.get(tool_name)
        if known in server_configs:
            return known
        return None

    def _is_missing_tool_for_server(
        self,
        exc: BaseException,
        server_name: Optional[str],
    ) -> bool:
        if not server_name or not isinstance(exc, ValueError):
            return False
        return "tool " in str(exc).lower() and " not found" in str(exc).lower()

    async def _invoke_tool_on_isolated_connection(
        self,
        server_name: str,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Any:
        """Invoke one tool on a short-lived, task-local MCP session.

        MCP streamable-HTTP sessions use AnyIO cancel scopes, which must be
        entered and exited by the same asyncio task.  Runtime recovery happens
        inside LangGraph's ``StructuredTool._arun`` task, so it must not close
        the persistent sessions created by the outer benchmark task.  Opening,
        invoking, and closing an isolated session here preserves that ownership.
        """
        server_config = self.config.get_server_configs().get(server_name)
        if server_config is None:
            raise ValueError(f"MCP server {server_name} is not configured")

        _, server_stack, tools = await self._open_server_connection(
            server_name,
            server_config,
        )
        try:
            tool = next(
                (
                    item
                    for item in tools
                    if str(getattr(item, "name", "") or "") == tool_name
                ),
                None,
            )
            if tool is None:
                raise ValueError(f"Tool {tool_name} not found on {server_name}")
            return await asyncio.wait_for(
                tool.ainvoke(params),
                timeout=self.config.tool_timeout_seconds,
            )
        finally:
            # This stack was entered in the current StructuredTool task, so it
            # is safe to close here without violating AnyIO cancel-scope order.
            await server_stack.aclose()

    async def _invoke_tool_with_isolated_recovery(
        self,
        tool_name: str,
        params: Dict[str, Any],
        exc: BaseException,
        server_name: str,
    ) -> Any:
        """Retry a failed tool without mutating shared MCP connections."""
        async with self._reconnect_lock:
            print(
                f"[MCP warning] {tool_name} connection failed with "
                f"{type(exc).__name__}: {exc}; retrying {server_name} through "
                "an isolated MCP session.",
                flush=True,
            )
            try:
                result = await self._invoke_tool_on_isolated_connection(
                    server_name,
                    tool_name,
                    params,
                )
            except asyncio.TimeoutError as retry_exc:
                message = (
                    f"MCP tool {tool_name} timed out after "
                    f"{self.config.tool_timeout_seconds}s during isolated retry"
                )
                print(f"[MCP error] {message}", flush=True)
                raise TimeoutError(message) from retry_exc
            except Exception as retry_exc:
                server_config = self.config.get_server_configs().get(server_name)
                recovered = False
                if server_config is not None:
                    recovered = await self._restart_unavailable_server(
                        server_name,
                        server_config,
                        self._server_error_message(retry_exc),
                    )
                if not recovered:
                    message = (
                        f"MCP tool {tool_name} connection failed during isolated retry: "
                        f"{type(retry_exc).__name__}: {retry_exc}"
                    )
                    print(f"[MCP error] {message}", flush=True)
                    raise MCPToolConnectionError(message) from retry_exc
                try:
                    result = await self._invoke_tool_on_isolated_connection(
                        server_name,
                        tool_name,
                        params,
                    )
                except asyncio.TimeoutError as final_exc:
                    message = (
                        f"MCP tool {tool_name} timed out after "
                        f"{self.config.tool_timeout_seconds}s after server recovery"
                    )
                    print(f"[MCP error] {message}", flush=True)
                    raise TimeoutError(message) from final_exc
                except Exception as final_exc:
                    message = (
                        f"MCP tool {tool_name} connection failed after isolated "
                        f"recovery: {type(final_exc).__name__}: {final_exc}"
                    )
                    print(f"[MCP error] {message}", flush=True)
                    raise MCPToolConnectionError(message) from final_exc

            self.unavailable_servers.pop(server_name, None)
            print(
                f"[MCP] {tool_name} completed through isolated recovery",
                flush=True,
            )
            return result
    
    async def invoke_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        server_hint: Optional[str] = None
    ) -> Any:
        """Invoke a tool with the given parameters.

        Args:
            tool_name: Name of the tool to invoke
            params: Parameters to pass to the tool
            server_hint: Optional configured MCP server name. This is used for
                targeted recovery when tool discovery or the transport fails.

        Returns:
            Tool invocation result
        """
        if tool_name in self.disabled_tool_names:
            raise ValueError(f"Tool {tool_name} is disabled")
        if self._tools is None:
            await self.initialize()
        params = self._normalize_medchem_tool_params(tool_name, params)
        cache_key = self._medchem_cache_key(tool_name, params)
        if cache_key and cache_key in self._medchem_tool_cache:
            print(f"[MCP cache] Reusing {tool_name} result", flush=True)
            return self._medchem_tool_cache[cache_key]
        timeout = self.config.tool_timeout_seconds
        last_connection_error: BaseException | None = None
        server_name = self._server_name_for_tool(tool_name, server_hint)
        for attempt in range(2):
            print(f"[MCP] Calling {tool_name} with timeout={timeout}s", flush=True)
            try:
                tool = self.get_tool(tool_name)
                result = await asyncio.wait_for(tool.ainvoke(params), timeout=timeout)
                print(f"[MCP] {tool_name} completed", flush=True)
                self._remember_medchem_tool_result(tool_name, params, result, cache_key)
                return result
            except asyncio.TimeoutError as exc:
                message = f"MCP tool {tool_name} timed out after {timeout}s"
                print(f"[MCP error] {message}", flush=True)
                raise TimeoutError(message) from exc
            except Exception as exc:
                recoverable = self.is_connection_error(exc) or self._is_missing_tool_for_server(
                    exc,
                    server_name,
                )
                if recoverable and attempt == 0:
                    last_connection_error = exc
                    if server_name is None:
                        message = (
                            f"MCP tool {tool_name} has no configured server for "
                            "isolated recovery"
                        )
                        raise MCPToolConnectionError(message) from exc
                    result = await self._invoke_tool_with_isolated_recovery(
                        tool_name,
                        params,
                        exc,
                        server_name,
                    )
                    self._remember_medchem_tool_result(tool_name, params, result, cache_key)
                    return result
                if tool_name == "smiles_to_fragments_str":
                    print(
                        f"[MCP warning] {tool_name} unavailable, using local fallback: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    from m3os.agents_v5.tools.fragments import smiles2fragments_str

                    return smiles2fragments_str(params.get("smiles_list", []))
                if tool_name == "validate_smiles_batch":
                    print(
                        f"[MCP warning] {tool_name} unavailable, using local fallback: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    from m3os.agents_v5.services.smiles_validation import (
                        validate_smiles_batch_messages,
                    )

                    return validate_smiles_batch_messages(params.get("smiles_list", []))
                if last_connection_error is not None or recoverable:
                    message = (
                        f"MCP tool {tool_name} connection failed after reconnect retry: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(f"[MCP error] {message}", flush=True)
                    raise MCPToolConnectionError(message) from exc
                print(f"[MCP error] {tool_name} failed: {type(exc).__name__}: {exc}", flush=True)
                raise

        raise MCPToolConnectionError(f"MCP tool {tool_name} connection failed after reconnect retry.")

    def format_medchem_memory(self, *, max_items: int = 8, max_chars: int = 14000) -> str:
        """Return reusable medicinal-chemistry knowledge retrieved in this session."""
        if not self._medchem_memory:
            return ""
        recent = self._medchem_memory[-max(1, int(max_items)) :]
        sections = [
            "[SHARED MEDICINAL CHEMISTRY KNOWLEDGE MEMORY]",
            (
                "Reuse these retrieved medicinal-chemistry facts before calling "
                "knowledge tools again. Only search for information that is not "
                "covered here or is too narrow for the current decision."
            ),
        ]
        for index, item in enumerate(recent, start=1):
            sections.append(
                "\n".join(
                    [
                        f"Memory {index}: {item.get('tool', '')}",
                        f"Query/params: {item.get('params', '')}",
                        "Content:",
                        str(item.get("content") or ""),
                    ]
                ).strip()
            )
        text = "\n\n".join(sections).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n...[shared medchem memory shortened for prompt]"

    def _normalize_medchem_tool_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(params or {})
        if tool_name == "read_medchem_sources":
            normalized.pop("max_chars_per_source", None)
        return normalized

    def _medchem_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        if tool_name not in MEDCHEM_MEMORY_TOOL_NAMES:
            return ""
        try:
            payload = json.dumps(params or {}, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            payload = str(params or {})
        return f"{tool_name}:{payload}"

    def _remember_medchem_tool_result(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        cache_key: str,
    ) -> None:
        if tool_name not in MEDCHEM_MEMORY_TOOL_NAMES:
            return
        if cache_key:
            self._medchem_tool_cache[cache_key] = result
        if tool_name not in MEDCHEM_PROMPT_MEMORY_TOOL_NAMES:
            return
        content = self._summarize_medchem_result_for_memory(tool_name, result)
        if not content:
            return
        params_text = self._compact_memory_text(
            json.dumps(params or {}, ensure_ascii=False, sort_keys=True, default=str),
            600,
        )
        memory_key = f"{tool_name}:{params_text}"
        self._medchem_memory = [
            item for item in self._medchem_memory if item.get("key") != memory_key
        ]
        self._medchem_memory.append(
            {
                "key": memory_key,
                "tool": tool_name,
                "params": params_text,
                "content": content,
            }
        )
        self._medchem_memory = self._medchem_memory[-24:]

    def _summarize_medchem_result_for_memory(self, tool_name: str, result: Any) -> str:
        if isinstance(result, dict):
            if tool_name == "read_medchem_sources":
                parts: List[str] = []
                for source in result.get("sources") or []:
                    if not isinstance(source, dict):
                        continue
                    parts.append(
                        "\n".join(
                            [
                                f"Source: {source.get('name') or ''}",
                                f"Description: {source.get('description') or ''}",
                                f"Path: {source.get('path') or ''}",
                                str(source.get("content") or ""),
                            ]
                        ).strip()
                    )
                return self._compact_memory_text("\n\n".join(parts), 8000)
            for key in ("answer", "content", "body", "text"):
                if result.get(key):
                    return self._compact_memory_text(str(result.get(key)), 8000)
            return self._compact_memory_text(json.dumps(result, ensure_ascii=False, default=str), 8000)
        return self._compact_memory_text(str(result or ""), 8000)

    @staticmethod
    def _compact_memory_text(text: str, max_chars: int) -> str:
        text = str(text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n...[memory content shortened for prompt; exact tool result is cached]"
    
    def get_tools_by_names(self, target_names: set) -> List[Any]:
        """Get tools filtered by their names.
        
        Args:
            target_names: Set of tool names to retrieve
            
        Returns:
            List of matching tools
        """
        if self._tools is None:
            raise ValueError("MCP client not initialized")
        return [tool for tool in self._tools if tool.name in target_names]
    
    def get_all_agent_tools(self) -> List[Any]:
        """Get all tools needed for the agents.
        
        This method retrieves tools from all configured servers that are
        relevant for the molecular optimization agents.
        
        Returns:
            List of LangChain tools
        """
        if self._tools is None:
            raise ValueError("MCP client not initialized")
        return self._tools.copy()


WIKI_USABLE_TOOL_NAMES = {
    "read_article",
    "find_concept",
    "search_articles",
    "get_concept",
    "answer_question",
}

WIKI_RAW_SOURCE_TOOL_NAMES = {
    "search_source_segments",
    "get_source_passages",
    "read_source_segment",
    "list_segments",
}

WIKI_TOOL_NAMES = WIKI_USABLE_TOOL_NAMES

FAST_MEDCHEM_TOOL_NAMES = {
    "list_medchem_sources",
    "read_medchem_sources",
}

RESEARCH_LITERATURE_TOOL_NAMES = {
    "web_search",
    "download_research_papers",
    "answer_research_paper_question",
}

MAIN_CONTEXT_LOOKUP_TOOL_NAMES = {
    "get_protein_uniprot_ids",
    "get_protein_sequence",
    "get_molecule_smiles",
}

# MOLECULE_IMAGE_TOOL_NAMES = {
#     "generate_molecule_topology_image",
#     "generate_molecule_difference_image",
#     "generate_molecule_pair_difference_image",
# }

MOLECULE_SCAFFOLD_TOOL_NAMES = {
    "extract_molecule_scaffold",
}

EVOLUTION_CASE_TOOL_NAMES = {
    "query_evolutionary_optimization_cases",
}

SMILES_VALIDATION_TOOL_NAMES = {
    "validate_smiles_batch",
}

PROTEIN_CONTEXT_TOOL_NAMES = {
    "generate_protein_ligand_complex",
    "analyze_protein_ligand_interactions",
}

NESSO_ACTIVITY_TOOL_NAMES = {
    "nesso_filter_by_cofolding",
    "nesso_predict_by_cofolding",
}

TRANSFORMERCPI2_ACTIVITY_TOOL_NAMES = {
    "transformercpi2_filter_by_cpi",
    "transformercpi2_filter_by_activity_constraints",
    "transformercpi2_predict_by_cpi",
}

MEDCHEM_MEMORY_TOOL_NAMES = (
    WIKI_TOOL_NAMES | FAST_MEDCHEM_TOOL_NAMES | RESEARCH_LITERATURE_TOOL_NAMES
)
MEDCHEM_RETRIEVAL_TOOL_NAMES = MEDCHEM_MEMORY_TOOL_NAMES
MEDCHEM_PROMPT_MEMORY_TOOL_NAMES = {
    "read_medchem_sources",
    "answer_question",
    "read_article",
    "find_concept",
    "get_concept",
    "web_search",
    "answer_research_paper_question",
}


# Tool name sets for filtering tools for specific agents
TOOL_NAMES_CREATIVE = {
    'generate_similarity_constrained_mol2mol',
    'setup_generation_Mol2Mol_LinkInvent',
    'prepare_smi_input',
    'run_generation',
    'setup_libinvent',
    'prepare_scaffold',
    'admet_filter_by_admetai',
    'nesso_filter_by_cofolding',
    'screen_reinvent_candidates_by_intersection',
} | SMILES_VALIDATION_TOOL_NAMES

TOOL_NAMES_RATIONAL = (
    PROTEIN_CONTEXT_TOOL_NAMES
    | EVOLUTION_CASE_TOOL_NAMES
    | MOLECULE_SCAFFOLD_TOOL_NAMES
    | SMILES_VALIDATION_TOOL_NAMES
)

TOOL_NAMES_CRITIC = {
    'admet_predict_by_admetai',
    'evaluate_candidate_constraints',
    'evaluate_nesso_activity',
} | SMILES_VALIDATION_TOOL_NAMES

TOOL_NAMES_MCP = {
    "smiles_to_fragments_str",
    "generate_iupac_name",
} | MAIN_CONTEXT_LOOKUP_TOOL_NAMES


def filter_tools_by_name(tools: List[Any], target_names: set) -> List[Any]:
    """Filter tools by their names.
    
    Args:
        tools: List of tools to filter
        target_names: Set of tool names to keep
        
    Returns:
        Filtered list of tools
    """
    return [tool for tool in tools if tool.name in target_names]
