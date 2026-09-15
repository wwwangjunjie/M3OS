# M3OS

M3OS is a research framework for multi-agent molecular optimization. It combines
specialized language-model agents, Monte Carlo graph search (MCGS), and external Model Context Protocol (MCP) tools to propose and rank lead-optimization candidates.


## Highlights

- Role-specialized Creative, Rational, Critic, Auditor, and Main agents.
- Search-state management with an explicit molecular optimization graph.
- SMILES validity gates, candidate deduplication, and auditable event traces.
- Optional protein, structure, table, document, and medicinal-chemistry context.
- MCP integration for generation, ADMET, affinity, structure, and naming tools.
- Multi-turn Python API with both regular and streaming responses.

## Workflow

```text
optimization request + lead molecule + target context
                         |
                  task preparation
                         |
        +----------------+----------------+
        |                                 |
  Creative Explorer                Rational Designer
        |                                 |
        +---------- candidate pool -------+
                         |
                  Critic + Auditor
                         |
               MCGS graph update/selection
                         |
             repeat until stopping criteria
                         |
                 report and candidates
```

## Repository layout

```text
M3OS/
├── m3os/agents_v5/
│   ├── agents/       # Agent implementations
│   ├── core/         # Configuration, state, tracing, and data models
│   ├── prompts/      # Versioned system and runtime prompts
│   ├── services/     # LLM, MCP, retrieval, document, and report services
│   ├── skills/       # Agent skills and supporting references
│   ├── tools/        # Molecular utilities and MCGS implementation
│   └── workflow/     # Search workflow and graph nodes
├── configs/
│   ├── m3os.default.toml  # Checked-in, non-sensitive defaults
│   └── settings.py        # TOML/dotenv configuration loader
├── mcp_server_improved.py # Core M3OS MCP tools
├── open_mcp_server.sh     # Core MCP launcher
├── pyproject.toml
└── uv.lock
```

External model servers live in the companion `M3OS_MCP` repository. Third-party
source trees, checkpoints, caches, environments, experiment outputs, and private
services are intentionally not included in either repository.

## Requirements

- Linux
- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) for the locked environment
- An API key for one supported LLM provider
- Optional NVIDIA GPU(s) for model-backed MCP services

Some optional dependencies and external services have their own licenses and
hardware requirements. Review those terms before installation.

## Installation

```bash
cd M3OS
uv sync --frozen
cp .env.example .env
```

Edit `.env` with your provider credentials and the MCP endpoints you intend to
run. 

The checked-in `uv.lock` is the reproducible environment record. Use
`uv sync --frozen` for paper reproduction and update the lock deliberately when
changing dependencies.

## Configuration

Configuration is loaded in this order:

1. `configs/m3os.default.toml` provides non-sensitive defaults.
2. Environment variables and `.env` override both TOML files.

Start from `.env.example`. At minimum, configure one of the following provider
groups:

- `KIMI_API_KEY`, `KIMI_BASE_URL`, and `KIMI_MODEL`
- `CLAUDE_API_KEY`, `CLAUDE_BASE_URL`, and `CLAUDE_MODEL`
- `GEMINI_API_KEY`, `GEMINI_BASE_URL`, and `GEMINI_MODEL`

Set `LLM_PROVIDER` to the corresponding provider name. Retrieval and embedding
features additionally use the OpenAI-compatible variables shown in the example
file.

The two private services used in internal deployments are not distributed:
`M3OS_fast_medchem_request` and `M3OS_MMP`. The public default configuration
does not require them.

## MCP services

Install and start the desired services from the companion `M3OS_MCP` repository,
then enable their URLs in `.env`. The public service ports are:

| Service | Default endpoint | Purpose |
| --- | --- | --- |
| M3OS core | `http://127.0.0.1:8010/mcp` | Retrieval and molecular utilities |
| ADMET-AI | `http://127.0.0.1:8000/mcp` | ADMET prediction and filtering |
| IUPAC | `http://127.0.0.1:8020/mcp` | SMILES-to-IUPAC conversion |
| Boltz/PLIP | `http://127.0.0.1:8040/mcp` | Complex prediction and interaction analysis |
| REINVENT | `http://127.0.0.1:8041/mcp` | Molecular generation |
| Nesso | `http://127.0.0.1:8051/mcp` | Protein-ligand affinity scoring |

Start the core server after installing the main environment:

```bash
./open_mcp_server.sh
```

Runtime logs and PID files are written below `tmp/main_mcp/` and are ignored by
Git.

## Python API

```python
import asyncio

from m3os.agents_v5 import create_main_agent_session


async def main() -> None:
    session = await create_main_agent_session(
        env_file=".env",
        protein_fasta_path="examples/target.fasta",
        enabled_generators=["creative", "rational"],
    )
    try:
        answer = await session.ask(
            "Optimize the lead for potency while retaining the stated ADMET constraints."
        )
        print(answer)
    finally:
        await session.close()


asyncio.run(main())
```

`M3OSChatSession.ask_stream()` yields frontend-friendly progress, tool, graph,
and final-answer events. `get_graph_snapshot()` returns the current search graph
for visualization or analysis.


## Testing

This compact public snapshot does not include the internal end-to-end benchmark
suite. Run the import/configuration smoke test below, then run the unit tests in
each enabled `M3OS_MCP` service as documented by that service.

```bash
uv run --frozen python -c \
  'from m3os.agents_v5 import M3OSChatSession, create_main_agent_session; print("M3OS import OK")'
```

End-to-end runs that call LLMs, external databases, model weights, or GPU MCP
services require the corresponding credentials and services. Keep new unit tests
independent of private data and machine-specific paths.

## Citation




