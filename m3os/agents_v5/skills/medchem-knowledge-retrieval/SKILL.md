---
name: medchem-knowledge-retrieval
description: Use this skill only inside the dedicated MedChem Retrieval Agent to provide medicinal chemistry knowledge for other agents. Always reuse shared medchem memory first, then try fast medchem with list_medchem_sources/read_medchem_sources, then use M3OS wiki, web search, and paper RAG only when narrower curated knowledge is unavailable or insufficient. For any M3OS wiki MCP usage, read the m3os-wiki-mcp-retrieval skill first.
allowed-tools: list_medchem_sources read_medchem_sources answer_question search_articles read_article find_concept get_concept web_search download_research_papers answer_research_paper_question
---

# medchem-knowledge-retrieval

## Overview

Reuse shared medicinal chemistry knowledge memory first. If the current memory does not cover the medicinal-chemistry question, use the curated fast medchem source catalog, then fall back to M3OS wiki when the curated sources do not answer the question. Before using M3OS wiki tools, read [m3os-wiki-mcp-retrieval](../m3os-wiki-mcp-retrieval/SKILL.md) and follow its tool-choice workflow. If wiki knowledge is unavailable, irrelevant, or too broad, use web search as a fast discovery step and paper RAG as a slower evidence step. Return only compact, source-grounded medicinal-chemistry knowledge to the requesting agent.

## Instructions

1. Determine whether extra knowledge is needed
- Use this skill when a requesting agent asks a focused medicinal-chemistry question, or when the request depends on uploaded paper/table evidence.
- Keep retrieval focused on the current molecule, objective, structural motifs, ADMET liabilities, and the specific medicinal-chemistry knowledge gap from the requesting agent.

2. Reuse shared memory first
- Check `[SHARED MEDICINAL CHEMISTRY KNOWLEDGE MEMORY]` before calling knowledge tools.
- If the memory already covers the current scaffold, functional group, property, toxicity risk, or relevant medicinal-chemistry principle, reuse it.
- Search only for missing or more specific information needed to answer the requesting agent's knowledge question.

3. Prefer fast medchem sources
- Call `list_medchem_sources` to inspect the curated source names and descriptions.
- Select source names that directly match the current scaffold, functional group, property, toxicity risk, or medicinal-chemistry concept.
- Call `read_medchem_sources` with the exact selected names to read the markdown content.
- Do not pass `max_chars_per_source`; read full source content.
- Read as many sources as are needed for a well-supported answer, but do not read unrelated sources just to add background.
- Do not call M3OS wiki before this fast medchem check unless the fast medchem tools are unavailable.

4. Fall back to M3OS wiki when needed
- If the fast medchem catalog has no relevant source, or the source content is too narrow for the current question, use M3OS wiki tools.
- Before calling M3OS wiki MCP tools, read [m3os-wiki-mcp-retrieval](../m3os-wiki-mcp-retrieval/SKILL.md).
- Use only the wiki tools covered there: `search_articles`, `find_concept`, `get_concept`, `read_article`, and `answer_question`.
- Do not use `list_articles`, `list_sources`, or `trace_lineage` for normal retrieval. They are intentionally avoided because they are broad, not user-facing, or internal provenance.

5. Use public web and paper RAG only as late fallback or explicit evidence mode
- If M3OS wiki does not cover the needed medicinal chemistry point, call `web_search` before downloading papers. Treat `web_search` as a fast discovery and triage tool: it should identify likely mechanisms, keywords, high-quality reviews, PubMed/Nature/ACS/JMC-style sources, or specific papers worth reading.
- Use `answer_research_paper_question` before downloading new papers only when the current session already appears to have relevant downloaded papers, or when a previous tool call just downloaded papers for this exact question. If the answer says no papers are available, the papers are unrelated, or the evidence is too weak, state that limitation instead of stretching the evidence.
- Call `download_research_papers` only when the user explicitly asks for papers/literature evidence, or when the requesting agent needs stronger primary-literature support than web snippets or wiki can provide.
- For routine fallback, call `download_research_papers` with `max_papers` set to 2 or 3. Use 5 only when the first small batch is insufficient or conflicting. Use 10 only for an explicit deep literature review or when the user asks for broad paper coverage.
- After downloading papers, call `answer_research_paper_question` with a focused question that includes the molecule/property/scaffold/risk and the knowledge gap to answer.
- Prefer paper RAG over web snippets for high-stakes claims about mechanisms, SAR cases, assay outcomes, toxicity mechanisms, or scaffold-specific medicinal-chemistry evidence. Prefer web snippets for quick orientation, source discovery, and deciding whether paper RAG is worth the latency.

6. Report evidence carefully
- State uncertainty when the retrieved source is general, scaffold-mismatched, or only indirectly related.
- Do not invent numerical property changes or causal claims that are not supported by retrieved knowledge, tool outputs, or the molecular context supplied by the requesting agent.

[Note]: Keep the full response under 3000 characters. Please give as detailed and accurate a response as possible.
