---
name: m3os-wiki-mcp-retrieval
description: Use when an M3OS or medicinal-chemistry retrieval agent needs to choose the right M3OS wiki MCP tool for compact knowledge retrieval. Covers search_articles, find_concept, get_concept, read_article, and answer_question; avoids list_articles, list_sources, trace_lineage, and raw source-segment tools.
---

# m3os-wiki-mcp-retrieval

## Overview

Use M3OS wiki as a curated generated medicinal-chemistry wiki. Retrieve the smallest useful context for the requesting agent. Do not browse broad inventories, dump full article bodies into the final answer, or expose raw source text.

The current wiki policy makes generated wiki articles public by default, while raw source passage access is denied. Treat the wiki as article/concept knowledge, not as a raw source archive.

## Do Not Use

- `list_articles`: too many articles; use `search_articles` with a small `limit` instead.
- `list_sources`: source inventory is not user-facing and does not answer medicinal-chemistry questions.
- `trace_lineage`: internal provenance; do not show to users or use for normal retrieval.
- `search_source_segments`, `get_source_passages`, `read_source_segment`, `list_segments`: raw source-segment tools are not useful under current source-access policy and segment data may be absent.

## Tool Roles

| Tool | Use For | Do Not Use For |
| --- | --- | --- |
| `search_articles` | Finding candidate wiki pages from a keyword, synonym, mechanism, scaffold, or property phrase. | Reading content; it returns metadata only. |
| `find_concept` | Cheaply resolving a term or alias to the canonical concept before deciding whether to fetch content. | Getting definitions or body text. |
| `get_concept` | Fetching one concept page by concept name or alias, including definition, body, aliases, and frontmatter. | Broad discovery across many possible pages. |
| `read_article` | Reading a known article by exact title/id, usually after `search_articles` or from a previous result. | Alias resolution; use `get_concept` for aliases. |
| `answer_question` | Getting a synthesized answer across the wiki when a natural-language question spans multiple pages. | Cheap lookup, exact reading, or cases where the caller can synthesize from one article. |

## Key Differences

- `search_articles` vs `get_concept`: `search_articles` returns multiple candidate pages and no body; `get_concept` returns one concept's content after concept/alias resolution.
- `find_concept` vs `get_concept`: `find_concept` is a lightweight resolver; `get_concept` pulls article content.
- `get_concept` vs `read_article`: `get_concept` accepts concept names and aliases; `read_article` expects an exact article name or id.
- `answer_question` vs primitives: `answer_question` calls the configured wiki QA model and returns a synthesized answer. Prefer primitives when one or two pages are enough and the requesting agent can reason.

## Workflow

1. Classify the request.
- If it is a broad natural-language question that likely needs several pages, use `answer_question`.
- If it is a specific medicinal-chemistry term, property, mechanism, risk, or scaffold motif, use `get_concept` first.
- If it is ambiguous, partial, or may match several pages, use `search_articles` first.
- If you already have an exact article title/id, use `read_article`.

2. Keep search narrow.
- For `search_articles`, use `limit` 3 to 5 by default. Use 10 only when the query is genuinely ambiguous.
- Do not use `list_articles` for discovery.
- Do not pass `min_status="draft"` unless the requesting agent explicitly asks for draft knowledge or published results are missing and you state that limitation.

3. Resolve and read.
- If `search_articles` returns a clear candidate and the user phrase is a concept or alias, call `get_concept` on the candidate `name`.
- If `search_articles` returns an exact article title/id and you need the raw wiki page body, call `read_article`.
- If you only need to know whether an alias maps to a concept, call `find_concept` and stop if the mapping is enough.

4. Synthesize compactly.
- Extract only the facts that answer the requesting agent's medicinal-chemistry gap.
- Cite compact source labels such as `M3OS wiki: Bioavailability` or `answer_question selected_pages: Bioavailability, Lipophilicity`.
- Do not paste full `body` or full `frontmatter` unless the user explicitly asks for raw wiki content.

5. Recover from misses.
- If `get_concept` returns empty `body`/`definition`, call `search_articles` with the same term or a more general synonym.
- If `search_articles` returns no result, try one narrower synonym and one broader parent term.
- If direct article/concept retrieval still finds nothing, try `answer_question` once with the user's original knowledge question or a focused rewritten version.
- If `answer_question` also reports no usable reference content, returns `index_found: false`, has no selected pages, or otherwise indicates the wiki has no support for the answer, state honestly that M3OS wiki did not contain relevant content.
- If wiki still does not cover the point, report the gap and fall back according to the parent medchem retrieval workflow.

## Input And Output Examples

Examples are schematic. Real outputs may include additional metadata, longer body text, or empty results.

### `search_articles`

Input:

```json
{
  "query": "oral bioavailability",
  "limit": 3
}
```

Output:

```json
[
  {
    "id": "Bioavailability",
    "name": "Bioavailability",
    "path": "wiki/Bioavailability.md",
    "related content": "...",
    "tags": ["bioavailability", "pharmacokinetics"],
    "confidence": 1.0,
    "source_count": 3,
    "single_source": false,
    "source_quality": "high",
    "status": "published",
    "kind": "concept",
    "score": 2
  }
]
```

Interpretation: candidate discovery only. Follow with `get_concept("Bioavailability")` or `read_article("Bioavailability")` if content is needed.

### `find_concept`

Input:

```json
{
  "query": "oral bioavailability"
}
```

Output:

```json
{
  "name": "Bioavailability",
  "canonical_article_id": "Bioavailability",
  "aliases": ["absorption", "oral bioavailability", "bioavailability/bioequivalence"]
}
```

Interpretation: confirms alias-to-concept mapping without pulling article body. If output is `null`, the term did not resolve to a visible concept.

### `get_concept`

Input:

```json
{
  "name": "oral bioavailability"
}
```

Output:

```json
{
  "name": "Bioavailability",
  "aliases": ["absorption", "oral bioavailability"],
  "canonical_article_id": "Bioavailability",
  "definition": "Bioavailability is the extent to which an administered drug reaches systemic circulation...",
  "body": "Bioavailability is the extent to which...\n\n## Determinants of oral bioavailability\n...",
  "frontmatter": {
    "title": "Bioavailability",
    "status": "published",
    "tags": ["bioavailability", "pharmacokinetics"]
  }
}
```

Interpretation: best default for a clear concept or synonym when you need content.

### `read_article`

Input:

```json
{
  "name_or_id": "Bioavailability"
}
```

Output:

```json
{
  "id": "Bioavailability",
  "name": "Bioavailability",
  "path": "wiki/Bioavailability.md",
  "body": "Bioavailability is the extent to which...\n\n## Determinants...",
  "frontmatter": {
    "title": "Bioavailability",
    "status": "published",
    "sources": ["raw/04-solubility-b307cbe9.md"]
  }
}
```

Interpretation: exact page read. Use after selecting a page; do not expect broad alias matching.

### `answer_question`

Input:

```json
{
  "question": "How can medicinal chemists improve oral bioavailability while preserving permeability?",
  "max_pages": 4
}
```

Output:

```json
{
  "answer": "The answer from the wiki...",
  "title": "Improving oral bioavailability while preserving permeability",
  "selected_pages": ["Bioavailability", "Lipophilicity", "H-bond donors"],
  "index_found": true
}
```

Interpretation: use for multi-page synthesis. Because it calls the configured model, prefer narrower primitive tools for simple lookup.

## Recommended Patterns

Specific term:

```text
get_concept(term)
if empty: search_articles(term, limit=5)
if still empty: answer_question(focused question, max_pages=3)
if no reference content: report that M3OS wiki did not find relevant content
```

Ambiguous phrase:

```text
search_articles(phrase, limit=5)
select candidate
get_concept(candidate.name) or read_article(candidate.name)
```

Broad mechanistic question:

```text
answer_question(question, max_pages=3-5)
if vague: search_articles(key concept, limit=3) then get_concept(...)
```

Exact title from previous tool:

```text
read_article(name_or_id)
```

Alias check only:

```text
find_concept(alias)
```

## Final Response Behavior

Return medicinal-chemistry implications to the requesting agent. Include the wiki page names used. State limitations when the article is general, the scaffold context is mismatched, the result came from draft material, or the wiki lacks a direct answer.
