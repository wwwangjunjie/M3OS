---
name: leadopt-intersection-screening
description: Screen REINVENT pools against explicit, fully specified runtime hard, ADMET, hold, and activity constraints, using direct constraint screening or complete-set intersection as appropriate.
---

# Lead-Optimization Intersection Screening

Use this skill when the task contains explicit baselines, target values,
intervals, hold tolerances, or hard constraints that cannot be represented by
the monotonic `higher`/`lower` ADMET ranker.

Do not use this skill merely because a task-contract object exists. At least
one requirement must be directly executable: an operator with its value or
interval; a hold with baseline and tolerance; or a baseline-relative objective
with an exact source/endpoint, baseline, direction, and numeric threshold. A
property with only an `increase`/`decrease` direction is a monotonic ranking
preference, not a constraint. When no executable constraint exists, return to
the ordinary monotonic ADMET or activity-only filtering route; do not invent
missing contract fields.

## Workflow

1. For a target-activity task, call
   `generate_similarity_constrained_mol2mol` once to request up to 10000 unique
   qualifying candidates for the current round. Use the non-empty real pool if
   the sampling limit is reached before all 10000 candidates are collected.
2. Use an attached structured task contract unchanged. If none is attached,
   construct `task_contract_json` from the current task using the runtime
   contract template in the Creative system prompt. Include only stated
   properties, operators, baselines, tolerances, and thresholds.
3. If there is no target-activity requirement, call
   `screen_reinvent_candidates_by_constraints` once. Treat its passing top 5
   as the complete Creative output for the round.
4. If activity is combined with hard, ADMET, baseline-relative, interval, or
   hold requirements, call `screen_reinvent_candidates_by_intersection` once
   for the current generated pool.
   The backend evaluates the complete hard/ADMET-qualified set with Nesso
   probability and affinity gates, without pre-truncation, then ranks and
   retains at most 10 molecules by Nesso affinity, using binding probability as
   a required floor and tie-breaker. A binding-affinity improvement objective
   is evaluated against the initial molecule using the Nesso affinity values;
   the improvement must meet or exceed the task threshold (for example,
   greater than or equal to 0.3 when the task threshold is 0.3).
5. When the intersection is non-empty, treat the returned shortlist as the
   complete screened Creative result for the current round. When it is empty
   but ADMET/hard-qualified molecules were evaluated by Nesso, use their full
   metrics, target gaps, and failure reasons to propose a compact unverified
   hypothesis batch for Critic evaluation. If no molecule passed the
   ADMET/hard stage, return no Creative candidate. Do not generate a second
   pool, retry the screen, or add a relaxed activity fallback. An empty intersection
   remains an empty screened result even when it informs new hypotheses.
6. Reuse the accumulated screening-context history from every completed prior
   round. Creative sees the current tool result immediately; both Creative and
   Rational receive completed-round screening evidence in later rounds.

Do not invent task thresholds. Do not truncate either source set before an
intersection. Do not manually merge CSV rows, relax a task threshold, reorder
the returned intersection, or add hand-authored molecules.
