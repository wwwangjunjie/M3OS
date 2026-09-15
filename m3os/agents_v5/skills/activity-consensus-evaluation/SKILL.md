---
name: activity-consensus-evaluation
description: Evaluate a compact candidate set with Nesso binding probability and affinity value, using deterministic task gates and ranking.
---

# Nesso Activity Evaluation

Use this skill for Critic evaluation when target activity is part of the task.

## Workflow

1. Submit the complete current-round candidate list to
   `evaluate_nesso_activity`. Include both Creative and Rational candidates;
   the backend adds the initial and current molecules and reuses cached results.
2. Use `Nesso_activity_pass` as the primary activity tier. It requires Nesso
   binding probability to satisfy the task's proxy threshold. Probability is a
   floor, not an initial-relative improvement criterion. The lower-is-better
   Nesso affinity value must satisfy the task's initial-relative improvement or
   hold rule. For an improvement objective, compute the change from the initial
   molecule using Nesso affinity values and require it to meet or exceed the
   task threshold (for example, greater than or equal to 0.3 for a 0.3 threshold).
3. Within a tier, use `Nesso_activity_rank` and `Nesso_activity_score`. The
   score and primary ordering follow the Nesso affinity value; probability is
   used only as a pass/fail floor and as a deterministic tie-breaker.
4. Report both raw Nesso outputs for each candidate. Preserve the distinction
   between `Nesso_probability_task_pass` and `Nesso_affinity_task_pass`.

Do not recompute the deterministic score manually, omit Rational candidates, or request a
second prediction for values already returned by this tool.
