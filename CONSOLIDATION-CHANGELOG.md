# ml-system — Consolidation Changelog

**Date:** 2026-06-30
**Phase:** Phase 2 Part 7 meta-repo (Model A — subtree fold)
**Tracking:** `governance/portfolio-ops/PHASE2-PART7-PILLAR-FOLD-PLAN.md`

---

## Summary

| Metric | Before | After |
|--------|-------:|------:|
| GitHub repos (this pillar) | 3 | **1** |

## Sub-project mapping

| Archived GitHub repo | Branch | Subfolder | Import commit |
|----------------------|--------|-----------|---------------|
| `vgandhi1/ecommerce-demand-forecast` | `main` | `ecommerce-demand-forecast/` | `85f1d78` |
| `vgandhi1/Sentinel-Stream` | `main` | `sentinel-fraud/` | `f518c8c` |
| `vgandhi1/Prism-Federated` | `main` | `Prism-Federated/` | `90b86b7` |

**Method:** `git subtree add` per sub (history preserved). Nested `.git` dirs removed; single `origin`.
**Backup:** `/tmp/mlsystem-gitdirs-20260630.tgz`

**Name mapping:** remote `Sentinel-Stream` folded into subfolder `sentinel-fraud/` (remote name unchanged, archived).
`sentinel-fraud` (FinTech streaming ML) is **not** the archived `SentinelFlow` (IIoT) — different project.

## Siblings archived

All three standalone repos archived (2026-06-30) with README redirect banners → this meta's subfolder.

## Pages (DONE 2026-06-30)

Combined `_site/<sub>/` published via `peaceiris/actions-gh-pages` → `gh-pages`; Pages source = gh-pages branch. All decks return 200:
  - https://vgandhi1.github.io/ml-system/ecommerce-demand-forecast/
  - https://vgandhi1.github.io/ml-system/sentinel-fraud/
  - https://vgandhi1.github.io/ml-system/Prism-Federated/
  - landing: https://vgandhi1.github.io/ml-system/
