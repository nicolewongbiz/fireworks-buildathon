# System comparison

Generated 2026-11-02 07:21 UTC by `src/evaluate.py` from 7 results file(s). All metrics recomputed from per-item records; brackets are bootstrap 95% CIs (1000 resamples, seed 0). Headline exact match is stereo-aware; cost per correct uses it.

| System | Test set | n | Exact (stereo) | Exact (no stereo) | Valid | Mean Tanimoto | Loop rate | Latency p50 / p95 | USD | Cost / correct | Cost / image |
|---|---|---:|---|---|---|---:|---:|---|---:|---:|---:|
| smoke_large_glm * | smoke | 10 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.000 | 100.0% | 1.01s / 1.37s | $0.0009 | inf (0 correct) | $0.0001 |
| smoke_large_kimi * | smoke | 10 | 80.0% [60.0, 100.0] | 80.0% [60.0, 100.0] | 100.0% [100.0, 100.0] | 0.968 | 0.0% | 1.19s / 3.46s | $0.0128 | $0.0016 | $0.0013 |
| smoke_large_qwen * | smoke | 10 | 20.0% [0.0, 50.0] | 20.0% [0.0, 50.0] | 80.0% [50.0, 100.0] | 0.388 | 80.0% | 1.31s / 1.69s | $0.0091 | $0.0046 | $0.0009 |
| smoke_molscribe * | smoke | 10 | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 1.000 | 0.0% | 0.94s / 1.06s | $0.0000 | $0.0000 | $0.0000 |
| smoke_tuned * | smoke | 10 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 90.0% [70.0, 100.0] | 0.174 | n/a | 2.05s / 8.07s | $0.0190 | inf (0 correct) | $0.0019 |
| smoke_untuned * | smoke | 10 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 70.0% [40.0, 90.0] | 0.122 | n/a | 2.87s / 7.55s | $0.15 | inf (0 correct) | $0.0151 |
| untuned | test_synthetic_384 | 500 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 67.0% [63.0, 71.4] | 0.109 | 19.2% | 2.20s / 4.00s | $0.19 | inf (0 correct) | $0.0004 |

Cost source per system: smoke_large_glm · smoke: summary.usd; smoke_large_kimi · smoke: summary.usd; smoke_large_qwen · smoke: summary.usd; smoke_molscribe · smoke: summary.usd; smoke_tuned · smoke: summary.usd; smoke_untuned · smoke: summary.usd; untuned · test_synthetic_384: summary.usd.

## Notes (* in the table)

- `smoke_large_glm` (smoke): items carry no tags
- `smoke_large_kimi` (smoke): items carry no tags
- `smoke_large_qwen` (smoke): items carry no tags
- `smoke_molscribe` (smoke): items carry no tags
- `smoke_tuned` (smoke): match.exact_stereo missing; derived with RDKit from pred_smiles vs truth
- `smoke_tuned` (smoke): match.hit_cap and summary.max_tokens missing; loop rate is n/a
- `smoke_tuned` (smoke): items carry no tags
- `smoke_tuned` (smoke): pre-57f43b5 summary: exact_match there is stereo-agnostic
- `smoke_untuned` (smoke): match.exact_stereo missing; derived with RDKit from pred_smiles vs truth
- `smoke_untuned` (smoke): match.hit_cap and summary.max_tokens missing; loop rate is n/a
- `smoke_untuned` (smoke): items carry no tags
- `smoke_untuned` (smoke): pre-57f43b5 summary: exact_match there is stereo-agnostic

## Charts

![chart_exact_vs_cost](chart_exact_vs_cost.png)
![chart_exact_by_testset](chart_exact_by_testset.png)

Dark-surface variants: chart_exact_vs_cost_dark.png, chart_exact_by_testset_dark.png

## Exact match (stereo-aware) by tag

Tags from the test manifest (`make_dataset.py` difficulty attributes are bucketed; list tags used as is). Cell = exact % (hits/n).

| Tag | untuned · test_synthetic_384 |
|---|---|
| heavy_atoms<=20 | 0.0% (0/110) |
| heavy_atoms 21-30 | 0.0% (0/366) |
| heavy_atoms 31-40 | 0.0% (0/24) |
| rings=1 | 0.0% (0/6) |
| rings=2 | 0.0% (0/152) |
| rings=3 | 0.0% (0/216) |
| rings>=4 | 0.0% (0/126) |
| stereocenters=0 | 0.0% (0/202) |
| stereocenters 1-2 | 0.0% (0/262) |
| stereocenters>=3 | 0.0% (0/36) |
| no E/Z bonds | 0.0% (0/478) |
| E/Z bonds | 0.0% (0/22) |
| charged | 0.0% (0/61) |
| neutral | 0.0% (0/439) |
| single fragment | 0.0% (0/473) |
| multi-fragment | 0.0% (0/27) |
| elements: CNO only | 0.0% (0/200) |
| elements: beyond CNO | 0.0% (0/300) |
| not bridged | 0.0% (0/481) |
| bridged | 0.0% (0/19) |
| not fused | 0.0% (0/298) |
| fused | 0.0% (0/202) |
