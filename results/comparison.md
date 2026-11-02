# System comparison

Generated 2026-11-02 07:34 UTC by `src/evaluate.py` from 11 results file(s). All metrics recomputed from per-item records; brackets are bootstrap 95% CIs (1000 resamples, seed 0). Headline exact match is stereo-aware; cost per correct uses it.

| System | Test set | n | Exact (stereo) | Exact (no stereo) | Valid | Mean Tanimoto | Loop rate | Latency p50 / p95 | USD | Cost / correct | Cost / image |
|---|---|---:|---|---|---|---:|---:|---|---:|---:|---:|
| molscribe | test_handdrawn_384 | 200 | 11.5% [7.5, 16.0] | 14.5% [9.5, 19.0] | 76.0% [70.0, 82.0] | 0.329 | 0.0% | 0.77s / 1.25s | $0.0000 | $0.0000 | $0.0000 |
| molscribe | test_real_uspto_384 | 200 | 89.0% [84.5, 93.0] | 92.0% [88.0, 95.5] | 96.5% [94.0, 98.5] | 0.953 | 0.0% | 0.99s / 1.32s | $0.0000 | $0.0000 | $0.0000 |
| molscribe | test_synthetic_384 | 500 | 93.4% [91.2, 95.4] | 95.4% [93.6, 97.2] | 99.2% [98.4, 99.8] | 0.981 | 0.0% | 0.89s / 1.09s | $0.0000 | $0.0000 | $0.0000 |
| n500_r384_lang_e3 | test_synthetic_384 | 500 | 6.8% [4.6, 9.0] | 11.4% [8.8, 14.4] | 72.0% [68.0, 75.8] | 0.416 | 0.0% | 3.37s / 4.93s | $0.26 | $0.0076 | $0.0005 |
| smoke_large_glm * | smoke | 10 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.000 | 100.0% | 1.01s / 1.37s | $0.0009 | inf (0 correct) | $0.0001 |
| smoke_large_kimi * | smoke | 10 | 80.0% [60.0, 100.0] | 80.0% [60.0, 100.0] | 100.0% [100.0, 100.0] | 0.968 | 0.0% | 1.19s / 3.46s | $0.0128 | $0.0016 | $0.0013 |
| smoke_large_qwen * | smoke | 10 | 20.0% [0.0, 50.0] | 20.0% [0.0, 50.0] | 80.0% [50.0, 100.0] | 0.388 | 80.0% | 1.31s / 1.69s | $0.0091 | $0.0046 | $0.0009 |
| smoke_molscribe * | smoke | 10 | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 1.000 | 0.0% | 0.94s / 1.06s | $0.0000 | $0.0000 | $0.0000 |
| smoke_tuned * | smoke | 10 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 90.0% [70.0, 100.0] | 0.174 | n/a | 2.05s / 8.07s | $0.0190 | inf (0 correct) | $0.0019 |
| smoke_untuned * | smoke | 10 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 70.0% [40.0, 90.0] | 0.122 | n/a | 2.87s / 7.55s | $0.15 | inf (0 correct) | $0.0151 |
| untuned | test_synthetic_384 | 500 | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 67.0% [63.0, 71.4] | 0.109 | 19.2% | 2.20s / 4.00s | $0.19 | inf (0 correct) | $0.0004 |

Cost source per system: molscribe · test_handdrawn_384: summary.usd; molscribe · test_real_uspto_384: summary.usd; molscribe · test_synthetic_384: summary.usd; n500_r384_lang_e3 · test_synthetic_384: summary.usd; smoke_large_glm · smoke: summary.usd; smoke_large_kimi · smoke: summary.usd; smoke_large_qwen · smoke: summary.usd; smoke_molscribe · smoke: summary.usd; smoke_tuned · smoke: summary.usd; smoke_untuned · smoke: summary.usd; untuned · test_synthetic_384: summary.usd.

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

| Tag | molscribe · test_handdrawn_384 | molscribe · test_real_uspto_384 | molscribe · test_synthetic_384 | n500_r384_lang_e3 · test_synthetic_384 | untuned · test_synthetic_384 |
|---|---|---|---|---|---|
| heavy_atoms<=20 | 11.5% (21/183) | 90.3% (28/31) | 94.5% (104/110) | 11.8% (13/110) | 0.0% (0/110) |
| heavy_atoms 21-30 | 11.8% (2/17) | 89.6% (60/67) | 93.4% (342/366) | 5.7% (21/366) | 0.0% (0/366) |
| heavy_atoms 31-40 | – | 92.5% (74/80) | 87.5% (21/24) | 0.0% (0/24) | 0.0% (0/24) |
| heavy_atoms>40 | – | 72.7% (16/22) | – | – | – |
| rings=0 | 6.8% (6/88) | 80.0% (4/5) | – | – | – |
| rings=1 | 16.4% (10/61) | 90.9% (20/22) | 100.0% (6/6) | 0.0% (0/6) | 0.0% (0/6) |
| rings=2 | 12.1% (4/33) | 87.0% (20/23) | 93.4% (142/152) | 11.2% (17/152) | 0.0% (0/152) |
| rings=3 | 20.0% (2/10) | 75.9% (41/54) | 94.4% (204/216) | 5.6% (12/216) | 0.0% (0/216) |
| rings>=4 | 12.5% (1/8) | 96.9% (93/96) | 91.3% (115/126) | 4.0% (5/126) | 0.0% (0/126) |
| stereocenters=0 | 10.8% (17/157) | 94.2% (147/156) | 97.5% (197/202) | 15.3% (31/202) | 0.0% (0/202) |
| stereocenters 1-2 | 16.1% (5/31) | 81.0% (17/21) | 93.1% (244/262) | 1.1% (3/262) | 0.0% (0/262) |
| stereocenters>=3 | 8.3% (1/12) | 60.9% (14/23) | 72.2% (26/36) | 0.0% (0/36) | 0.0% (0/36) |
| no E/Z bonds | 12.0% (22/183) | 90.4% (161/178) | 93.5% (447/478) | 6.9% (33/478) | 0.0% (0/478) |
| E/Z bonds | 5.9% (1/17) | 77.3% (17/22) | 90.9% (20/22) | 4.5% (1/22) | 0.0% (0/22) |
| charged | 15.4% (4/26) | 100.0% (6/6) | 82.0% (50/61) | 1.6% (1/61) | 0.0% (0/61) |
| neutral | 10.9% (19/174) | 88.7% (172/194) | 95.0% (417/439) | 7.5% (33/439) | 0.0% (0/439) |
| single fragment | 11.5% (23/200) | 89.8% (177/197) | 94.5% (447/473) | 7.2% (34/473) | 0.0% (0/473) |
| multi-fragment | – | 33.3% (1/3) | 74.1% (20/27) | 0.0% (0/27) | 0.0% (0/27) |
| elements: CNO only | 12.8% (6/47) | 87.6% (78/89) | 93.0% (186/200) | 6.0% (12/200) | 0.0% (0/200) |
| elements: beyond CNO | 11.1% (17/153) | 90.1% (100/111) | 93.7% (281/300) | 7.3% (22/300) | 0.0% (0/300) |
| not bridged | 11.5% (22/191) | 88.8% (175/197) | 95.0% (457/481) | 7.1% (34/481) | 0.0% (0/481) |
| bridged | 11.1% (1/9) | 100.0% (3/3) | 52.6% (10/19) | 0.0% (0/19) | 0.0% (0/19) |
| not fused | 11.3% (20/177) | 85.0% (102/120) | 95.6% (285/298) | 9.1% (27/298) | 0.0% (0/298) |
| fused | 13.0% (3/23) | 95.0% (76/80) | 90.1% (182/202) | 3.5% (7/202) | 0.0% (0/202) |
