# Paired comparisons

Generated 2026-11-02 08:12 UTC by `src/evaluate.py`. Systems scored on the same test set saw the same images (matched on `image_path`), so each pair is compared item by item. Difference is A − B in percentage points. The 95% interval is a paired bootstrap of the difference (1000 resamples over the shared images, seed 0). McNemar p is the exact two-sided binomial test of only-A vs only-B among the discordant pairs. Headline metric is stereo-aware exact match (`match.exact_stereo`, falling back to `match.exact`); the stereo-agnostic row (`match.exact`) follows it.

Rule of thumb for a single system at n=500: the binomial 95% half-width is about ±2.2 points at 7%, ±3.1 at 15% and ±4.4 at 50%. The paired test below is sharper than comparing two such intervals because the systems share images, so quote the paired interval and McNemar p-value when ranking systems.

Restricted to `--pairs n500_r384_lang_e3:untuned,molscribe:n500_r384_lang_e3,large_kimi:n500_r384_lang_e3,molscribe:large_kimi`.

## test_real_uspto_384

| A | B | Metric | n shared | Both right | Only A | Only B | Both wrong | A | B | A − B [paired 95%] | McNemar p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| molscribe | large_kimi | stereo-aware | 200 | 104 | 74 | 8 | 14 | 89.0% | 56.0% | +33.0 pts [+25.5, +41.0] | < 0.001 |
| molscribe | large_kimi | stereo-agnostic | 200 | 129 | 55 | 9 | 7 | 92.0% | 69.0% | +23.0 pts [+16.5, +30.0] | < 0.001 |

## test_synthetic_384

| A | B | Metric | n shared | Both right | Only A | Only B | Both wrong | A | B | A − B [paired 95%] | McNemar p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| n500_r384_lang_e3 | untuned | stereo-aware | 500 | 0 | 36 | 0 | 464 | 7.2% | 0.0% | +7.2 pts [+5.0, +9.4] | < 0.001 |
| n500_r384_lang_e3 | untuned | stereo-agnostic | 500 | 0 | 60 | 0 | 440 | 12.0% | 0.0% | +12.0 pts [+9.2, +15.0] | < 0.001 |
| molscribe | n500_r384_lang_e3 | stereo-aware | 500 | 36 | 431 | 0 | 33 | 93.4% | 7.2% | +86.2 pts [+83.4, +89.2] | < 0.001 |
| molscribe | n500_r384_lang_e3 | stereo-agnostic | 500 | 60 | 417 | 0 | 23 | 95.4% | 12.0% | +83.4 pts [+80.0, +86.6] | < 0.001 |
| large_kimi | n500_r384_lang_e3 | stereo-aware | 500 | 30 | 183 | 6 | 281 | 42.6% | 7.2% | +35.4 pts [+30.8, +40.0] | < 0.001 |
| large_kimi | n500_r384_lang_e3 | stereo-agnostic | 500 | 55 | 329 | 5 | 111 | 76.8% | 12.0% | +64.8 pts [+60.6, +69.2] | < 0.001 |
| molscribe | large_kimi | stereo-aware | 500 | 207 | 260 | 6 | 27 | 93.4% | 42.6% | +50.8 pts [+46.0, +55.4] | < 0.001 |
| molscribe | large_kimi | stereo-agnostic | 500 | 371 | 106 | 13 | 10 | 95.4% | 76.8% | +18.6 pts [+14.6, +22.4] | < 0.001 |
