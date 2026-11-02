# Paired comparisons

Generated 2026-11-02 07:48 UTC by `src/evaluate.py`. Systems scored on the same test set saw the same images (matched on `image_path`), so each pair is compared item by item. Difference is A − B in percentage points. The 95% interval is a paired bootstrap of the difference (1000 resamples over the shared images, seed 0). McNemar p is the exact two-sided binomial test of only-A vs only-B among the discordant pairs. Headline metric is stereo-aware exact match (`match.exact_stereo`, falling back to `match.exact`); the stereo-agnostic row (`match.exact`) follows it.

Rule of thumb for a single system at n=500: the binomial 95% half-width is about ±2.2 points at 7%, ±3.1 at 15% and ±4.4 at 50%. The paired test below is sharper than comparing two such intervals because the systems share images, so quote the paired interval and McNemar p-value when ranking systems.

Restricted to `--pairs n500_r384_lang_e3:untuned,molscribe:n500_r384_lang_e3`.

## test_synthetic_384

| A | B | Metric | n shared | Both right | Only A | Only B | Both wrong | A | B | A − B [paired 95%] | McNemar p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| n500_r384_lang_e3 | untuned | stereo-aware | 500 | 0 | 34 | 0 | 466 | 6.8% | 0.0% | +6.8 pts [+4.6, +9.0] | < 0.001 |
| n500_r384_lang_e3 | untuned | stereo-agnostic | 500 | 0 | 57 | 0 | 443 | 11.4% | 0.0% | +11.4 pts [+8.8, +14.4] | < 0.001 |
| molscribe | n500_r384_lang_e3 | stereo-aware | 500 | 34 | 433 | 0 | 33 | 93.4% | 6.8% | +86.6 pts [+83.8, +89.4] | < 0.001 |
| molscribe | n500_r384_lang_e3 | stereo-agnostic | 500 | 57 | 420 | 0 | 23 | 95.4% | 11.4% | +84.0 pts [+81.0, +87.0] | < 0.001 |
