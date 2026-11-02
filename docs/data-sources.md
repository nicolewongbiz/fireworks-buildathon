# Data sources for molecule-image -> SMILES (OCSR) fine-tuning

Compiled 2026-11-01. All downloads live under `data/raw/<source>/` with a `SOURCE.md` (URL, license, sha256, download date) in each folder. `data/raw` is ~704 MB total; do **not** commit it.

## 1. SMILES source for synthetic training renders: ZINC250k

| | |
|---|---|
| File | `data/raw/zinc250k/250k_rndm_zinc_drugs_clean_3.csv` (22.6 MB, 249,455 rows, columns `smiles,logP,qed,SAS`) |
| URL | https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv |
| License | Hosting repo (aspuru-guzik-group/chemical_vae) is Apache-2.0; molecules are a random 250k subset of ZINC12 drug-like/clean, and ZINC is free for everyone to use and redistribute. Cite ZINC (Sterling & Irwin 2015) + Gomez-Bombarelli et al. 2018. |
| Download time | 1.6 s (GitHub raw, no LFS) |
| sha256 | `35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae` |
| Gotcha | SMILES cells are quoted and contain a trailing `\n` inside the quotes -- `.str.strip()` them. |

**RDKit stats (full file, 249,455 SMILES):**
- Parse rate: 249,455 / 249,455 (100%), all unique
- Heavy atoms: p10 = 17, p50 = 23, p90 = 29 (min 6, max 38, mean 23.2); 99.96% fall in the 10-40 range
- Stereo marks (`@`, `/`, `\`): 63.1% of molecules
- Formally charged atoms (`[NH+]`, `[O-]`, ... -- ZINC pH-7 protonation states): 32.6%

Sample SMILES:
```
N#Cc1nccn1Cc1ccncc1
c1cc(C[NH2+]CC[NH+]2CCCCCC2)c2c(c1)OCCCO2
COc1ccccc1-n1cnnc1SCC(=O)c1ccc(Cl)c(Cl)c1
Cn1ccnc1[C@@H](NC(=O)Cc1csc(-c2ccc(Cl)cc2)n1)c1ccccc1
COc1cc(/C=C2\SC(=S)N(c3ccccc3C)C2=O)ccc1OC(=O)c1cccs1
```

Why not the others: MOSES (MIT, ZINC-derived) is a Git-LFS file and offers nothing over ZINC250k for our size; ChEMBL is CC BY-SA 3.0 (share-alike) and a multi-GB download; PubChem is public domain but huge and needs filtering; GuacaMol is ChEMBL-derived (same license); QM9 molecules are too small (<= 9 heavy atoms).

Training note: ~1/3 of ZINC SMILES carry explicit charges (protonated amines, carboxylates). Decide early whether to neutralise (e.g. `rdMolStandardize.Uncharger`) before rendering, since patent drawings almost always show neutral forms; otherwise the model learns to emit `[NH+]`.

## 2. Real-world labeled OCSR benchmarks found

| Dataset | URL | License | Images | Labels | Style | Download size |
|---|---|---|---|---|---|---|
| **USPTO (OSRA validation set)** | https://osra.sourceforge.net/uspto-validation-updated.zip | US patent drawings (public domain); MOL ground truth distributed with OSRA, no explicit data license -- widely reused as a research benchmark (Rajan 2020, Img2Mol, MolScribe, DECIMER) | 5,719 TIF | MOL files (1 per image) | Patent scans, bilevel, clean single structures | 14 MB zip (36 MB extracted) |
| **MolScribe `real.zip` bundle** | https://huggingface.co/yujieq/MolScribe/resolve/main/real.zip | MIT (MolScribe); underlying images keep their own provenance (see below) | 62,782 PNG: USPTO 5,719, UOB 5,740, CLEF 992, ACS 331, Staker 50,000 | CSV `image_id,file_path,SMILES` per subset | USPTO/CLEF/Staker = patent scans; UOB = clean renders; ACS = RGB journal figures | 184 MB zip (270 MB extracted) |
| UOB / Maybridge UoB (original) | http://www.cs.bham.ac.uk/research/groupings/reasoning/sdag/chemical.php | Academic benchmark (Sadawi et al.); no explicit license | 5,740 | MOL | Clean renders of Maybridge catalogue compounds | n/a -- original page now redirects to the Birmingham CS homepage (dead). Use the copy inside MolScribe `real.zip`. |
| CLEF-IP 2012 structure recognition | http://www.ifs.tuwien.ac.at/~clef-ip/download/2012/qrels/clef-ip-2012-chem-recognition-qrels.tgz | CLEF-IP research-use terms | 865-992 | MOL | Patent scans, many Markush/R-groups | n/a -- URL returns 404 (dead). Use MolScribe copy (992 images). |
| JPO (Chem-Infty organic subset) | https://osra.sourceforge.net/images-organic.zip + https://osra.sourceforge.net/molfiles-organic.zip | **CC BY-NC-ND 2.1 JP** (non-commercial, no derivatives) | 450 | MOL | Japanese patent scans, lower quality, Japanese text | 2.6 MB + 0.35 MB (available; not downloaded because of ND clause) |
| Staker et al. 2019 | Google Drive folder linked from MolScribe README (https://drive.google.com/drive/folders/16OjPwQ7bQ486VhdX4DWpfYzRsTGgJkSu); 50k subset inside `real.zip` | Paper-derived, patent images; no explicit license | 50,000 (subset) | SMILES CSV | 256x256 grayscale patent crops | in `real.zip` |
| ACS (MolScribe) | inside `real.zip` | Journal figures, publisher copyright -- internal evaluation only | 331 | SMILES CSV (84% RDKit-parsable; many R-groups) | RGB figures from ACS papers | in `real.zip` |
| **DECIMER Hand-drawn (HDM) v1.2** | https://zenodo.org/records/7617107 | **CC BY 4.0** | 5,088 PNG | SMILES TSV (+ SD files) | Hand-drawn structures, photographed/scanned | 79 MB images + 0.23 MB TSV (+2.3 MB SDF) |
| ChemPix (Weir et al. 2021) | https://github.com/mtzgroup/ChemPixCH | Paper CC BY-NC 3.0; data license unclear | ~600 crowd-sourced | SMILES | Hand-drawn hydrocarbons only | small; not downloaded (hydrocarbon-only, less useful) |
| Img2Mol perturbed sets | https://github.com/bayer-science-for-a-better-life/Img2Mol (`benchmark_data/`); MolScribe `perturb.zip` (396 MB) | Img2Mol code Apache-2.0, weights CC BY-NC 4.0 | USPTO/UOB/CLEF/Staker perturbed | SMILES CSV | Degraded versions of the above | 396 MB; not downloaded |
| MolScribe `synthetic.zip` | https://huggingface.co/yujieq/MolScribe/resolve/main/synthetic.zip | MIT | Indigo + ChemDraw renders | SMILES CSV | Clean synthetic renders (not real-world) | 232 MB; not downloaded |
| MolScribe `uspto_mol.zip` (training) | https://huggingface.co/yujieq/MolScribe/resolve/main/uspto_mol.zip | MIT / US patents | ~680k patent images | MOL/SMILES | Patent scans -- real-world *training* data if we want it | 2.18 GB; not downloaded |

**Recommendation**
- **Primary (patent-style, clean labels, small): USPTO OSRA validation set** (`data/raw/uspto_osra/`, 14 MB, 5,719 images with MOL files). It is the canonical benchmark used in every OCSR paper since 2012, so our numbers will be directly comparable to OSRA/MolVec/Imago/Img2Mol/MolScribe/DECIMER. Use the MolScribe `real/USPTO.csv` SMILES as a cross-check (98.5% canonical agreement; the rest are mainly MOL files RDKit cannot sanitize, salts and R-group tokens).
- **Secondary / bonus:** the MolScribe `real.zip` bundle (`data/raw/molscribe_real/`) gives UOB (clean renders, 100% parsable) and CLEF for free in the same format, plus 50k Staker crops that could even be used as *real-world fine-tuning data* if the synthetic-only model underperforms.
- **Stretch (hand-drawn): DECIMER HDM** (`data/raw/decimer_handdrawn/`, CC BY 4.0, 5,088 images, 100% parsable SMILES). Cleanly licensed, well-known, and small.

## 3. What was downloaded where

| Path | Contents | Size on disk | Verified |
|---|---|---|---|
| `data/raw/zinc250k/250k_rndm_zinc_drugs_clean_3.csv` | 249,455 SMILES + logP/QED/SAS | 22 MB | 100% RDKit parse; stats above |
| `data/raw/uspto_osra/uspto-validation-updated.zip` + extracted `uspto-validation-updated/{images,molfiles}` | 5,719 TIF + 5,719 MOL | 66 MB | 5 TIFs opened with PIL (mode "1", e.g. 826x594); 5,704/5,719 MOL files parse with RDKit (15 fail); 5 MOL -> canonical SMILES all match MolScribe CSV |
| `data/raw/molscribe_real/real.zip` + extracted `real/` | USPTO, UOB, CLEF, acs, staker PNGs + CSVs | 449 MB | 5 PNGs per subset opened; 5 SMILES per subset canonicalised; parse rates USPTO 98.5%, UOB 100%, CLEF 91.1%, ACS 83.7%, Staker 99.1% |
| `data/raw/decimer_handdrawn/DECIMER_HDM_Dataset_Images.zip`, `DECIMER_HDM_Dataset_SMILES.tsv` + extracted PNGs | 5,088 hand-drawn PNGs + TSV labels | 167 MB | 5 PNGs opened (mode "P", e.g. 687x429); 5,086/5,088 SMILES parse; all 5,088 images present |

Each folder has a `SOURCE.md` with URL, license, date and sha256. Helper scripts used for verification live in the session scratchpad, not in the repo.

## 4. Licensing caveats

1. **ZINC250k**: Apache-2.0 repo + ZINC "free for everyone". Fine for a hackathon demo and for releasing rendered images/weights. Add a citation.
2. **USPTO OSRA set**: images are US government patent publications (public domain). The MOL ground truth has no explicit license text; it has been redistributed freely for 13+ years in research. Safe for benchmarking/demo; cite OSRA (Filippov & Nicklaus) and Rajan et al. 2020.
3. **MolScribe real.zip**: the *bundle* is MIT, but MIT only covers MolScribe's packaging. UOB and CLEF originals carry academic-use terms and their original hosts are now dead; **ACS images are copyrighted journal figures -- use for internal evaluation only, do not redistribute or show in slides**. Staker images are patent-derived (public domain source).
4. **DECIMER HDM**: CC BY 4.0 -- attribution required (Brinkhaus et al., J. Cheminform. 2022, 14:36). Authors ask that it be used for benchmarking only.
5. **JPO / Chem-Infty**: CC BY-NC-ND 2.1 JP -- non-commercial and *no derivatives*, so do not augment/perturb/redistribute it; skipped.
6. **Img2Mol weights** are CC BY-NC 4.0 (non-commercial) -- relevant only if the team wants to compare against Img2Mol predictions, not for data.
7. **Label hygiene**: ~1.5% of USPTO, ~9% of CLEF and ~16% of ACS labels in MolScribe's CSVs are Markush/abbreviation tokens (`[R']`, `[1*]`, `[(].[)]`) that RDKit cannot parse. Filter with `Chem.MolFromSmiles(...) is not None` and score on canonical SMILES (or InChI) exact match, as the field does.
