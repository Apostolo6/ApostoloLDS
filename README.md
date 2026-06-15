# HMLS Pipeline for *Pinus pinaster*

Open-source five-phase handheld mobile LiDAR (HMLS) processing pipeline for tree detection, stem reconstruction and volume estimation in Mediterranean *Pinus pinaster* plantations. Calibrated and validated against a destructively-measured reference of 45 trees in the Alge area of Lousã, central Portugal.

This repository accompanies:

> Apóstolo, P.; Brandão, I.; Fidalgo, B.; Salas, R.; Mendes, M. *An open-source handheld mobile LiDAR pipeline for tree detection, stem reconstruction, and volume estimation in Mediterranean Pinus pinaster plantations, calibrated and validated against destructive reference*. **2026**. \[DOI / link to be added once accepted\].

---

## Contents

- [Headline results](#headline-results)
- [Pipeline overview](#pipeline-overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Repository structure](#repository-structure)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Data](#data)
- [How to cite](#how-to-cite)
- [Licence](#licence)
- [Acknowledgements](#acknowledgements)

---

## Headline results

On 45 destructively-measured *Pinus pinaster* trees across four plantation plots in Lousã, central Portugal (Alge1, Alge4, Alge5, Alge6; with a fifth plot Alge2 excluded as documented sensitivity analysis), the pipeline produces:

| Variable | MAPE | MedAPE | Bias | n |
| --- | --- | --- | --- | --- |
| Stem volume | 9.28 % | 7.70 % | +2.27 % | 45 |
| Total height | 4.04 % | 3.27 % | +0.13 % | 45 |
| DBH | 3.77 % | 2.46 % | +0.13 % | 45 |

Calibration constants collapse to near unity (0.9659 for volume, 1.0549 for DBH, 1.0270 for height), evidence that the upstream geometric pipeline is approximately unbiased.

Head-to-head against the FJD Trion Model and a manual CloudCompare workflow on the matched 40-tree sub-sample, the pipeline **outperforms both baselines on the DBH median absolute percentage error** (2.74 % against 3.59 % and 4.92 %).

See the paper for the full per-plot decomposition, the head-to-head comparison and the documented failure modes.

---

## Pipeline overview

The pipeline runs in five sequential phases. Phase 2 uses CUDA opportunistically when a GPU is available and falls back to CPU otherwise; all other phases are CPU-only.

```
 Raw SLAM cloud (.las)
        |
        v
 [Phase 1] PDAL
   · Voxel deduplication (5 mm)
   · Statistical outlier removal
   · Cloth Simulation Filter (CSF) ground classification
   · Height-Above-Ground (HAG) computation
   · Geometric debris marking
        |
        v
 [Phase 2] TLS2trees instance segmentation
   · Skeleton-following semantic pre-labelling
   · TLS2trees graph-based individual tree segmentation
   · Per-tree leaf-off and leaf-on PLY export
        |
        v
 [Phase 2.5] Per-tree setup
   · CSF-anchored base height z_base (KD-tree median on Phase-1 ground)
   · DBH prior radius (RANSAC-ready)
   · Leaf-on height with apex correction
        |
        v
 [Phase 3] Stem reconstruction
   · Per-slice RANSAC + sequential prior + Gauss-Newton geometric refit
   · Continuity filter (rolling-median skeleton + drift bound)
   · Species-shape taper (4th-degree polynomial, R²=0.97 on destructive reference)
       OR Kozak 1988 / power-law alternatives
   · Three-estimator DBH ensemble (median of: robust local linear, near-DBH median, 3-D cone fit)
   · Tube filter + taper-driven analytical synthetic-fill
        |
        v
 [Phase 4] Volume integration
   · Analytical trapezoidal integration of fitted taper
   · Form-factor sanity clamp (f ∈ [0.34, 0.62])
   · Per-tree quality flag and uncertainty estimate
   · Stand-level cartographic output
```

---

## Installation

The pipeline requires Python 3.8–3.11 (Open3D does not support Python 3.12+).

```bash
# Conda environment
conda create -n lidar python=3.11
conda activate lidar

# Install Python dependencies
pip install -r requirements.txt

# PDAL is required for Phase 1 (install via conda)
conda install -c conda-forge pdal python-pdal

# TLS2trees (Wilkes et al., 2023) — Phase 2 dependency
git clone https://github.com/tls-tools-ucl/TLS2trees.git
cd TLS2trees && pip install -e . && cd ..
```

Confirm the installation:

```bash
python config.py
```

The verification script reports the Python version, CUDA availability (optional, only used by Phase 2), and the status of all required modules.

### Hardware notes

The pipeline was developed and tested on an HP Z4 G5 workstation (Intel Xeon w3-2423 at 2.11 GHz, 64 GB DDR5 RAM, NVIDIA RTX A2000 12 GB, Windows 11). A representative full run on five plots (≈ 4.6 GB of raw clouds, ≈ 290 trees) takes ≈ 6.5 hours on this configuration. The pipeline is single-machine and does not require distributed execution.

---

## Quick start

```bash
# 1. Place your raw .las file(s) in ./input/
cp /path/to/your/cloud.las ./input/

# 2. Run the full pipeline (Phases 1 → 4) with the maritime pine preset
python executar.py
```

Per-plot outputs are written to `./output/<plot_name>_<timestamp>/`. The principal deliverable per tree is the row in `fase4/summary_volume_v8.csv`, which carries the reported DBH, height, volume, model selected, extrapolated fraction, sigma of the per-slice residual, quality flag (`alta` / `media` / `baixa`) and per-tree uncertainty value.

To re-run only one of the phases (for instance after changing a calibration constant in Phase 4):

```bash
# Re-run Phase 4 only (~1–2 minutes)
python rerun_fase4_only.py

# Re-run Phases 2.5 + 3 + 4 (~3–5 minutes)
python rerun_fase2_5_3_4_all.py
```

---

## Repository structure

```
.
├── config.py                     # Central configuration + species presets
├── executar.py                   # End-to-end orchestrator (Phases 1 → 4)
├── executar_trajeto.py           # Variant with externally supplied trajectory
├── fase1_pdal.py                 # Phase 1: PDAL pre-processing
├── fase2_instance.py             # Phase 2: TLS2trees instance segmentation
├── fase2_5_per_tree_setup.py     # Phase 2.5: per-tree setup
├── fase3_v8_taper.py             # Phase 3: stem reconstruction (v8)
├── fase4_v8_volume.py            # Phase 4: volume integration (v8)
├── taper_models.py               # Species shape, Kozak, power-law
├── logging_utils.py              # Logger setup
├── requirements.txt              # Python dependencies
├── EXPLICACAO_PIPELINE_v8.md     # Detailed v8 design rationale (Portuguese)
└── README.md                     # This file
```

---

## Configuration

All tunable parameters live in `config.py`. The principal entry points are:

- **`get_preset(species, base_dir)`** returns a `PipelineConfig` with the parameters tuned for the requested species. The preset shipped here, `"pinheiro"` (maritime pine), is the one used in the paper.
- **`PipelineConfig` dataclass** carries every parameter of every phase (CSF settings, RANSAC bounds, DBH slab geometry, calibration constants, etc.) and is serialised to a JSON sidecar alongside the run output so each result is reproducible.

Calibration constants of the v8 design (see `fase4_v8_volume.py`):

| Constant | Value | Function |
| --- | --- | --- |
| `VOLUME_CALIB_GLOBAL` | 0.9659 | Single-factor volume calibration |
| `DBH_REPORT_CALIB_GLOBAL` | 1.0549 | Single-factor DBH calibration |
| `H_LEAFON_CORRECTION` | 1.0270 | Leaf-on height correction |
| `FF_LO`, `FF_HI` | 0.34, 0.62 | Form-factor sanity clamp bounds |

To recalibrate the four constants on a new destructive reference, see the `calibra_lopo.py` script (leave-one-plot-out cross-validation).

---

## Outputs

Per plot, in `./output/<plot_name>_<timestamp>/`:

- `fase1/<name>_preprocessed.las` — pre-processed cloud with HAG and CSF classification.
- `fase2/trees/<name>_T<id>.leafoff.ply` and `<name>_T<id>.leafon.ply` — per-tree segmented clouds.
- `fase2_5/trees/<name>_T<id>_setup.json` — per-tree base height, prior DBH, leaf-on height.
- `fase3/<name>_T<id>_taper_meta.json` — fitted taper parameters, residual sigma, retained-slice count, extrapolated fraction.
- `fase3/<name>_T<id>_perfil_v8.csv` — discrete profile of the fitted taper at 0.10 m intervals.
- `fase3/<name>_T<id>_extended.ply` — stem points plus taper-driven synthetic-fill vertices.
- `fase4/summary_volume_v8.csv` — final per-tree record (the principal deliverable).
- `fase4/<name>.mesh.ply` — visual watertight mesh (consistency check, not the primary volume).
- `fase4/verificacao_sanidade.txt` — plot-level diagnostic warnings.
- `mapa_arvores.png` — stand-level cartographic output (tree positions, heights by colour, volumes by marker size).

The principal `summary_volume_v8.csv` carries, per tree:

```
tree_id, status, DBH_cm, H_final_m, H_fonte,
volume_perfil_dm3, volume_mesh_visual_dm3, diff_pct,
s_top_fiavel_m, fraccao_extrapolada, sigma_residuo_mm,
incerteza_estimada_pct, taper_model, qualidade,
cobertura_DBH, n_pontos_real, flags, motivo_erro
```

---

## Data

The destructive reference used to calibrate and validate the pipeline (45 trees over 4 plots; with the Alge2 sensitivity set bringing the total to 59) is provided alongside this code at `[VERIFICAR — Zenodo DOI or equivalent]`. The raw SLAM clouds used in the paper are made available subject to the data-sharing terms of the project.

---

## How to cite

If you use this pipeline in your work, please cite the paper:

```bibtex
@article{apostolo2026hmls,
  title   = {An open-source handheld mobile LiDAR pipeline for tree detection,
             stem reconstruction, and volume estimation in Mediterranean
             {Pinus pinaster} plantations, calibrated and validated against
             destructive reference},
  author  = {Ap{\'o}stolo, Pedro and Brand{\~a}o, Ivon and Fidalgo, Beatriz
             and Salas, Ra{\'u}l and Mendes, Mateus},
  journal = {[VERIFICAR — journal]},
  year    = {2026},
  doi     = {[VERIFICAR — DOI]}
}
```

The species-shape polynomial of Eq. 2 of the paper, calibrated on 366 caliper observations across 59 destructively-measured *P. pinaster* trees, is hard-coded in `taper_models.py` as `ESPECIE_COEFS`. To extend the pipeline to a new species, calibrate an equivalent 4th-degree polynomial against destructive reference and add it as a new entry in `taper_models.py`.

---

## Licence

Released under the **MIT License**. See [`LICENSE`](./LICENSE) for the full text. In short: you are free to use, modify, redistribute and commercialise the code, with the only condition that the copyright notice is preserved.

---

## Acknowledgements

This work was developed at RCM2+ — Research Center for Management Engineering and Asset Systems, Coimbra Institute of Engineering (ISEC) of the Polytechnic University of Coimbra. We thank the field crew responsible for the destructive sampling and caliper measurements at the Alge plantation block. We acknowledge the open-source projects that the pipeline builds on, in particular PDAL (Butler et al., 2021), TLS2trees (Wilkes et al., 2023), and the broader Python scientific stack (NumPy, SciPy, scikit-learn, Open3D, laspy, plyfile, trimesh).

---

*Contact:* Pedro Apóstolo — `p.apostolo.pma2@gmail.com`. Bug reports, pull requests and questions are welcome via the GitHub issue tracker.
