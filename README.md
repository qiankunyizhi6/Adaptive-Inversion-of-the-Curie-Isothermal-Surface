# Adaptive Inversion of the Curie Isothermal Surface — Ordos Basin and Surrounding Regions

Source code for the block-wise, multi-constraint and Bayesian-optimized inversion of the
**Curie isothermal surface (CIS)** described in:

> *Adaptive Inversion of the Curie Isothermal Surface in the Ordos Basin and Surrounding
> Regions Based on Multiple Constraints and Bayesian Optimization*,
> Q. Fan, G. Gao, Z. Wang, C. Bai, L. Wen

The de-fractal spectral method is made adaptive: instead of selecting the fractal index,
the moving-window size and the spectral fitting bands by hand, these four parameters are
searched jointly with a **Tree-structured Parzen Estimator (TPE)** inside each tectonic
block, under an objective function that combines a point-fitting term, a
spatial-distribution constraint and a valid-point coverage penalty.

![Curie depth map of the Ordos Basin and surrounding regions](figures/curie_depth_map.png)

*Curie isothermal surface obtained for the Ordos Basin and surrounding regions. The basin
interior is the deepest part of the study area (36–40 km in the western and northern parts),
whereas the surrounding tectonic belts — the northeastern Tibetan Plateau, the Qilian Fold
System, the Qinling Orogen and the Taihang Mountains–Shanxi Rift — are markedly shallower
(20–30 km).*

---

## Table of contents

- [Example results](#example-results)
- [Study area and input data](#study-area-and-input-data)
- [Method](#method)
- [Output maps](#output-maps)
- [Cross-validation](#cross-validation)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Output files](#output-files)
- [Citation](#citation)
- [Data sources](#data-sources)

---

## Example results

The numbers below come from the example run shipped with this repository
(`data/` + `blocks/`, 100 TPE trials per block, 0.2° output grid). The full report is in
[`example_output/RESULTS.txt`](example_output/RESULTS.txt).

**Fit at the 180 reference points**

| Quantity | Value |
| --- | --- |
| RMSE | **5.371 km** |
| MAE | **4.045 km** |
| Bias (predicted − reference) | −0.623 km |
| Pearson *r* | 0.756 |
| Reference depth range | 15.0 – 45.0 km |
| Predicted depth range | 21.6 – 41.9 km |

**Uncertainty**, computed after the optimization from the standard errors of the two fitted
spectral slopes and *not* used inside the objective function:

| Quantity | Value |
| --- | --- |
| Median ΔZb | **2.094 km** |
| ΔZb ≤ 5 km | 173 / 180 points |
| ΔZb ≤ 10 km | 180 / 180 points |

The misfit is dominated by a small number of isolated reference points; the median
uncertainty is an order of magnitude smaller than the depth range of the model, and the
spatial cross-validation carried out in the paper shows that the spectral parameters
optimized inside one subregion transfer to only about **43 %** of the reference points of
the whole study area, which is why the model is presented as a *regional* rather than a
locally resolved description of the Curie surface.

**Optimal spectral parameters per block** (full table in
[`example_output/block_parameters.txt`](example_output/block_parameters.txt)):

| Block | *b2* | Window (°) | Zt band (km⁻¹) | Z0 band (km⁻¹) | RMSE (km) | Reference points |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9 | 5.0 | 0.04411 – 0.07145 | 0.01593 – 0.02934 | 1.077 | 6 |
| 2 | 1.0 | 3.0 | default | default | — | 0 |
| 3 | 1.2 | 4.8 | 0.04358 – 0.08606 | 0.00533 – 0.02703 | 2.149 | 3 |
| 4 | 0.4 | 3.2 | 0.05767 – 0.08893 | 0.00826 – 0.04954 | 8.264 | 14 |
| 5 | 0.0 | 2.6 | 0.05297 – 0.07418 | 0.00356 – 0.05238 | 5.831 | 25 |
| 6 | 0.5 | 5.0 | 0.04128 – 0.06859 | 0.01690 – 0.03974 | 3.135 | 11 |
| 7 | 0.7 | 3.0 | 0.05281 – 0.08713 | 0.01706 – 0.05245 | 2.600 | 9 |
| 8 | 2.7 | 4.6 | 0.04687 – 0.06744 | 0.00204 – 0.02263 | 4.528 | 17 |
| 9 | 1.5 | 2.2 | 0.06392 – 0.07871 | 0.01154 – 0.03535 | 5.266 | 20 |
| 10 | 1.2 | 2.4 | 0.05323 – 0.08293 | 0.00298 – 0.05205 | 5.899 | 19 |
| 11 | 0.4 | 3.2 | 0.02651 – 0.08500 | 0.01792 – 0.02343 | 4.838 | 3 |
| 12 | 0.1 | 2.4 | 0.05591 – 0.06538 | 0.01915 – 0.04077 | 5.835 | 6 |
| 13 | 2.1 | 3.2 | 0.05337 – 0.07943 | 0.00479 – 0.03502 | 4.784 | 40 |
| 14 | 0.9 | 4.8 | 0.05547 – 0.08672 | 0.00628 – 0.03581 | 4.083 | 6 |

> Block 2 contains no reference point, so the optimizer falls back to the default spectral
> parameters there (`status = no_true_points_default`). The values quoted for its RMSE are
> placeholders and do not represent a fit quality.

The optimal parameters differ substantially between blocks (*b2* from 0.0 to 2.7, window
size from 2.2° to 5.0°), which is the quantitative justification for inverting the study
area block by block instead of applying one parameter set everywhere.

---

## Study area and input data

The study area covers approximately **101–117° E, 32–42° N** and includes the Ordos Basin
with the Yinshan orogenic belt to the north, the Qinling orogen to the south, the Alxa Block
and the Qilian Fold System to the west and southwest, and the Taihang Mountains–Shanxi Rift
to the east.

The block division follows the major fault zones and basement boundaries of the region. The
14 blocks and the reference points are shown below; the polygons are digitized with
`utilities/digitize_faults.py`, whose vertices snap to the mapped fault traces.

![Tectonic blocks and reference points](figures/blocks_and_reference_points.png)

| File | Columns | Description |
| --- | --- | --- |
| `data/ordos_magnetic_anomaly.txt` | `lon lat magnetic_anomaly` | Crustal magnetic anomaly grid (281 × 401 nodes, 0.05°) derived from EMAG2v3 |
| `data/curie_depth_reference_points.txt` | `lon lat Zb_km` | 180 reference Curie depths converted from measured heat flow |
| `data/fault_traces.txt` | `>` header, then `lon, lat` | Multi-line fault traces in EPSG:4326 |
| `blocks/block_<N>_fixed.txt` | `lon lat` | Closed polygon of block `N`, non-overlapping, tiling the study area |

---

## Method

### 1. De-fractal spectral analysis

For a moving window centred on each grid node, the radially averaged power spectrum of the
magnetic anomaly is computed. A fractal index correction is applied before the two
least-squares fits, and the depths follow from the slopes of `ln(power)` against the
wavenumber `k`:

```
Zt = slope over the medium-to-high wavenumber band      (top of the magnetic layer)
Z0 = slope over the low wavenumber band                 (centroid of the magnetic layer)
Zb = 2 * Z0 - Zt                                        (bottom = Curie isothermal surface)
```

The factor `2π` is applied when converting the spectral slopes to depths.

### 2. Parameters searched by Bayesian optimization

| Parameter | Meaning | Range used here |
| --- | --- | --- |
| `b2` | fractal index | 0 – 3.0, step 0.1 |
| `window_size` | moving-window size (°) | 2.0 – 5.0, step 0.2 |
| `k_cent_min`, `k_cent_max` | fitting band for `Z0` (km⁻¹) | 0.002 – 0.020, 0.020 – 0.060 |
| `k_top_min`, `k_top_max` | fitting band for `Zt` (km⁻¹) | 0.025 – 0.065, 0.060 – 0.090 |

The wavenumber ranges are expressed as real wavenumbers rather than integer indices. A
minimum band width and a minimum number of spectral points per band are enforced.

### 3. Objective function

```
L = lambda_p * L_p  +  lambda_s * L_s  +  lambda_m * L_m
```

* **`L_p` — point-fitting term**

  ```
  L_p = sqrt( (1/Nv) * sum_i ( Zb_inv,i - Zb_ref,i )^2 )
  ```

  All valid reference points enter with equal weight, so no east–west depth trend is
  imposed a priori.

* **`L_s` — spatial-distribution constraint**

  ```
  L_s = gamma_1 * L_pair + gamma_2 * L_corr + gamma_3 * L_sign
  ```

  `L_pair` compares the depth differences of point pairs with the distance weight
  `omega_ij = 1 / (d_ij + 0.05)`, `L_corr` penalises departure of the Pearson correlation
  from 1, and `L_sign` penalises sign inconsistencies between the inverted and the reference
  depth differences.

* **`L_m` — valid-point coverage penalty**

  ```
  L_m = 1 - Nv / N
  ```

  which discourages parameter combinations that leave too few valid reference points inside
  the moving window.

The weights are **not** tuned by hand. They are determined by a nested optimization in which
an outer TPE search over the weight space is combined with the inner optimization of the
spectral parameters, and each weight combination is scored by **spatial cross-validation**
so that neighbouring reference points cannot fall on opposite sides of the split. The values
that are optimal for this study area are `lambda_p = 1`, `lambda_s = 1.10`,
`gamma_1 = gamma_2 = 0.20`, `gamma_3 = 0.60`, `lambda_m = 30`.

### 4. Block-wise inversion

A single parameter set cannot represent the whole study area, because the magnetic-anomaly
spectra differ between the stable basin interior and the surrounding orogenic and rift
belts. The four spectral parameters are therefore optimized independently inside each of the
14 blocks, and the full map is computed block by block. A final interactive review stage
allows individual blocks to be re-optimized with a global "deeper / shallower" feedback.

Each review round re-optimizes one block and records the parameters, RMSE, spatial loss,
pairwise-difference RMSE, spatial correlation and sign-mismatch rate before and after the
change; the log is written to `review_maps/review_history.json`.

![Block review round 5](figures/block_review_round_05.png)

*Interactive review: the 14 block polygons, the reference points (open circles) and the
Curie depth obtained after five review rounds.*

---

## Output maps

**Curie depth and its uncertainty** — the uncertainty is computed after the optimization
from the standard errors of the two fitted spectral slopes and is used only for the
independent assessment of the model:

| Curie depth | Uncertainty ΔZb |
| --- | --- |
| ![Curie depth](figures/curie_depth_map.png) | ![Curie depth uncertainty](figures/curie_depth_uncertainty.png) |

**Parameters recovered by the optimizer** — the fractal index and the window size that were
selected for each block, together with the fill method of every node:

| Fractal index *b2* | Window size (°) | Fill method |
| --- | --- | --- |
| ![Fractal index map](figures/fractal_index_map.png) | ![Window size map](figures/window_size_map.png) | ![Fill method map](figures/fill_method_map.png) |

The fill-method map documents how each node was obtained: `0` direct fit, `1` fallback
spectral band, `2/3` nearest block, `4` nearest grid node. In the example run 179 of the 180
nodes are direct fits (`direct`), with a single node filled by the fallback band.

---

## Cross-validation

The 625 heat-flow sites of the region are split into five **spatially separated** folds, so
that neighbouring sites cannot fall on opposite sides of the split, and each candidate
Curie-depth field is converted to surface heat flow on the held-out fold.

![Five-fold spatial cross-validation splits](validation/five_fold_spatial_cv_splits.png)

*Five-fold spatial split of the heat-flow sites; the last panel shows all 625 sites. The
folds are built block by block so that the test points are spatially separated from the
training points.*

Heat flow is used to build the reference Curie depths, so its agreement with a Curie-depth
model is **not** an independent measure of accuracy and is **not** used to tune the model
here. It is reported in [`validation/cross_validation_summary.txt`](validation/cross_validation_summary.txt)
only so that the recovered field can be compared with published models in a common currency:

| Model | RMSE (mW m⁻²) | MAE (mW m⁻²) | Correlation |
| --- | --- | --- | --- |
| *final_curie_depth* (this study) | **33.48 ± 0.93** | **28.90 ± 0.79** | 0.158 ± 0.067 |
| *data_jlm* (this study) | 34.97 ± 1.22 | 31.33 ± 0.64 | 0.053 ± 0.028 |
| *curie_depth_map* (this study) | 35.95 ± 1.22 | 32.50 ± 0.68 | 0.201 ± 0.055 |
| 41598_2017_srep45129 | 46.58 ± 0.55 | 42.87 ± 0.90 | 0.217 ± 0.067 |
| *curie_isotherm_depth_xyz* | 54.60 ± 1.55 | 48.88 ± 1.66 | 0.113 ± 0.042 |

The RMSE and MAE of the models produced in this study are the lowest of the set, while the
linear correlation with the observed heat flow stays low for **every** model, including the
published ones. The low correlation is therefore a property of the heat-flow-to-Curie-depth
comparison itself, controlled jointly by the adopted thermophysical parameters, by the
conversion model and by the sparse and uneven distribution of the heat-flow sites; it is not
a specific deficiency of one inversion.

---

## Repository layout

```
CurieInversion_Ordos/
├── curie_block_bayes_three_class_prior_inversion.py   # main inversion program
├── blocks/                                            # 14 tectonic-block polygons
├── data/                                              # magnetic anomaly, reference depths, faults
├── utilities/
│   ├── digitize_faults.py                             # interactive block digitizer
│   └── clean_blocks.py                                # overlap removal / gap filling
├── figures/                                           # example result figures
├── example_output/                                    # example run: parameters, predictions, report
├── validation/                                        # spatial cross-validation materials
├── config.example.json
├── requirements.txt
└── LICENSE
```

---

## Installation

Python 3.9 or later is required.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `optuna`, `matplotlib`,
`geopandas`, `shapely`.

> `geopandas` and `shapely` are needed only by the two scripts in `utilities/`. The main
> inversion program needs only `numpy`, `scikit-learn`, `optuna` and `matplotlib`.

---

## Usage

### Main inversion

Run from the repository root so that the relative paths resolve:

```bash
python curie_block_bayes_three_class_prior_inversion.py
```

All settings have defaults that point at the files in this repository. They can be
overridden with a JSON configuration file:

```bash
python curie_block_bayes_three_class_prior_inversion.py --config config.example.json
```

| Setting | Meaning |
| --- | --- |
| `data_file` | magnetic anomaly grid, three columns `lon lat value` |
| `measured_zb_file` | reference Curie depths, three columns `lon lat Zb_km` |
| `block_dir` | directory holding `block_<N>_fixed.txt` |
| `output_dir` | directory for all output files and figures |
| `lon_range`, `lat_range` | extent of the output map |
| `output_resolution` | output grid spacing in degrees |
| `n_trials` | number of TPE trials per block |

### Utilities

**`utilities/digitize_faults.py`** — interactive digitizer used to create the block
polygons. Each polygon drawn with the mouse has its vertices snapped to the nearest fault
trace or to the study-area outline, and is written to `block_<N>.txt`.

```bash
python utilities/digitize_faults.py --faults data/fault_traces.txt \
    --outdir blocks --lon-min 100 --lon-max 118 --lat-min 31 --lat-max 43
```

* hold the left mouse button and drag to draw one block, then release
* press `u` to undo the last block, `q` to quit

**`utilities/clean_blocks.py`** — repairs the digitized polygons before inversion: validates
the geometries, removes overlaps (lower block numbers take priority), assigns any remaining
gap to the nearest block, forces every geometry to a single polygon, and writes
`block_<N>_fixed.txt`.

```bash
python utilities/clean_blocks.py --blockdir blocks --pattern "block_*_fixed.txt" --plot
```

---

## Output files

Written into `output_dir`:

| File | Content |
| --- | --- |
| `block_parameters.txt` | optimal `b2`, window size, wavenumber bands, RMSE and uncertainty per block |
| `block_training_predictions.txt` | predicted versus reference depth at every reference point |
| `curie_depth_map.txt` | full-map Curie depth `lon lat Zb` |
| `uncertainty_map.txt` | full-map uncertainty `lon lat dZb` |
| `Zt_map.txt`, `Z0_map.txt` | full-map top and centroid depths of the magnetic layer |
| `b2_map.txt`, `window_size_map.txt` | block parameters assigned to each grid node |
| `b2_used_map.txt`, `window_size_used_map.txt` | parameters actually used, including fallback bands |
| `fill_method_map.txt` | how each node was filled |
| `raw_nan_mask_map.txt` | mask of the nodes left empty by the direct fit |
| `*.png` | maps of the Curie depth, uncertainty, `Zt`, `Z0`, block parameters and fill method |
| `review_maps/` | preview figures, per-round parameters and the re-optimization history |

---

## Citation

If you use this code, please cite:

```
Fan, Q.; Gao, G.; Wang, Z.; Bai, C.; Wen, L. Adaptive Inversion of the Curie Isothermal
Surface in the Ordos Basin and Surrounding Regions Based on Multiple Constraints and
Bayesian Optimization. Applied Sciences (under review).
```

---

## Data sources

* Magnetic anomaly: EMAG2v3, National Centers for Environmental Information (NCEI) —
  <https://www.ncei.noaa.gov/products/emag2v3>
* Crustal model: CRUST1.0 — <https://igppweb.ucsd.edu/~gabi/crust1.html>
* Gravity model: WGM2012, Bureau Gravimétrique International — <http://bgi.omp.obs-mip.fr>
* Terrestrial heat flow: China Terrestrial Heat Flow Database — <http://www.heatflow.org.cn>

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

## Contact

Guoming Gao — School of Earth Sciences, Yunnan University, Kunming, China
