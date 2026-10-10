# Adaptive Inversion of the Curie Isothermal Surface — Ordos Basin and Surrounding Regions

This repository contains the source code for the block-wise, multi-constraint and
Bayesian-optimized inversion of the **Curie isothermal surface (CIS)** described in:

> *Adaptive Inversion of the Curie Isothermal Surface in the Ordos Basin and Surrounding
> Regions Based on Multiple Constraints and Bayesian Optimization*,
> Q. Fan, G. Gao, Z. Wang, C. Bai, L. Wen — submitted to *Applied Sciences*.

The de-fractal spectral method is made adaptive: instead of selecting the fractal index,
the moving-window size and the spectral fitting bands by hand, these four parameters are
searched jointly with a **Tree-structured Parzen Estimator (TPE)** in each tectonic block,
under an objective function that combines a point-fitting term, a spatial-distribution
constraint and a valid-point coverage penalty.

---

## Repository layout

```
CurieInversion_Ordos/
├── curie_block_bayes_three_class_prior_inversion.py   # main inversion program
├── blocks/                                            # 14 tectonic-block polygons
│   └── block_1_fixed.txt ... block_14_fixed.txt
├── data/
│   ├── ordos_magnetic_anomaly.txt                     # magnetic anomaly grid
│   ├── curie_depth_reference_points.txt               # heat-flow-derived reference depths
│   └── fault_traces.txt                               # fault traces used for the block division
├── utilities/
│   ├── digitize_faults.py                             # interactive block digitizer
│   └── clean_blocks.py                                # overlap removal / gap filling
├── config.example.json
├── requirements.txt
└── README.md
```

---

## Data description

| File | Columns | Description |
| --- | --- | --- |
| `data/ordos_magnetic_anomaly.txt` | `lon lat magnetic_anomaly` | Crustal magnetic anomaly grid derived from the EMAG2v3 model, gridded at 0.05° over the study area. |
| `data/curie_depth_reference_points.txt` | `lon lat Zb_km` | Reference Curie depths obtained from measured heat flow by the heat-conduction conversion. All points are used as constraints at their own locations. |
| `data/fault_traces.txt` | `>`, then `lon, lat` per line | Multi-line fault traces in EPSG:4326, one polyline per `>` header. Used to draw block boundaries. |
| `blocks/block_<N>_fixed.txt` | `lon lat` per line | Closed polygon of tectonic block `N`, in EPSG:4326. Blocks are non-overlapping and tile the study area. |

The study area covers approximately **101–117° E, 32–42° N**.

---

## Method

### 1. De-fractal spectral analysis

For a moving window centred on each grid node, the radially averaged power spectrum of the
magnetic anomaly is computed. A fractal index correction is applied before the two
least-squares fits, and the depths are obtained as

```
Zt   =  slope of ln(power) versus k over the medium-to-high wavenumber band
Z0   =  slope of ln(power) versus k over the low wavenumber band
Zb   =  2 * Z0 - Zt
```

where `Zt` is the depth to the top of the magnetic layer, `Z0` its centroid depth and `Zb`
the depth to the bottom of the magnetic layer, taken as the Curie isothermal surface.
The factor `2*pi` is applied when converting spectral slopes to depths.

### 2. Parameters searched by Bayesian optimization

| Parameter | Meaning | Range used here |
| --- | --- | --- |
| `b2` | fractal index | 0–3.0, step 0.1 |
| `window_size` | moving-window size (°) | 2.0–5.0, step 0.2 |
| `k_cent_min`, `k_cent_max` | fitting band for `Z0` (km⁻¹) | 0.002–0.020, 0.020–0.060 |
| `k_top_min`, `k_top_max` | fitting band for `Zt` (km⁻¹) | 0.025–0.065, 0.060–0.090 |

The wavenumber ranges are expressed as real wavenumbers rather than integer indices. A
minimum band width and a minimum number of spectral points per band are enforced, and a
fallback band search together with nearest-neighbour filling is used so that the final map
contains no gaps.

### 3. Objective function

```
L = lambda_p * L_p  +  lambda_s * L_s  +  lambda_m * L_m
```

* **`L_p` — point-fitting term.** Root-mean-square difference between the inverted and the
  reference Curie depths at the valid reference points:

  ```
  L_p = sqrt( (1/Nv) * sum_i ( Zb_inv,i - Zb_ref,i )^2 )
  ```

* **`L_s` — spatial-distribution constraint.** Built from the inverted and reference depth
  fields so that the spatial pattern, not only the pointwise error, is reproduced:

  ```
  L_s = gamma_1 * L_pair + gamma_2 * L_corr + gamma_3 * L_sign
  ```

  `L_pair` compares depth differences of point pairs with a distance weight
  `omega_ij = 1 / (d_ij + 0.05)`, `L_corr` penalises departure of the Pearson correlation
  from 1, and `L_sign` penalises sign inconsistencies between the inverted and reference
  depth differences.

* **`L_m` — valid-point coverage penalty.** `L_m = 1 - Nv / N`, which discourages parameter
  combinations that leave too few valid reference points inside the moving window.

Constraint weights (`lambda_p`, `lambda_s`, `gamma_1..3`, `lambda_m`) are not tuned by hand:
they are determined by a **nested optimization**, in which an outer TPE search over the
weight space is combined with the inner optimization of the spectral parameters, and each
weight combination is scored by **spatial cross-validation** so that neighbouring reference
points cannot fall on opposite sides of the split. Weights that are optimal for the study
area are `lambda_p = 1`, `lambda_s = 1.10`, `gamma_1 = gamma_2 = 0.20`, `gamma_3 = 0.60`,
`lambda_m = 30`.

### 4. Block-wise inversion

A single set of spectral parameters cannot represent the whole study area, because the
magnetic-anomaly spectra differ between the stable basin interior and the surrounding
orogenic and rift belts. The study area is therefore divided into **14 tectonic blocks**
along major fault zones and basement boundaries. The four spectral parameters are optimized
independently inside each block, and the full map is then computed block by block with the
parameters of the block that contains each node. A final interactive review stage allows
individual blocks to be re-optimized with a global "deeper / shallower" feedback.

### 5. Uncertainty

The CIS uncertainty is computed **after** the optimization from the standard errors of the
two fitted spectral slopes; it is **not** part of the objective function. It is used on its
own to assess the reliability and spatial stability of the final model, and as a quality
flag on the output maps.

---

## Installation

Python 3.9 or later is required.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `optuna`, `matplotlib`,
`geopandas`, `shapely`.

> `geopandas` and `shapely` are only needed by the two scripts in `utilities/`. The main
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

`config.example.json`:

```json
{
  "data_file": "data/ordos_magnetic_anomaly.txt",
  "measured_zb_file": "data/curie_depth_reference_points.txt",
  "block_dir": "blocks",
  "output_dir": "Curie_Depth_Block_Bayes_ThreeClassMagPrior",
  "lon_range": [101.0, 117.0],
  "lat_range": [32.0, 42.0],
  "output_resolution": 0.2,
  "n_trials": 100
}
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
polygons. It loads the fault traces and the study-area outline, lets you draw one closed
polygon per block with the mouse (the vertices snap to the nearest fault trace or to the
outline), and writes each polygon to `block_<N>.txt`.

```bash
python utilities/digitize_faults.py --faults data/fault_traces.txt \
    --outdir blocks --lon-min 100 --lon-max 118 --lat-min 31 --lat-max 43
```

* hold the left mouse button and drag to draw one block, then release
* press `u` to undo the last block
* press `q` to quit

**`utilities/clean_blocks.py`** — repairs the digitized polygons before inversion. It
validates each geometry, removes overlaps (lower block numbers take priority), assigns any
remaining gap to the nearest block, forces every geometry to a single polygon, and writes
`block_<N>_fixed.txt`.

```bash
python utilities/clean_blocks.py --blockdir blocks --pattern "block_*_fixed.txt" --plot
```

---

## Output files

Written into `output_dir`:

| File | Content |
| --- | --- |
| `block_parameters.txt` | optimal `b2`, window size, wavenumber bands, RMSE and uncertainty for each block |
| `block_training_predictions.txt` | predicted versus reference depth at every reference point |
| `curie_depth_map.txt` | full-map Curie depth `lon lat Zb` |
| `uncertainty_map.txt` | full-map uncertainty `lon lat dZb` |
| `Zt_map.txt`, `Z0_map.txt` | full-map top and centroid depths of the magnetic layer |
| `b2_map.txt`, `window_size_map.txt` | block parameters assigned to each grid node |
| `b2_used_map.txt`, `window_size_used_map.txt` | parameters actually used, including fallback bands |
| `fill_method_map.txt` | how each node was filled (0 direct, 1 fallback band, 2/3 nearest block, 4 nearest grid) |
| `*.png` | maps of the Curie depth, uncertainty, `Zt`, `Z0`, block parameters and fill method |
| `review_maps/` | preview figures and the re-optimization history of the interactive review stage |

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
* Gravity model: WGM2012, Bureau Gravimétrique International —
  <http://bgi.omp.obs-mip.fr>
* Terrestrial heat flow: China Terrestrial Heat Flow Database —
  <http://www.heatflow.org.cn>

---

## License

Released under the MIT License. See `LICENSE`.

## Contact

Guoming Gao — School of Earth Sciences, Yunnan University, Kunming, China
