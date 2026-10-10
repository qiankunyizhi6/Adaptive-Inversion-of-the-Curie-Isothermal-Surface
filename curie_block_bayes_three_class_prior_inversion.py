# -*- coding: utf-8 -*-
"""
Block-wise Bayesian (TPE) inversion of the Curie isothermal surface,
constrained by heat-flow-derived reference depths.
----------------------------------------------------------------------------
Main features
1. Block-wise inversion: the spectral parameters are optimized independently
   inside each tectonic block, and the full map is then computed with the
   parameters of the block that contains each node.
2. The Bayesian search is performed over real wavenumber ranges rather than
   over integer spectral indices:
   - k_top_min,  k_top_max : fitting band for the top depth Zt (higher wavenumbers)
   - k_cent_min, k_cent_max: fitting band for the centroid depth Z0 (lower wavenumbers)
3. Heat flow is not recomputed and the heat-flow RMSE is not used as an objective.
4. The reference Curie depths Zb_ref(km) are used directly in the optimization.
5. The objective function does not rely on the block RMSE alone; a spatial
   distribution constraint is added on top of the point-fitting term:
   - pointwise RMSE;
   - RMSE of the depth differences between point pairs (spatial relief);
   - correlation loss (overall spatial trend);
   - sign consistency of the depth differences (relative deep/shallow pattern).
6. All valid reference points enter the point-fitting term with equal weight, so
   that no east-west depth trend is imposed a priori.
7. Uncertainty is not used as a hard rejection criterion: parameter
   combinations with dZb <= 5 km are preferred, and if that cannot be met the
   uncertainty is kept within 10 km where possible.
8. When the full map is computed, a fallback-band search and nearest-neighbour
   filling are applied so that no node of the study area is left empty.
9. b2_map / window_size_map report the block-optimized parameters without the
   speckle introduced by the fallback bands; the parameters actually used are
   stored separately in b2_used_map / window_size_used_map.
10. After the first optimization a preview figure and a console review step allow
    individual blocks to be re-optimized, with an optional instruction to shift
    the next round towards deeper or shallower values.

Inputs (default paths are relative to this file)
1. data/ordos_magnetic_anomaly.txt        three columns: lon lat magnetic_anomaly
2. data/curie_depth_reference_points.txt  three columns: lon lat Zb_ref(km)
3. blocks/block_*_fixed.txt               polygon of each tectonic block

Outputs (written to output_dir)
1. block_parameters.txt            optimal b2, window size, wavenumber bands, RMSE and uncertainty per block
2. block_training_predictions.txt  predicted versus reference depth at each reference point
3. curie_depth_map.txt              full-map Curie depth, lon lat Zb
4. uncertainty_map.txt             full-map uncertainty, lon lat dZb
5. Zt_map.txt                      full-map depth to the top of the magnetic layer
6. Z0_map.txt                      full-map centroid depth of the magnetic layer
7. b2_map.txt                      fractal index used at each grid node
8. window_size_map.txt             window size used at each grid node
"""

import os
import random
import hashlib
import warnings
import re
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=RuntimeWarning)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


# ============================================================
# 0. 随机种子设置
# ============================================================
GLOBAL_SEED = 42
# 关闭后沿用一次优化直接输出的流程。
ENABLE_INTERACTIVE_REVIEW = True


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """固定 Python random、NumPy 和 Optuna 相关随机性。

    注意：如果想让 Python 的哈希随机化也完全固定，建议在命令行运行前设置：
    Windows CMD: set PYTHONHASHSEED=42
    PowerShell:  $env:PYTHONHASHSEED="42"
    Linux/macOS: export PYTHONHASHSEED=42
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def stable_seed_from_name(name: str, base_seed: int = GLOBAL_SEED) -> int:
    """根据块体名称生成稳定的块体专属种子，避免使用 Python 内置 hash。"""
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return int((base_seed + offset) % (2 ** 32 - 1))


set_global_seed(GLOBAL_SEED)


# ============================================================
# 1. 可调参数区
# ============================================================
PARAM_SPACE = {
    # 分形指数搜索范围
    "b2_min": 0.0,
    "b2_max": 3.0,
    "b2_step": 0.1,

    # 窗口大小搜索范围，单位：度
    # 为了减少空值，这里加入 2.5、3.5、5.0 作为候选窗口。
    "window_size_choices": [2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0],

    # --------------------------------------------------------
    # 波数范围搜索空间，单位：km^-1
    # 原则：Z0 使用低波数段；Zt 使用中高波数段。
    # 由于 dl=5.5 km 时 Nyquist 波数约为 0.0909 km^-1，
    # 因此高波数上限不要设置得过高，否则可用谱点会明显减少。
    # --------------------------------------------------------
    "k_cent_min_range": (0.002, 0.020),
    "k_cent_max_range": (0.020, 0.060),

    "k_top_min_range": (0.025, 0.065),
    "k_top_max_range": (0.060, 0.090),

    # 频带最小宽度，单位：km^-1
    "min_band_width": 0.004,

    # 每个拟合波数段内至少需要的谱点数。
    # 为了优先保证全图无空值，这里允许 2 个点拟合；
    # 当只有 2 个点时，不确定性会用保守代理值估计。
    "min_points_per_band": 2,

    # 居里面深度合理范围，单位：km
    "zb_min": 18.0,
    "zb_max": 42.0,

    # 不确定性参数仅用于反演完成后的 dZb 输出和独立验证，不进入优化目标。
    "preferred_uncertainty_km": 5.0,
    "soft_uncertainty_km": 10.0,

    # 是否要求中心深度拟合频段完全低于顶界拟合频段。
    # 这里保留 True，确保 Z0 属于低波数域、Zt 属于中高波数域。
    "require_separated_bands": True,

    # 径向平均谱的 CF 筛选阈值。原程序是 1.0；这里放宽到 1.5，减少频谱点不足导致的空值。
    "spectrum_cf_max": 1.5,

    # 目标函数权重。
    "uncertainty_weight": 0.05,
    "uncertainty_excess_weight": 0.20,
    "high_uncertainty_weight": 0.60,

    # 有效点覆盖率惩罚。缺点越多，loss 越大，促使优化器优先选择覆盖更多真值点的参数。
    "missing_point_penalty": 20.0,

    # --------------------------------------------------------
    # 空间分布约束权重。
    # 这部分解决“块体整体 RMSE 小，但空间高低分布不对”的问题。
    # pairwise_difference 约束任意两个真值点之间的深度差；
    # correlation 约束预测值与真值的空间变化趋势；
    # sign_mismatch 约束深浅相对关系是否反了。
    # --------------------------------------------------------
    "spatial_distribution_weight": 0.80,
    "pairwise_difference_weight": 0.70,
    "correlation_weight": 1.20,
    "sign_mismatch_weight": 0.80,
    "spatial_min_points": 3,
    "spatial_sign_threshold_km": 0.50,
    "max_spatial_points_for_pairs": 180,

    # 不确定性置信因子.
    "uncertainty_confidence_factor": 1.0,

    # 只有两个谱点时的斜率标准误差代理值。
    # 该值会对应大约数 km 级别的 ΔZb，用于避免 2 点拟合被误判为零不确定性。
    "two_point_slope_se_proxy": 20.0,

    # 一个块体内至少需要多少个有效真值点参与 RMSE。
    "min_valid_points": 1,
}

DEFAULT_BLOCK_PARAMS = {
    "b2": 1.0,
    "window_size": 3.0,
    "k_top_min": 0.030,
    "k_top_max": 0.080,
    "k_cent_min": 0.005,
    "k_cent_max": 0.055,
    "rmse": 1000.0,
    "mean_uncertainty": np.nan,
    "valid_points": 0,
    "spatial_loss": np.nan,
    "pairwise_diff_rmse": np.nan,
    "spatial_corr": np.nan,
    "sign_mismatch_rate": np.nan,
    "status": "default",
}


# ============================================================
# 1. 数据读取与块体工具
# ============================================================
def load_measured_curie_depth(measured_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载居里面真值数据，格式为三列：lon lat Zb_true(km)。"""
    data = np.loadtxt(measured_file)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("居里面真值文件格式应为三列：lon lat Zb_true(km)")

    lons = data[:, 0].astype(float)
    lats = data[:, 1].astype(float)
    zbs = data[:, 2].astype(float)
    return lons, lats, zbs


def load_block_polygons(block_dir: str) -> Dict[str, List[Tuple[float, float]]]:
    """读取块体边界文件：block_*_fixed.txt。"""
    block_polygons = {}
    block_files = [
        f for f in os.listdir(block_dir)
        if f.startswith("block_") and f.endswith("_fixed.txt")
    ]

    for block_file in sorted(block_files):
        block_name = block_file.replace("_fixed.txt", "")
        file_path = os.path.join(block_dir, block_file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                coords = list(map(float, content.split()))

            polygon = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    polygon.append((coords[i], coords[i + 1]))

            if len(polygon) >= 3:
                block_polygons[block_name] = polygon
            else:
                print(f"⚠️ {block_file} 点数不足，已跳过")

        except Exception as e:
            print(f"读取 {block_file} 时出错: {e}")

    return block_polygons


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """射线法判断点是否在多边形内部。"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]

    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    xinters = p1x
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def get_points_in_block(
    block_polygon: List[Tuple[float, float]],
    measured_lons: np.ndarray,
    measured_lats: np.ndarray,
    measured_zbs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取某个块体内部的居里面真值点。"""
    block_lons, block_lats, block_zbs = [], [], []

    for lon, lat, zb in zip(measured_lons, measured_lats, measured_zbs):
        if point_in_polygon((lon, lat), block_polygon):
            block_lons.append(lon)
            block_lats.append(lat)
            block_zbs.append(zb)

    return np.array(block_lons), np.array(block_lats), np.array(block_zbs)


def find_block_for_point(
    point: Tuple[float, float],
    block_polygons: Dict[str, List[Tuple[float, float]]],
) -> Optional[str]:
    """查找某个点所属块体。"""
    for block_name, polygon in block_polygons.items():
        if point_in_polygon(point, polygon):
            return block_name
    return None


# ============================================================
# 2. 磁异常网格与频谱分析
# ============================================================
def prepare_magnetic_grid(data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取磁异常三列数据，并转为规则网格。"""
    mag_data = np.loadtxt(data_file)
    if mag_data.ndim != 2 or mag_data.shape[1] < 3:
        raise ValueError("磁异常文件格式应为三列：lon lat magnetic_anomaly")

    lon_grid = np.unique(mag_data[:, 0])
    lat_grid = np.unique(mag_data[:, 1])
    value_grid = np.full((len(lat_grid), len(lon_grid)), np.nan)

    for lon, lat, value in mag_data[:, :3]:
        lon_idx = np.argmin(np.abs(lon_grid - lon))
        lat_idx = np.argmin(np.abs(lat_grid - lat))
        value_grid[lat_idx, lon_idx] = value

    return lon_grid, lat_grid, value_grid


def detrend_2d(Z: np.ndarray) -> np.ndarray:
    """二维平面去趋势。"""
    N, M = Z.shape
    X, Y = np.meshgrid(np.arange(1, M + 1), np.arange(1, N + 1))
    Xcol, Ycol, Zcol = X.ravel(), Y.ravel(), Z.ravel()

    finite = np.isfinite(Zcol)
    if finite.sum() < 4:
        raise ValueError("有效数据点过少，无法去趋势")

    A = np.column_stack((Xcol[finite], Ycol[finite], np.ones(finite.sum())))
    Coefficients, *_ = np.linalg.lstsq(A, Zcol[finite], rcond=None)
    XCoeff, YCoeff, CCoeff = Coefficients
    Z_p = XCoeff * X + YCoeff * Y + CCoeff
    return Z - Z_p


def raas(anom: np.ndarray, dl: float = 5.5) -> np.ndarray:
    """
    径向平均振幅谱。

    返回数组列：
    0: 波数 w，单位 km^-1
    1: ln(EA)
    2: 谱估计置信项 CF
    3: 每个环带样本数 nm
    """
    ny, nx = anom.shape
    N = max(nx, ny)
    if N % 2 != 0:
        N += 1

    TFA = np.fft.fftshift(np.fft.fft2(anom, s=(N, N)))
    EA = 2 * np.abs(TFA) / (ny * nx)

    n = np.arange(-N // 2, N // 2)
    NX, NY = np.meshgrid(n, n)
    rho = np.hypot(NX, NY)
    rho = np.round(rho).astype(int)

    num_bins = N // 2 + 1
    Pf = np.zeros(num_bins)
    CF = np.zeros(num_bins)
    nm = np.zeros(num_bins)
    eps = 1e-10

    for r in range(num_bins):
        idx = np.where(rho == r)
        count = idx[0].size
        nm[r] = count
        if count > 0:
            log_EA = np.log(np.where(EA[idx] == 0, eps, EA[idx]))
            Pf[r] = np.nanmean(log_EA)
            CF[r] = 1.96 * np.nanstd(log_EA) / np.sqrt(count)
        else:
            Pf[r] = np.nan
            CF[r] = np.nan

    kn = 1 / (2 * dl)
    k0 = 1 / (N * dl)
    w = np.arange(0, kn + k0 / 2, k0)

    cv2 = np.column_stack((w[1:], Pf[1:], CF[1:], nm[1:]))

    # 保留谱估计较稳定的点；如果你想放宽频谱可用点，可把 1 改大，例如 1.5。
    cv3 = cv2[np.isfinite(cv2[:, 1]) & np.isfinite(cv2[:, 2]) & (cv2[:, 2] <= PARAM_SPACE["spectrum_cf_max"])]
    return cv3


def fractal_correction(x: np.ndarray, b1: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    去分形校正。

    返回：
    w  : 波数 km^-1
    pw : 用于顶界 Zt 拟合的谱
    ps : 用于中心深度 Z0 拟合的缩放谱
    ci : 谱估计置信项
    """
    w = x[:, 0]
    eps = 1e-12
    ww = np.maximum(2 * np.pi * w, eps)

    ps = (x[:, 1] - np.log(ww)) + np.log(ww ** (0.5 * b1))
    pw = x[:, 1] + np.log(ww ** (0.5 * b1))
    return w, pw, ps, x[:, 2]


def linear_fit_with_uncertainty(
    x: np.ndarray,
    y: np.ndarray,
    confidence_factor: float = 1.0,
) -> Optional[Tuple[np.ndarray, float, float]]:
    """
    一元线性拟合，并估计斜率标准误差。

    返回：
    p        : np.polyfit 得到的 [slope, intercept]
    slope_se : slope 的标准误差，乘以 confidence_factor
    rmse_fit : 拟合残差 RMSE

    说明：
    - 标准线性回归至少需要 3 个点才能估计残差方差；
    - 为了减少空值，若某个波段只有 2 个谱点，仍允许拟合，
      但给定一个保守的 slope_se 代理值，避免不确定性被低估为 0。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return None

    if np.nanstd(x) <= 0:
        return None

    try:
        p = np.polyfit(x, y, 1)
    except Exception:
        return None

    y_hat = np.polyval(p, x)
    residual = y - y_hat
    rmse_fit = float(np.sqrt(np.mean(residual ** 2))) if len(residual) > 0 else np.nan

    if len(x) == 2:
        slope_se = confidence_factor * PARAM_SPACE["two_point_slope_se_proxy"]
        return p, float(slope_se), rmse_fit

    dof = len(x) - 2
    if dof <= 0:
        return None

    s2 = np.sum(residual ** 2) / dof
    s = np.sqrt(max(s2, 0.0))
    sxx = np.sum((x - np.mean(x)) ** 2)
    if sxx <= 0:
        return None

    slope_se = confidence_factor * s / np.sqrt(sxx)
    return p, float(slope_se), rmse_fit


def extract_window_data(
    lon0: float,
    lat0: float,
    window_size: float,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    value_grid: np.ndarray,
) -> Optional[np.ndarray]:
    """从磁异常规则网格中提取指定中心点和窗口大小的数据。"""
    lon_min = lon0 - window_size / 2
    lon_max = lon0 + window_size / 2
    lat_min = lat0 - window_size / 2
    lat_max = lat0 + window_size / 2

    lon_mask = (lon_grid >= lon_min) & (lon_grid <= lon_max)
    lat_mask = (lat_grid >= lat_min) & (lat_grid <= lat_max)

    if lon_mask.sum() < 2 or lat_mask.sum() < 2:
        return None

    lon_idx1, lon_idx2 = np.where(lon_mask)[0][[0, -1]]
    lat_idx1, lat_idx2 = np.where(lat_mask)[0][[0, -1]]
    window_data = value_grid[lat_idx1:lat_idx2 + 1, lon_idx1:lon_idx2 + 1]

    if window_data.size < 4 or np.isnan(window_data).all():
        return None

    # 如窗口内存在少量 NaN，用窗口均值填充，避免最小二乘和 FFT 崩溃。
    if np.isnan(window_data).any():
        mean_val = np.nanmean(window_data)
        if not np.isfinite(mean_val):
            return None
        window_data = np.where(np.isfinite(window_data), window_data, mean_val)

    return window_data


def compute_curie_for_single_point(
    lon0: float,
    lat0: float,
    params: Dict[str, float],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    value_grid: np.ndarray,
    calculate_uncertainty: bool = True,
) -> Optional[Dict[str, float]]:
    """对单个点计算 Zt、Z0、Zb；需要时再计算 ΔZb。"""
    b2 = float(params["b2"])
    window_size = float(params["window_size"])
    k_top_min = float(params["k_top_min"])
    k_top_max = float(params["k_top_max"])
    k_cent_min = float(params["k_cent_min"])
    k_cent_max = float(params["k_cent_max"])

    # 确保 Z0 为低波数域，Zt 为中高波数域。
    if PARAM_SPACE["require_separated_bands"] and not (k_cent_max < k_top_min):
        return None

    window_data = extract_window_data(lon0, lat0, window_size, lon_grid, lat_grid, value_grid)
    if window_data is None:
        return None

    try:
        dt = detrend_2d(window_data)
    except Exception:
        return None

    a2 = raas(dt, dl=5.5)
    if a2.size == 0:
        return None

    w, pw, ps, ci = fractal_correction(a2, b2)

    top_mask = (w >= k_top_min) & (w <= k_top_max)
    cent_mask = (w >= k_cent_min) & (w <= k_cent_max)

    if top_mask.sum() < PARAM_SPACE["min_points_per_band"]:
        return None
    if cent_mask.sum() < PARAM_SPACE["min_points_per_band"]:
        return None

    top_fit = linear_fit_with_uncertainty(
        w[top_mask],
        pw[top_mask],
        confidence_factor=PARAM_SPACE["uncertainty_confidence_factor"],
    )
    cent_fit = linear_fit_with_uncertainty(
        w[cent_mask],
        ps[cent_mask],
        confidence_factor=PARAM_SPACE["uncertainty_confidence_factor"],
    )

    if top_fit is None or cent_fit is None:
        return None

    p_top, slope_se_top, rmse_top = top_fit
    p_cent, slope_se_cent, rmse_cent = cent_fit

    # 深度参数，单位 km，因为 w 的单位为 km^-1
    Zt = -p_top[0] / (2 * np.pi)
    Z0 = -p_cent[0] / (2 * np.pi)
    Zb = 2 * Z0 - Zt

    # 不确定性传播
    dZt = abs(slope_se_top) / (2 * np.pi)
    dZ0 = abs(slope_se_cent) / (2 * np.pi)
    dZb = np.sqrt((2 * dZ0) ** 2 + dZt ** 2) if calculate_uncertainty else np.nan

    if not (np.isfinite(Zt) and np.isfinite(Z0) and np.isfinite(Zb)):
        return None

    # 只保留基本深度范围。ΔZb 不再硬剔除，避免大片空值。
    if not (PARAM_SPACE["zb_min"] <= Zb <= PARAM_SPACE["zb_max"]):
        return None

    return {
        "Zt": float(Zt),
        "Z0": float(Z0),
        "Zb": float(Zb),
        "dZt": float(dZt),
        "dZ0": float(dZ0),
        "dZb": float(dZb),
        "rmse_fit_top": float(rmse_top),
        "rmse_fit_cent": float(rmse_cent),
        "n_top": int(top_mask.sum()),
        "n_cent": int(cent_mask.sum()),
    }


def compute_curie_for_measured_points(
    params: Dict[str, float],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    value_grid: np.ndarray,
    measured_lons: np.ndarray,
    measured_lats: np.ndarray,
    calculate_uncertainty: bool = True,
) -> Dict[str, np.ndarray]:
    """对一个块体内的所有居里面真值点计算预测 Zb。"""
    n = len(measured_lons)
    pred_zbs = np.full(n, np.nan)
    uncertainties = np.full(n, np.nan)
    zts = np.full(n, np.nan)
    z0s = np.full(n, np.nan)

    for i, (lon0, lat0) in enumerate(zip(measured_lons, measured_lats)):
        res = compute_curie_for_single_point(
            lon0=float(lon0),
            lat0=float(lat0),
            params=params,
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            value_grid=value_grid,
            calculate_uncertainty=calculate_uncertainty,
        )
        if res is None:
            continue

        pred_zbs[i] = res["Zb"]
        uncertainties[i] = res["dZb"]
        zts[i] = res["Zt"]
        z0s[i] = res["Z0"]

    return {
        "pred_zbs": pred_zbs,
        "uncertainties": uncertainties,
        "Zt": zts,
        "Z0": z0s,
    }


def calculate_spatial_distribution_metrics(
    measured_lons: np.ndarray,
    measured_lats: np.ndarray,
    true_zbs: np.ndarray,
    pred_zbs: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, float]:
    """
    计算空间分布一致性指标。

    目的：避免目标函数只追求块体整体 RMSE，而忽略块体内部深浅高低的空间格局。

    指标含义：
    - pairwise_diff_rmse：任意两个真值点之间的“深度差”误差，越小表示空间起伏越一致；
    - spatial_corr：预测 Zb 与真值 Zb 的相关系数，越接近 1 表示空间趋势越一致；
    - sign_mismatch_rate：深浅相对关系反号的比例，越小越好；
    - spatial_loss：综合空间损失，单位近似为 km，可直接叠加到优化目标。
    """
    valid = (
        valid_mask
        & np.isfinite(measured_lons)
        & np.isfinite(measured_lats)
        & np.isfinite(true_zbs)
        & np.isfinite(pred_zbs)
    )
    idx = np.where(valid)[0]
    n = len(idx)

    out = {
        "spatial_loss": 0.0,
        "pairwise_diff_rmse": np.nan,
        "spatial_corr": np.nan,
        "corr_loss": 0.0,
        "sign_mismatch_rate": np.nan,
        "n_spatial_points": int(n),
    }

    if n < 2:
        # 只有 0 或 1 个有效点时无法描述空间分布；交给 missing_penalty 处理。
        return out

    # 为避免点数很多时 O(n^2) 过慢，做确定性抽样。
    max_n = int(PARAM_SPACE["max_spatial_points_for_pairs"])
    if n > max_n:
        pick = np.linspace(0, n - 1, max_n).astype(int)
        idx = idx[pick]
        n = len(idx)

    x = measured_lons[idx].astype(float)
    y = measured_lats[idx].astype(float)
    z_true = true_zbs[idx].astype(float)
    z_pred = pred_zbs[idx].astype(float)

    std_true = float(np.nanstd(z_true))
    std_pred = float(np.nanstd(z_pred))

    # 1) 相关系数损失：控制整体空间趋势。
    if n >= PARAM_SPACE["spatial_min_points"] and std_true > 1e-8 and std_pred > 1e-8:
        corr = float(np.corrcoef(z_true, z_pred)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
        corr = float(np.clip(corr, -1.0, 1.0))
        corr_loss = std_true * max(0.0, 1.0 - corr)
    else:
        # 预测几乎为常数时，RMSE 可能还可以，但空间趋势是不可信的。
        corr = 0.0
        corr_loss = std_true if std_true > 1e-8 else 0.0

    # 2) 点对深度差损失：控制空间高低起伏。
    ii, jj = np.triu_indices(n, k=1)
    if len(ii) == 0:
        return out

    dx = x[ii] - x[jj]
    dy = y[ii] - y[jj]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    ok = np.isfinite(dist) & (dist > 0)

    true_diff = z_true[ii] - z_true[jj]
    pred_diff = z_pred[ii] - z_pred[jj]
    diff_err = pred_diff - true_diff

    ok &= np.isfinite(true_diff) & np.isfinite(pred_diff) & np.isfinite(diff_err)
    if not np.any(ok):
        out["spatial_corr"] = corr
        out["corr_loss"] = corr_loss
        out["spatial_loss"] = PARAM_SPACE["correlation_weight"] * corr_loss
        return out

    # 距离越近的点对越能刻画局部空间格局，因此权重大一些。
    w = 1.0 / (dist[ok] + 0.05)
    pairwise_diff_rmse = float(np.sqrt(np.sum(w * diff_err[ok] ** 2) / np.sum(w)))

    # 3) 高低关系反号比例：例如真值 A 比 B 深，但预测 A 比 B 浅，会被惩罚。
    threshold = float(PARAM_SPACE["spatial_sign_threshold_km"])
    sign_ok = np.abs(true_diff[ok]) >= threshold
    if np.any(sign_ok):
        sign_true = np.sign(true_diff[ok][sign_ok])
        sign_pred = np.sign(pred_diff[ok][sign_ok])
        sign_mismatch_rate = float(np.mean(sign_true != sign_pred))
    else:
        sign_mismatch_rate = 0.0

    spatial_loss = (
        PARAM_SPACE["pairwise_difference_weight"] * pairwise_diff_rmse
        + PARAM_SPACE["correlation_weight"] * corr_loss
        + PARAM_SPACE["sign_mismatch_weight"] * sign_mismatch_rate * max(std_true, 1.0)
    )

    out.update({
        "spatial_loss": float(spatial_loss),
        "pairwise_diff_rmse": float(pairwise_diff_rmse),
        "spatial_corr": float(corr),
        "corr_loss": float(corr_loss),
        "sign_mismatch_rate": float(sign_mismatch_rate),
        "n_spatial_points": int(n),
    })
    return out


# ============================================================
# 3. 贝叶斯优化：每个块体独立优化
# ============================================================
def suggest_params_by_wave_number(trial: optuna.Trial) -> Dict[str, float]:
    """Optuna 按真实波数范围采样参数。"""
    b2 = trial.suggest_float(
        "b2",
        PARAM_SPACE["b2_min"],
        PARAM_SPACE["b2_max"],
        step=PARAM_SPACE["b2_step"],
    )
    window_size = trial.suggest_categorical(
        "window_size",
        PARAM_SPACE["window_size_choices"],
    )

    min_bw = PARAM_SPACE["min_band_width"]

    kcmin_low, kcmin_high = PARAM_SPACE["k_cent_min_range"]
    kcmax_low, kcmax_high = PARAM_SPACE["k_cent_max_range"]
    ktmin_low, ktmin_high = PARAM_SPACE["k_top_min_range"]
    ktmax_low, ktmax_high = PARAM_SPACE["k_top_max_range"]

    k_cent_min = trial.suggest_float("k_cent_min", kcmin_low, kcmin_high)
    k_cent_max_lower = max(kcmax_low, k_cent_min + min_bw)
    if k_cent_max_lower >= kcmax_high:
        raise optuna.TrialPruned("Invalid center band width")
    k_cent_max = trial.suggest_float("k_cent_max", k_cent_max_lower, kcmax_high)

    k_top_min = trial.suggest_float("k_top_min", ktmin_low, ktmin_high)
    k_top_max_lower = max(ktmax_low, k_top_min + min_bw)
    if k_top_max_lower >= ktmax_high:
        raise optuna.TrialPruned("Invalid top band width")
    k_top_max = trial.suggest_float("k_top_max", k_top_max_lower, ktmax_high)

    if PARAM_SPACE["require_separated_bands"] and not (k_cent_max < k_top_min):
        raise optuna.TrialPruned("Center band should be lower than top band")

    return {
        "b2": float(b2),
        "window_size": float(window_size),
        "k_top_min": float(k_top_min),
        "k_top_max": float(k_top_max),
        "k_cent_min": float(k_cent_min),
        "k_cent_max": float(k_cent_max),
    }


def objective(
    trial: optuna.Trial,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    value_grid: np.ndarray,
    measured_lons: np.ndarray,
    measured_lats: np.ndarray,
    measured_zbs: np.ndarray,
    depth_feedback: Optional[Dict[str, object]] = None,
) -> float:
    """
    目标函数：等权点位 RMSE + 空间分布损失 + 有效点覆盖率惩罚。

    与只计算整个块体 RMSE 不同，这里额外加入空间分布约束：
    - pairwise_diff_rmse：预测结果是否保持真值点之间的深浅差异；
    - spatial_corr：预测结果是否与真值具有一致的空间趋势；
    - sign_mismatch_rate：预测结果是否把深浅相对关系反过来。
    """
    params = suggest_params_by_wave_number(trial)

    result = compute_curie_for_measured_points(
        params=params,
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        value_grid=value_grid,
        measured_lons=measured_lons,
        measured_lats=measured_lats,
        calculate_uncertainty=False,
    )

    pred_zbs = result["pred_zbs"]
    uncertainties = result["uncertainties"]

    # 优化有效性只由 Zb 预测和参考深度决定；dZb 留到结果阶段独立计算。
    valid = np.isfinite(pred_zbs) & np.isfinite(measured_zbs)

    if valid.sum() < PARAM_SPACE["min_valid_points"]:
        raise optuna.TrialPruned("Too few valid Zb points")

    # 所有有效参考点使用相同权重。
    rmse_zb = mean_squared_error(measured_zbs[valid], pred_zbs[valid]) ** 0.5

    if not np.isfinite(rmse_zb):
        raise optuna.TrialPruned("Invalid RMSE")

    unc_valid = uncertainties[valid & np.isfinite(uncertainties)]
    mean_uncertainty = float(np.mean(unc_valid)) if len(unc_valid) else np.nan

    valid_ratio = valid.sum() / max(len(measured_lons), 1)
    missing_penalty = PARAM_SPACE["missing_point_penalty"] * (1.0 - valid_ratio)

    spatial_metrics = calculate_spatial_distribution_metrics(
        measured_lons=measured_lons,
        measured_lats=measured_lats,
        true_zbs=measured_zbs,
        pred_zbs=pred_zbs,
        valid_mask=valid,
    )
    spatial_loss = float(spatial_metrics["spatial_loss"])

    loss = (
        rmse_zb
        + PARAM_SPACE["spatial_distribution_weight"] * spatial_loss
        + missing_penalty
    )

    # 仅人工指定方向的重算轮次启用；原参考点 RMSE 和空间约束保持不变。
    if depth_feedback is not None:
        feedback_metrics = calculate_depth_feedback(
            params, depth_feedback, lon_grid, lat_grid, value_grid
        )
        loss += feedback_metrics["feedback_missing_penalty"]
        for key, value in feedback_metrics.items():
            trial.set_user_attr(key, float(value))

    trial.set_user_attr("rmse_zb", float(rmse_zb))
    trial.set_user_attr("mean_uncertainty", mean_uncertainty)
    trial.set_user_attr("valid_points", int(valid.sum()))
    trial.set_user_attr("valid_ratio", float(valid_ratio))
    trial.set_user_attr("n_unc_valid", int(len(unc_valid)))
    trial.set_user_attr("spatial_loss", float(spatial_loss))
    trial.set_user_attr("pairwise_diff_rmse", float(spatial_metrics["pairwise_diff_rmse"]))
    trial.set_user_attr("spatial_corr", float(spatial_metrics["spatial_corr"]))
    trial.set_user_attr("sign_mismatch_rate", float(spatial_metrics["sign_mismatch_rate"]))
    trial.set_user_attr("n_spatial_points", int(spatial_metrics["n_spatial_points"]))

    return float(loss)

def optimize_for_block(
    block_name: str,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    value_grid: np.ndarray,
    measured_lons: np.ndarray,
    measured_lats: np.ndarray,
    measured_zbs: np.ndarray,
    n_trials: int = 50,
    depth_feedback: Optional[Dict[str, object]] = None,
    retry_index: int = 0,
) -> Dict[str, float]:
    """对单个块体进行贝叶斯优化。"""
    if len(measured_lons) == 0:
        print("    ⚠️ 块体中没有居里面真值点，使用默认参数")
        p = DEFAULT_BLOCK_PARAMS.copy()
        p["status"] = "no_true_points_default"
        return p

    successful_trials = 0
    failed_trials = 0

    def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal successful_trials, failed_trials
        if trial.state == optuna.trial.TrialState.COMPLETE:
            successful_trials += 1
        elif trial.state == optuna.trial.TrialState.PRUNED:
            failed_trials += 1

        done = successful_trials + failed_trials
        if done % 10 == 0:
            print(
                f"    进度: {done}/{n_trials} 次试验 "
                f"(成功: {successful_trials}, 失败: {failed_trials})"
            )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=stable_seed_from_name(block_name if retry_index == 0 else f"{block_name}:retry:{retry_index}"),
            n_startup_trials=10,
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    try:
        study.optimize(
            lambda trial: objective(
                trial,
                lon_grid,
                lat_grid,
                value_grid,
                measured_lons,
                measured_lats,
                measured_zbs,
                depth_feedback,
            ),
            n_trials=n_trials,
            callbacks=[trial_callback],
            show_progress_bar=False,
            n_jobs=1,
        )
    except Exception as e:
        print(f"    优化过程中出现异常: {e}")

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete_trials) == 0:
        print("    ⚠️ 所有试验均失败，使用默认参数")
        p = DEFAULT_BLOCK_PARAMS.copy()
        p["status"] = "optimization_failed_default"
        return p

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params.update({
        "rmse": float(best_trial.user_attrs.get("rmse_zb", best_trial.value)),
        "loss": float(best_trial.value),
        "mean_uncertainty": float(best_trial.user_attrs.get("mean_uncertainty", np.nan)),
        "valid_points": int(best_trial.user_attrs.get("valid_points", 0)),
        "valid_ratio": float(best_trial.user_attrs.get("valid_ratio", np.nan)),
        "spatial_loss": float(best_trial.user_attrs.get("spatial_loss", np.nan)),
        "pairwise_diff_rmse": float(best_trial.user_attrs.get("pairwise_diff_rmse", np.nan)),
        "spatial_corr": float(best_trial.user_attrs.get("spatial_corr", np.nan)),
        "sign_mismatch_rate": float(best_trial.user_attrs.get("sign_mismatch_rate", np.nan)),
        "n_spatial_points": int(best_trial.user_attrs.get("n_spatial_points", 0)),
        "status": "optimized",
    })
    if depth_feedback is not None:
        best_params.update({
            "depth_direction": int(depth_feedback["direction"]),
            "achieved_shift_km": float(best_trial.user_attrs["achieved_shift_km"]),
            "feedback_valid_ratio": float(best_trial.user_attrs["feedback_valid_ratio"]),
        })

    print(f"    ✅ 优化完成: {successful_trials} 次成功, {failed_trials} 次失败")
    print(f"    最佳 Zb RMSE: {best_params['rmse']:.4f} km")
    print(f"    平均 ΔZb: {best_params['mean_uncertainty']:.4f} km")
    print(f"    空间损失: {best_params['spatial_loss']:.4f}")
    print(f"    点对深度差RMSE: {best_params['pairwise_diff_rmse']:.4f} km")
    print(f"    空间相关系数: {best_params['spatial_corr']:.4f}")
    print(f"    深浅关系反号比例: {best_params['sign_mismatch_rate']:.3f}")
    print(f"    有效真值点: {best_params['valid_points']}/{len(measured_lons)}")
    print(f"    b2={best_params['b2']:.2f}, window_size={best_params['window_size']:.2f}")
    print(
        f"    Zt 波数范围: "
        f"[{best_params['k_top_min']:.5f}, {best_params['k_top_max']:.5f}] km^-1"
    )
    print(
        f"    Z0 波数范围: "
        f"[{best_params['k_cent_min']:.5f}, {best_params['k_cent_max']:.5f}] km^-1"
    )

    return best_params


# ============================================================
# 4. 全图计算与输出
# ============================================================
def polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """计算多边形顶点平均中心，用于未覆盖网格的最近块体参数继承。"""
    arr = np.asarray(polygon, dtype=float)
    return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))


def find_nearest_block_for_point(
    point: Tuple[float, float],
    block_polygons: Dict[str, List[Tuple[float, float]]],
) -> Optional[str]:
    """当网格点不在任何块体内时，使用最近块体的参数，避免出现大面积空白。"""
    if not block_polygons:
        return None
    lon, lat = point
    best_name = None
    best_dist = np.inf
    for block_name, polygon in block_polygons.items():
        clon, clat = polygon_centroid(polygon)
        dist = (lon - clon) ** 2 + (lat - clat) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = block_name
    return best_name


def build_fallback_param_candidates(base_params: Dict[str, float]) -> List[Dict[str, float]]:
    """
    当块体最优波数范围在某些网格点失败时，使用备用低/中高波数段搜索。
    原则仍然是：Z0 低波数，Zt 中高波数。
    """
    b2_candidates = []
    for v in [base_params.get("b2", 1.0), 0.0, 1.0, 2.0]:
        fv = float(v)
        if PARAM_SPACE["b2_min"] <= fv <= PARAM_SPACE["b2_max"] and fv not in b2_candidates:
            b2_candidates.append(fv)

    ws_candidates = []
    for v in [base_params.get("window_size", 3.0), 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        fv = float(v)
        if fv not in ws_candidates:
            ws_candidates.append(fv)

    # 低波数 Z0 与中高波数 Zt 的备用组合。
    band_pairs = [
        ((0.003, 0.030), (0.035, 0.085)),
        ((0.005, 0.040), (0.045, 0.090)),
        ((0.002, 0.025), (0.030, 0.075)),
        ((0.010, 0.045), (0.050, 0.090)),
        ((0.003, 0.035), (0.040, 0.088)),
    ]

    candidates: List[Dict[str, float]] = []

    # 先放入块体最优参数。
    candidates.append({
        "b2": float(base_params.get("b2", 1.0)),
        "window_size": float(base_params.get("window_size", 3.0)),
        "k_top_min": float(base_params.get("k_top_min", 0.035)),
        "k_top_max": float(base_params.get("k_top_max", 0.085)),
        "k_cent_min": float(base_params.get("k_cent_min", 0.003)),
        "k_cent_max": float(base_params.get("k_cent_max", 0.030)),
    })

    for b2 in b2_candidates:
        for ws in ws_candidates:
            for cent_band, top_band in band_pairs:
                candidates.append({
                    "b2": float(b2),
                    "window_size": float(ws),
                    "k_cent_min": float(cent_band[0]),
                    "k_cent_max": float(cent_band[1]),
                    "k_top_min": float(top_band[0]),
                    "k_top_max": float(top_band[1]),
                })

    # 去重
    unique = []
    seen = set()
    for p in candidates:
        key = tuple(round(float(p[k]), 6) for k in ["b2", "window_size", "k_cent_min", "k_cent_max", "k_top_min", "k_top_max"])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def compute_curie_with_fallback(
    lon0: float,
    lat0: float,
    base_params: Dict[str, float],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    value_grid: np.ndarray,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], str]:
    """优先使用块体最优参数；失败时启用备用波数段，选择 ΔZb 最小的可用结果。"""
    direct = compute_curie_for_single_point(lon0, lat0, base_params, lon_grid, lat_grid, value_grid)
    if direct is not None:
        return direct, base_params, "direct"

    best_res = None
    best_params = None
    best_score = np.inf
    preferred = PARAM_SPACE["preferred_uncertainty_km"]
    soft = PARAM_SPACE["soft_uncertainty_km"]

    for cand in build_fallback_param_candidates(base_params):
        res = compute_curie_for_single_point(lon0, lat0, cand, lon_grid, lat_grid, value_grid)
        if res is None:
            continue
        dz = float(res["dZb"])
        score = dz + 2.0 * max(dz - preferred, 0.0) + 5.0 * max(dz - soft, 0.0)
        if score < best_score:
            best_score = score
            best_res = res
            best_params = cand

    if best_res is not None:
        return best_res, best_params, "fallback_band"

    return None, None, "failed"


def fill_nan_nearest(lons: np.ndarray, lats: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """对剩余 NaN 做最近邻填充，确保出图不留空白。"""
    filled = np.array(grid, dtype=float, copy=True)
    if np.isfinite(filled).all():
        return filled
    valid = np.isfinite(filled)
    if valid.sum() == 0:
        return filled

    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    points_valid = np.column_stack([lon_mesh[valid], lat_mesh[valid]])
    values_valid = filled[valid]
    points_nan = np.column_stack([lon_mesh[~valid], lat_mesh[~valid]])

    try:
        from scipy.interpolate import griddata
        filled_values = griddata(points_valid, values_valid, points_nan, method="nearest")
        filled[~valid] = filled_values
    except Exception:
        # 无 scipy 时使用朴素最近邻，网格规模不大时可接受。
        for idx, (x, y) in zip(zip(*np.where(~valid)), points_nan):
            dist2 = (points_valid[:, 0] - x) ** 2 + (points_valid[:, 1] - y) ** 2
            filled[idx] = values_valid[np.argmin(dist2)]
    return filled


def compute_curie_depth_for_full_map(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    value_grid: np.ndarray,
    block_params: Dict[str, Dict[str, float]],
    block_polygons: Dict[str, List[Tuple[float, float]]],
    output_dir: str,
    output_resolution: float = 0.1,
    lon_range: Tuple[float, float] = (101.0, 117.0),
    lat_range: Tuple[float, float] = (32.0, 42.0),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    使用每个块体的最优参数计算全图。

    为了优先保证全图无空值，采用三级策略：
    1. 块体最优参数直接计算；
    2. 失败点使用备用低/中高波数段重新计算；
    3. 仍失败的点使用最近邻有效值填充。
    """
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    lons_full = np.arange(lon_min, lon_max + output_resolution / 2, output_resolution)
    lats_full = np.arange(lat_min, lat_max + output_resolution / 2, output_resolution)

    print(
        f"研究区域网格: 经度 {lons_full[0]:.1f}-{lons_full[-1]:.1f}, "
        f"纬度 {lats_full[0]:.1f}-{lats_full[-1]:.1f}"
    )
    print(f"网格点数: {len(lons_full)} x {len(lats_full)} = {len(lons_full) * len(lats_full)}")

    shape = (len(lats_full), len(lons_full))
    maps = {
        "Zb": np.full(shape, np.nan),
        "dZb": np.full(shape, np.nan),
        "Zt": np.full(shape, np.nan),
        "Z0": np.full(shape, np.nan),
        # b2/window_size 表示“块体贝叶斯优化得到的参数”，用于展示分块参数，不混入备用搜索参数。
        "b2": np.full(shape, np.nan),
        "window_size": np.full(shape, np.nan),
        # b2_used/window_size_used 表示实际计算该网格点时使用的参数，仅用于诊断备用搜索。
        "b2_used": np.full(shape, np.nan),
        "window_size_used": np.full(shape, np.nan),
        # 0=direct, 1=fallback_band, 2=nearest_block_direct, 3=nearest_block_fallback, 4=nearest_grid_fill
        "fill_method": np.full(shape, np.nan),
    }

    failed_reasons = {
        "no_block_use_nearest_block": 0,
        "no_block_params": 0,
        "direct_success": 0,
        "fallback_success": 0,
        "nearest_block_direct_success": 0,
        "nearest_block_fallback_success": 0,
        "failed_before_grid_fill": 0,
    }
    block_coverage = {name: 0 for name in block_polygons.keys()}
    valid_points = 0

    total_points = len(lats_full) * len(lons_full)
    processed_points = 0

    for i, lat0 in enumerate(lats_full):
        for j, lon0 in enumerate(lons_full):
            processed_points += 1
            if processed_points % 1000 == 0:
                print(
                    f"进度: {processed_points}/{total_points} "
                    f"({processed_points / total_points * 100:.1f}%)"
                )

            block_name = find_block_for_point((float(lon0), float(lat0)), block_polygons)
            used_nearest_block = False
            if block_name is None:
                block_name = find_nearest_block_for_point((float(lon0), float(lat0)), block_polygons)
                used_nearest_block = True
                failed_reasons["no_block_use_nearest_block"] += 1

            if block_name is None or block_name not in block_params:
                failed_reasons["no_block_params"] += 1
                continue

            if not used_nearest_block:
                block_coverage[block_name] += 1

            base_params = block_params[block_name]
            # 这里先记录块体优化参数。即使后续某个网格点启用备用频段，b2_map/window_size_map 仍表示块体级优化结果，避免出现斑点图。
            maps["b2"][i, j] = float(base_params.get("b2", np.nan))
            maps["window_size"][i, j] = float(base_params.get("window_size", np.nan))

            res, used_params, method = compute_curie_with_fallback(
                lon0=float(lon0),
                lat0=float(lat0),
                base_params=base_params,
                lon_grid=lon_grid,
                lat_grid=lat_grid,
                value_grid=value_grid,
            )

            if res is None or used_params is None:
                failed_reasons["failed_before_grid_fill"] += 1
                continue

            maps["Zb"][i, j] = res["Zb"]
            maps["dZb"][i, j] = res["dZb"]
            maps["Zt"][i, j] = res["Zt"]
            maps["Z0"][i, j] = res["Z0"]
            maps["b2_used"][i, j] = used_params["b2"]
            maps["window_size_used"][i, j] = used_params["window_size"]

            if used_nearest_block and method == "direct":
                maps["fill_method"][i, j] = 2
                failed_reasons["nearest_block_direct_success"] += 1
            elif used_nearest_block and method == "fallback_band":
                maps["fill_method"][i, j] = 3
                failed_reasons["nearest_block_fallback_success"] += 1
            elif method == "direct":
                maps["fill_method"][i, j] = 0
                failed_reasons["direct_success"] += 1
            else:
                maps["fill_method"][i, j] = 1
                failed_reasons["fallback_success"] += 1

            valid_points += 1

    print(f"直接/备用计算成功 {valid_points} 个点")
    print(f"计算阶段统计: {failed_reasons}")
    print(f"块体覆盖情况: {block_coverage}")

    # 保存填充前的 NaN 掩膜，便于后期判断哪些点来自最近邻补值。
    raw_nan_mask = ~np.isfinite(maps["Zb"])
    maps["raw_nan_mask"] = raw_nan_mask.astype(float)

    n_before_fill = int(np.sum(raw_nan_mask))
    if n_before_fill > 0:
        print(f"仍有 {n_before_fill} 个点无法直接反演，开始最近邻补值以保证全图无空值...")
        for key in ["Zb", "dZb", "Zt", "Z0", "b2", "window_size", "b2_used", "window_size_used"]:
            maps[key] = fill_nan_nearest(lons_full, lats_full, maps[key])
        maps["fill_method"][raw_nan_mask] = 4

    n_after_fill = int(np.sum(~np.isfinite(maps["Zb"])))
    print(f"最近邻补值后剩余空值: {n_after_fill}")

    return lons_full, lats_full, maps


def save_map_txt(path: str, lons: np.ndarray, lats: np.ndarray, grid: np.ndarray, value_name: str) -> None:
    """保存 lon lat value 三列数据。"""
    with open(path, "w", encoding="utf-8") as f:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                val = grid[i, j]
                if np.isfinite(val):
                    f.write(f"{lon:.3f} {lat:.3f} {val:.6f}\n")
    print(f"✅ {value_name} 已保存: {path}")


def save_block_parameters(path: str, block_params: Dict[str, Dict[str, float]]) -> None:
    """保存每个块体的最优参数。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("块体最佳参数汇总\n")
        f.write("=" * 40 + "\n")
        f.write("说明：本程序直接使用居里面真值 Zb_true(km) 优化，不再计算热流。\n")
        f.write("说明：频谱拟合区间使用真实波数范围 km^-1，不再使用整数索引 myrange。\n")
        f.write("不确定性说明：ΔZb 不参与参数优化，仅在最终全图计算阶段输出，用于独立验证。\n\n")

        for block_name, p in block_params.items():
            f.write(f"{block_name}:\n")
            f.write(f"  status: {p.get('status', 'unknown')}\n")
            f.write(f"  b2: {float(p['b2']):.3f}\n")
            f.write(f"  window_size: {float(p['window_size']):.3f} deg\n")
            f.write(f"  Zt k-range: [{float(p['k_top_min']):.6f}, {float(p['k_top_max']):.6f}] km^-1\n")
            f.write(f"  Z0 k-range: [{float(p['k_cent_min']):.6f}, {float(p['k_cent_max']):.6f}] km^-1\n")
            f.write(f"  Zb RMSE: {float(p.get('rmse', np.nan)):.6f} km\n")
            f.write(f"  loss: {float(p.get('loss', np.nan)):.6f}\n")
            f.write(f"  mean ΔZb: {float(p.get('mean_uncertainty', np.nan)):.6f} km\n")
            if "depth_direction" in p:
                direction_label = "deeper" if p["depth_direction"] > 0 else "shallower"
                f.write(f"  requested depth direction: {direction_label}\n")
                f.write(f"  achieved mean depth shift: {p['achieved_shift_km']:+.6f} km\n")
                f.write(f"  feedback grid coverage: {p['feedback_valid_ratio']:.6f}\n")
            f.write(f"  valid true points: {int(p.get('valid_points', 0))}\n\n")


def save_training_predictions(
    path: str,
    block_params: Dict[str, Dict[str, float]],
    block_polygons: Dict[str, List[Tuple[float, float]]],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    value_grid: np.ndarray,
    measured_lons: np.ndarray,
    measured_lats: np.ndarray,
    measured_zbs: np.ndarray,
) -> None:
    """保存真值点上的预测结果。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("block lon lat Zb_true Zb_pred residual dZb Zt Z0 b2 window_size k_top_min k_top_max k_cent_min k_cent_max method\n")

        for lon, lat, zb_true in zip(measured_lons, measured_lats, measured_zbs):
            block_name = find_block_for_point((float(lon), float(lat)), block_polygons)
            if block_name is None:
                block_name = find_nearest_block_for_point((float(lon), float(lat)), block_polygons)
            if block_name is None or block_name not in block_params:
                continue

            params = block_params[block_name]
            res, used_params, method = compute_curie_with_fallback(
                lon0=float(lon),
                lat0=float(lat),
                base_params=params,
                lon_grid=lon_grid,
                lat_grid=lat_grid,
                value_grid=value_grid,
            )
            if res is None or used_params is None:
                continue

            residual = res["Zb"] - zb_true
            f.write(
                f"{block_name} {lon:.3f} {lat:.3f} {zb_true:.6f} {res['Zb']:.6f} "
                f"{residual:.6f} {res['dZb']:.6f} {res['Zt']:.6f} {res['Z0']:.6f} "
                f"{float(used_params['b2']):.3f} {float(used_params['window_size']):.3f} "
                f"{float(used_params['k_top_min']):.6f} {float(used_params['k_top_max']):.6f} "
                f"{float(used_params['k_cent_min']):.6f} {float(used_params['k_cent_max']):.6f} {method}\n"
            )

    print(f"✅ 真值点预测结果已保存: {path}")

def plot_grid_map(
    lons: np.ndarray,
    lats: np.ndarray,
    grid: np.ndarray,
    output_file: str,
    title: str,
    colorbar_label: str,
) -> None:
    """绘制网格图。"""
    if np.isfinite(grid).sum() == 0:
        print(f"⚠️ {title} 没有有效值，跳过绘图")
        return

    plt.figure(figsize=(12, 8))
    contour = plt.contourf(lons, lats, grid, levels=20, cmap="jet")
    plt.colorbar(contour, label=colorbar_label)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 图件已保存: {output_file}")


def plot_uncovered_points(
    no_block_points: List[Tuple[float, float]],
    block_polygons: Dict[str, List[Tuple[float, float]]],
    output_file: str,
) -> None:
    """绘制未被块体覆盖的点。"""
    plt.figure(figsize=(12, 8))
    for idx, (block_name, polygon) in enumerate(block_polygons.items()):
        poly_lons = [p[0] for p in polygon] + [polygon[0][0]]
        poly_lats = [p[1] for p in polygon] + [polygon[0][1]]
        plt.plot(poly_lons, poly_lats, "k-", linewidth=1, label="Blocks" if idx == 0 else "")

    if no_block_points:
        uncovered_lons = [p[0] for p in no_block_points]
        uncovered_lats = [p[1] for p in no_block_points]
        plt.scatter(uncovered_lons, uncovered_lats, c="red", s=20, marker="x", label="Uncovered points")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Uncovered grid points")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def check_block_coverage(
    block_polygons: Dict[str, List[Tuple[float, float]]],
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    resolution: float = 0.1,
) -> float:
    """检查块体对研究区网格的覆盖率。"""
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    lons_check = np.arange(lon_min, lon_max + resolution / 2, resolution)
    lats_check = np.arange(lat_min, lat_max + resolution / 2, resolution)

    covered = 0
    not_covered = 0

    for lon in lons_check:
        for lat in lats_check:
            if find_block_for_point((float(lon), float(lat)), block_polygons) is not None:
                covered += 1
            else:
                not_covered += 1

    coverage_rate = covered / (covered + not_covered) * 100
    print(f"块体覆盖检查: {covered}/{covered + not_covered} 点被覆盖 ({coverage_rate:.1f}%)")
    return coverage_rate


# ============================================================
# 5. 主程序
# ============================================================
def natural_block_key(name: str) -> list:
    """按块体名称中的数字排序，保证图上编号与审核表一致。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def parse_review_selection(text: str, block_names: List[str]) -> Optional[List[str]]:
    """空输入/ok 表示完成；无效选择抛出异常，避免误结束审核。"""
    text = text.strip().lower()
    if text in {"", "ok", "保留", "全部保留", "完成"}:
        return None
    if text in {"q", "quit", "退出"}:
        raise KeyboardInterrupt("已退出审核，已有审核文件保留。")
    if text in {"all", "全部"}:
        return block_names.copy()
    name_map = {name.lower(): name for name in block_names}
    selected = []
    for token in re.split(r"[,，;；\s]+", text):
        if not token:
            continue
        if token.isdigit() and 1 <= int(token) <= len(block_names):
            name = block_names[int(token) - 1]
        elif token in name_map:
            name = name_map[token]
        else:
            raise ValueError(f"无法识别块体：{token}。请使用表中编号或完整名称。")
        if name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError("没有选中块体。")
    return selected


def ask_depth_direction(block_name: str) -> int:
    """仅选择方向：1 变深，-1 变浅，0 自由重算。"""
    while True:
        try:
            text = input(f"{block_name} 下一轮方向 [回车/0=自由重算，1/深=整体变深，2/浅=整体变浅]: ").strip().lower()
        except EOFError:
            return 0
        if text in {"", "0", "自由", "不指定"}:
            return 0
        if text in {"1", "深", "变深", "deeper", "2", "浅", "变浅", "shallower"}:
            return 1 if text in {"1", "深", "变深", "deeper"} else -1
        print("请输入 0、1 或 2。")


def build_depth_feedback(
    block_name: str,
    block_polygons: Dict[str, List[Tuple[float, float]]],
    current_params: Dict[str, float],
    lons: np.ndarray,
    lats: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    value_grid: np.ndarray,
    direction: int,
) -> Dict[str, object]:
    """用当前输出网格中属于该块体的直接谱反演点建立固定比较基准。"""
    if direction not in {-1, 1}:
        raise ValueError("方向必须是变深或变浅。")
    points = [(float(lon), float(lat)) for lat in lats for lon in lons
              if find_block_for_point((float(lon), float(lat)), block_polygons) == block_name]
    if not points:
        raise ValueError("当前输出网格中没有属于该块体的点。")
    coords = np.asarray(points)
    result = compute_curie_for_measured_points(
        current_params, lon_grid, lat_grid, value_grid, coords[:, 0], coords[:, 1],
        calculate_uncertainty=False,
    )
    valid = np.isfinite(result["pred_zbs"])
    if not np.any(valid):
        raise ValueError("该块体没有直接谱反演有效网格点，无法建立方向比较基准。")
    baseline = result["pred_zbs"][valid]
    return {"lons": coords[valid, 0], "lats": coords[valid, 1],
            "baseline_zbs": baseline, "direction": direction}


def calculate_depth_feedback(
    params: Dict[str, float],
    feedback: Dict[str, object],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    value_grid: np.ndarray,
) -> Dict[str, float]:
    """在相同网格点比较平均深度变化，避免用不同覆盖范围比较均值。"""
    result = compute_curie_for_measured_points(
        params, lon_grid, lat_grid, value_grid, feedback["lons"], feedback["lats"],
        calculate_uncertainty=False,
    )
    pred = result["pred_zbs"]
    valid = np.isfinite(pred)
    if not np.any(valid):
        raise optuna.TrialPruned("No valid grid points for depth feedback")
    shift = float(np.mean(pred[valid] - feedback["baseline_zbs"][valid]))
    ratio = float(np.mean(valid))
    # 只筛选变化方向，不指定最小变化幅度；保留原目标函数择优。
    if int(feedback["direction"]) * shift <= 0:
        raise optuna.TrialPruned("Depth change does not match the requested direction")
    missing_penalty = PARAM_SPACE["missing_point_penalty"] * (1.0 - ratio)
    return {"achieved_shift_km": shift, "feedback_missing_penalty": float(missing_penalty),
            "feedback_valid_ratio": ratio}


def plot_block_review_map(
    lons: np.ndarray, lats: np.ndarray, zb_grid: np.ndarray,
    block_polygons: Dict[str, List[Tuple[float, float]]], block_names: List[str],
    measured_lons: np.ndarray, measured_lats: np.ndarray,
    output_file: str, review_round: int,
) -> None:
    """显示并保存带有块体编号和参考点的审核预览图。"""
    fig, ax = plt.subplots(figsize=(13, 8))
    if np.any(np.isfinite(zb_grid)):
        image = ax.pcolormesh(lons, lats, zb_grid, shading="auto", cmap="jet",
                              vmin=PARAM_SPACE["zb_min"], vmax=PARAM_SPACE["zb_max"])
        fig.colorbar(image, ax=ax, label="Curie Depth (km)")
    for index, name in enumerate(block_names, 1):
        polygon = block_polygons[name]
        coords = np.asarray(polygon + [polygon[0]])
        ax.plot(coords[:, 0], coords[:, 1], color="black", linewidth=1)
        lon, lat = polygon_centroid(polygon)
        ax.text(lon, lat, f"{index}: {name}", ha="center", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.scatter(measured_lons, measured_lats, s=10, facecolors="white",
               edgecolors="black", linewidths=0.4, label="Reference points")
    ax.set(xlabel="Longitude", ylabel="Latitude", title=f"Block Review Round {review_round}",
           xlim=(lons[0], lons[-1]), ylim=(lats[0], lats[-1]))
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    print(f"审核预览图已保存：{output_file}；关闭图窗后在控制台继续。")
    plt.show()
    plt.close(fig)


def interactive_review_and_reoptimize_blocks(
    all_block_params: Dict[str, Dict[str, float]],
    block_polygons: Dict[str, List[Tuple[float, float]]],
    lon_grid: np.ndarray, lat_grid: np.ndarray, value_grid: np.ndarray,
    measured_lons: np.ndarray, measured_lats: np.ndarray, measured_zbs: np.ndarray,
    n_trials: int, output_dir: str, output_resolution: float,
    lon_range: Tuple[float, float], lat_range: Tuple[float, float],
) -> Dict[str, Dict[str, float]]:
    """预览、选择块体、指定下一轮方向、重算并再次审核。"""
    review_dir = os.path.join(output_dir, "review_maps")
    os.makedirs(review_dir, exist_ok=True)
    block_names = sorted(all_block_params, key=natural_block_key)
    attempts = {name: 0 for name in block_names}
    history = []
    review_round = 0
    while True:
        lons, lats, maps = compute_curie_depth_for_full_map(
            lon_grid, lat_grid, value_grid, all_block_params, block_polygons,
            review_dir, output_resolution, lon_range, lat_range
        )
        save_block_parameters(os.path.join(review_dir, f"round_{review_round:02d}_parameters.txt"), all_block_params)
        save_map_txt(os.path.join(review_dir, f"round_{review_round:02d}_Zb.txt"), lons, lats, maps["Zb"], "Zb_km")
        plot_block_review_map(lons, lats, maps["Zb"], block_polygons, block_names,
                              measured_lons, measured_lats,
                              os.path.join(review_dir, f"round_{review_round:02d}.png"), review_round)
        print(f"\n第 {review_round} 轮审核：只指定变深/变浅，正变化=变深，负变化=变浅。")
        print("编号  块体               重算  有效点  RMSE/km  ΔZb/km  corr    指定方向  实际变化/km")
        for index, name in enumerate(block_names, 1):
            p = all_block_params[name]
            direction_label = {1: "变深", -1: "变浅"}.get(p.get("depth_direction"), "自由")
            print(f"{index:>4}  {name:<18} {attempts[name]:>4}  {p.get('valid_points', 0):>6}  {p.get('rmse', np.nan):>7.3f}  "
                  f"{p.get('mean_uncertainty', np.nan):>7.3f}  {p.get('spatial_corr', np.nan):>6.3f}  "
                  f"{direction_label:>8}  {p.get('achieved_shift_km', np.nan):>11.3f}")
            print(f"      b2={p['b2']:.2f}, window={p['window_size']:.2f}, "
                  f"Z0=[{p['k_cent_min']:.5f}, {p['k_cent_max']:.5f}], "
                  f"Zt=[{p['k_top_min']:.5f}, {p['k_top_max']:.5f}], "
                  f"pair-RMSE={p.get('pairwise_diff_rmse', np.nan):.3f}, "
                  f"sign-mismatch={p.get('sign_mismatch_rate', np.nan):.3f}, "
                  f"status={p.get('status', 'unknown')}")
        try:
            selected = parse_review_selection(input("重算块体编号/名称（逗号分隔，all=全部，回车/ok=保留并输出，q=退出）："), block_names)
        except EOFError:
            print("无交互输入，保留当前结果并输出。")
            return all_block_params
        except ValueError as exc:
            print(exc)
            continue
        if selected is None:
            return all_block_params
        # 按用户输入顺序先收集全部方向，然后统一执行本轮重算。
        directions = {name: ask_depth_direction(name) for name in selected}
        for name in selected:
            block_lons, block_lats, block_zbs = get_points_in_block(
                block_polygons[name], measured_lons, measured_lats, measured_zbs
            )
            if not len(block_lons):
                print(f"{name} 没有参考点，保留当前参数。")
                continue
            old_params = all_block_params[name].copy()
            direction = directions[name]
            feedback = None
            if direction != 0:
                try:
                    feedback = build_depth_feedback(name, block_polygons, old_params, lons, lats,
                                                    lon_grid, lat_grid, value_grid, direction)
                except ValueError as exc:
                    print(f"{name}：{exc} 本轮保留原结果，下轮可重新设置。")
                    continue
            attempts[name] += 1
            candidate = optimize_for_block(name, lon_grid, lat_grid, value_grid,
                                           block_lons, block_lats, block_zbs,
                                           n_trials=n_trials, depth_feedback=feedback,
                                           retry_index=attempts[name])
            if candidate["status"] != "optimized":
                print(f"{name} 本轮未找到满足条件的有效谱参数，保留原结果。")
                continue
            if feedback is not None:
                achieved = candidate["achieved_shift_km"]
                direction_label = "变深" if direction > 0 else "变浅"
                print(f"{name} 指定{direction_label}，实际平均深度变化 {achieved:+.3f} km，"
                      f"比较点覆盖率 {candidate['feedback_valid_ratio']:.1%}。")
            all_block_params[name] = candidate
            history.append({"round": review_round + 1, "block": name, "attempt": attempts[name],
                            "accepted": True, "before": old_params, "candidate": candidate})
            with open(os.path.join(review_dir, "review_history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        review_round += 1


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def main(
    data_file: str = os.path.join(_REPO_ROOT, "data", "ordos_magnetic_anomaly.txt"),
    measured_zb_file: str = os.path.join(_REPO_ROOT, "data", "curie_depth_reference_points.txt"),
    block_dir: str = os.path.join(_REPO_ROOT, "blocks"),
    output_dir: str = "Curie_Depth_Block_Bayes_ThreeClassMagPrior",
    lon_range: Tuple[float, float] = (101.0, 117.0),
    lat_range: Tuple[float, float] = (32.0, 42.0),
    output_resolution: float = 0.2,
    n_trials: int = 100,
) -> None:
    # --------------------------------------------------------
    # 文件路径默认使用本仓库自带的相对路径，
    # 也可通过命令行 --config JSON 文件覆盖（见文件末尾 argparse 部分）。
    # --------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    set_global_seed(GLOBAL_SEED)
    print(f"随机种子 GLOBAL_SEED = {GLOBAL_SEED}")

    # --------------------------------------------------------
    # 数据读取
    # --------------------------------------------------------
    print("加载磁异常数据...")
    lon_grid, lat_grid, value_grid = prepare_magnetic_grid(data_file)
    print(f"磁异常网格形状: {value_grid.shape}")
    print(f"磁异常经度范围: {lon_grid.min():.3f}-{lon_grid.max():.3f}")
    print(f"磁异常纬度范围: {lat_grid.min():.3f}-{lat_grid.max():.3f}")

    print("加载居里面真值点...")
    measured_lons, measured_lats, measured_zbs = load_measured_curie_depth(measured_zb_file)

    block_polygons = load_block_polygons(block_dir)

    print(f"研究区域: {lon_range[0]}-{lon_range[1]}°E, {lat_range[0]}-{lat_range[1]}°N")
    print(f"找到 {len(block_polygons)} 个块体")
    print(f"加载了 {len(measured_zbs)} 个居里面真值点")
    print(f"居里面真值范围: {np.nanmin(measured_zbs):.2f} - {np.nanmax(measured_zbs):.2f} km")

    # 裁剪研究区真值点
    region_mask = (
        (measured_lons >= lon_range[0]) & (measured_lons <= lon_range[1]) &
        (measured_lats >= lat_range[0]) & (measured_lats <= lat_range[1])
    )
    region_lons = measured_lons[region_mask]
    region_lats = measured_lats[region_mask]
    region_zbs = measured_zbs[region_mask]

    print(f"研究区域内共有 {len(region_lons)} 个居里面真值点")

    print("检查块体覆盖情况...")
    check_block_coverage(block_polygons, lon_range, lat_range, resolution=output_resolution)

    # --------------------------------------------------------
    # 每个块体独立贝叶斯优化
    # --------------------------------------------------------
    all_block_params: Dict[str, Dict[str, float]] = {}

    for block_name, block_polygon in block_polygons.items():
        print(f"\n▶️ 优化块体 {block_name}...")
        block_lons, block_lats, block_zbs = get_points_in_block(
            block_polygon,
            region_lons,
            region_lats,
            region_zbs,
        )
        print(f"    包含 {len(block_lons)} 个居里面真值点")

        best_params = optimize_for_block(
            block_name=block_name,
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            value_grid=value_grid,
            measured_lons=block_lons,
            measured_lats=block_lats,
            measured_zbs=block_zbs,
            n_trials=n_trials,
        )
        all_block_params[block_name] = best_params

    if ENABLE_INTERACTIVE_REVIEW and all_block_params:
        all_block_params = interactive_review_and_reoptimize_blocks(
            all_block_params, block_polygons, lon_grid, lat_grid, value_grid,
            region_lons, region_lats, region_zbs, n_trials, output_dir,
            output_resolution, lon_range, lat_range
        )

    # 保存块体参数
    save_block_parameters(os.path.join(output_dir, "block_parameters.txt"), all_block_params)

    # 保存真值点预测结果
    save_training_predictions(
        path=os.path.join(output_dir, "block_training_predictions.txt"),
        block_params=all_block_params,
        block_polygons=block_polygons,
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        value_grid=value_grid,
        measured_lons=region_lons,
        measured_lats=region_lats,
        measured_zbs=region_zbs,
    )

    # --------------------------------------------------------
    # 使用分块最优参数计算全图
    # --------------------------------------------------------
    print("\n🔄 使用各块体最优参数计算全图居里面深度与不确定性...")
    lons_full, lats_full, maps = compute_curie_depth_for_full_map(
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        value_grid=value_grid,
        block_params=all_block_params,
        block_polygons=block_polygons,
        output_dir=output_dir,
        output_resolution=output_resolution,
        lon_range=lon_range,
        lat_range=lat_range,
    )

    # 保存全图数据
    save_map_txt(os.path.join(output_dir, "curie_depth_map.txt"), lons_full, lats_full, maps["Zb"], "Curie depth")
    save_map_txt(os.path.join(output_dir, "uncertainty_map.txt"), lons_full, lats_full, maps["dZb"], "Uncertainty dZb")
    save_map_txt(os.path.join(output_dir, "Zt_map.txt"), lons_full, lats_full, maps["Zt"], "Depth to the top of the magnetic layer Zt")
    save_map_txt(os.path.join(output_dir, "Z0_map.txt"), lons_full, lats_full, maps["Z0"], "Centroid depth of the magnetic layer Z0")
    save_map_txt(os.path.join(output_dir, "b2_map.txt"), lons_full, lats_full, maps["b2"], "Optimized fractal index b2 by block")
    save_map_txt(os.path.join(output_dir, "window_size_map.txt"), lons_full, lats_full, maps["window_size"], "Optimized window size by block")
    save_map_txt(os.path.join(output_dir, "b2_used_map.txt"), lons_full, lats_full, maps["b2_used"], "Fractal index b2 actually used")
    save_map_txt(os.path.join(output_dir, "window_size_used_map.txt"), lons_full, lats_full, maps["window_size_used"], "Window size actually used")
    save_map_txt(os.path.join(output_dir, "fill_method_map.txt"), lons_full, lats_full, maps["fill_method"], "Fill method")
    save_map_txt(os.path.join(output_dir, "raw_nan_mask_map.txt"), lons_full, lats_full, maps["raw_nan_mask"], "Raw missing-value mask")

    # 绘图
    plot_grid_map(
        lons_full,
        lats_full,
        maps["Zb"],
        os.path.join(output_dir, "curie_depth_map.png"),
        "Curie Depth Map",
        "Curie Depth (km)",
    )
    plot_grid_map(
        lons_full,
        lats_full,
        maps["dZb"],
        os.path.join(output_dir, "uncertainty_map.png"),
        "Curie Depth Uncertainty Map",
        "ΔZb (km)",
    )
    plot_grid_map(
        lons_full,
        lats_full,
        maps["b2"],
        os.path.join(output_dir, "b2_map.png"),
        "Optimized Fractal Index by Block",
        "b2",
    )
    plot_grid_map(
        lons_full,
        lats_full,
        maps["window_size"],
        os.path.join(output_dir, "window_size_map.png"),
        "Optimized Window Size by Block",
        "Window size (degree)",
    )
    plot_grid_map(
        lons_full,
        lats_full,
        maps["fill_method"],
        os.path.join(output_dir, "fill_method_map.png"),
        "Fill Method Map",
        "0 direct, 1 fallback, 2/3 nearest block, 4 nearest grid",
    )

    print("\n🎉 所有处理完成！")
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Block-wise Bayesian (TPE) inversion of the Curie isothermal "
                    "surface constrained by heat-flow-derived reference depths."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional JSON file that overrides the default settings. Supported keys: "
             "data_file, measured_zb_file, block_dir, output_dir, lon_range, lat_range, "
             "output_resolution, n_trials.",
    )
    cli = parser.parse_args()

    if cli.config:
        with open(cli.config, "r", encoding="utf-8") as fh:
            _overrides = json.load(fh)
        _base = main
        _defaults = {
            "data_file": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "data", "ordos_magnetic_anomaly.txt"),
            "measured_zb_file": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "data", "curie_depth_reference_points.txt"),
            "block_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocks"),
            "output_dir": "Curie_Depth_Block_Bayes_ThreeClassMagPrior",
        }
        _defaults.update(_overrides)
        _defaults["lon_range"] = tuple(_defaults.get("lon_range", (101.0, 117.0)))
        _defaults["lat_range"] = tuple(_defaults.get("lat_range", (32.0, 42.0)))
        main(**_defaults)
    else:
        main()
