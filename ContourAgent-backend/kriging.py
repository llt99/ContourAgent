import numpy as np
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from skgstat import Variogram
import traceback
from typing import Dict, Any, List
from scipy import stats


class Interpolator:
    """插值计算工具（自动选择方法 → 检验正态性 → 选择最优变异函数 → 插值并返回网格）"""
    def __init__(self):
        self.basin_union = None

    def set_basin_boundary(self, basin_geom):
        self.basin_union = basin_geom

    # ================== 数据特征分析 ==================
    def suggest_kriging_method(self, lons, lats, values, alpha=0.05):
        """
        自动建议使用 Ordinary 或 Universal Kriging
        逻辑：
          - 存在显著空间趋势（value 与 lon 或 lat 线性相关且显著） -> UK
          - 数据非正态或变异系数很大 -> UK
          - 否则 -> OK
        返回 dict: {"suggestion": "ok"/"uk", "reason": "...", "metrics": {...}}
        """
        mask = np.isfinite(values)
        lons = np.array(lons)[mask]
        lats = np.array(lats)[mask]
        values = np.array(values)[mask]
        if len(values) < 10:
            return {"suggestion": "ok", "reason": "样本点太少 (<10)，避免 UK 过拟合", "metrics": {}}

        # 相关性（pearson）
        try:
            corr_x, p_x = stats.pearsonr(values, lons)
        except Exception:
            corr_x, p_x = 0.0, 1.0
        try:
            corr_y, p_y = stats.pearsonr(values, lats)
        except Exception:
            corr_y, p_y = 0.0, 1.0

        trend_strength = max(abs(corr_x), abs(corr_y))

        # 正态性 & 偏度 & 变异系数
        try:
            shapiro_stat, shapiro_p = stats.shapiro(values)
        except Exception:
            shapiro_stat, shapiro_p = None, 1.0
        skewness = float(stats.skew(values))
        cv = float(np.std(values) / (np.mean(values) + 1e-9))

        # 决策规则（可调整阈值）
        if trend_strength > 0.3 and (p_x < alpha or p_y < alpha):
            reason = f"存在显著空间趋势 (corr_x={corr_x:.3f}, p_x={p_x:.3f}; corr_y={corr_y:.3f}, p_y={p_y:.3f})"
            return {"suggestion": "uk", "reason": reason,
                    "metrics": {"corr_x": corr_x, "p_x": p_x, "corr_y": corr_y, "p_y": p_y,
                                "shapiro_p": shapiro_p, "skewness": skewness, "cv": cv}}
        if shapiro_p is not None and shapiro_p < alpha and abs(skewness) > 1:
            reason = f"分布偏态且非正态 (shapiro_p={shapiro_p:.3f}, skewness={skewness:.3f})"
            return {"suggestion": "uk", "reason": reason,
                    "metrics": {"shapiro_p": shapiro_p, "skewness": skewness, "cv": cv}}
        if cv > 1.0:
            reason = f"变异系数较大 (CV={cv:.3f})，方差可能随位置变化"
            return {"suggestion": "uk", "reason": reason, "metrics": {"cv": cv}}

        return {"suggestion": "ok", "reason": "数据平稳且无明显趋势", "metrics": {"shapiro_p": shapiro_p, "skewness": skewness, "cv": cv}}

    # ================== 工具函数 ==================
    def check_normality_and_transform(self, values, alpha=0.05):
        """检查正态性，如果不符合进行 Box-Cox 变换
        返回: transformed_values, lmbda, was_transformed, shift, shapiro_p
        """
        values = np.array(values, dtype=float)
        # shift 以保证正数用于 boxcox
        min_val = np.min(values)
        shift = 0.0
        if min_val <= 0:
            shift = abs(min_val) + 1e-6
            values_shifted = values + shift
        else:
            values_shifted = values.copy()

        # Shapiro-Wilk 检验（try 防止异常）
        try:
            _, p = stats.shapiro(values_shifted)
        except Exception:
            p = 1.0

        if p < alpha:
            # 尝试 Box-Cox（若全为正）
            try:
                transformed_values, lmbda = stats.boxcox(values_shifted)
                return transformed_values, lmbda, True, shift, p
            except Exception:
                # 退化到 log1p 当所有值> -1
                try:
                    if np.all(values_shifted > 0):
                        transformed_values = np.log1p(values_shifted)
                        return transformed_values, None, True, shift, p
                except Exception:
                    pass
                # 如果不能变换则返回原始
                return values, None, False, 0.0, p
        else:
            return values, None, False, 0.0, p

    def inverse_boxcox(self, y, lmbda):
        """Box-Cox 反变换"""
        if lmbda is None:
            return y
        y = np.array(y)
        if lmbda == 0:
            return np.exp(y)
        safe_y = y * lmbda + 1
        safe_y = np.where(safe_y <= 0, np.nan, safe_y)
        return np.power(safe_y, 1.0 / lmbda)

    def standardize(self, values):
        """标准化（返回 standardized, mean, std）"""
        mean = float(np.mean(values))
        std = float(np.std(values))
        std = std if std > 1e-10 else 1.0
        return (values - mean) / std, mean, std

    def destandardize(self, z, mean, std):
        """反标准化"""
        return z * std + mean

    def generate_grid(self, lons, lats, target_res=0.03, extent=None,
                      margin=0.5, margin_ratio=None, force_extent=None,
                      clip_file="scBasin.geojson"):
        """
        生成插值网格：
          1) 优先使用 force_extent（若显式给定）
          2) 其次使用 self.basin_union（通过 set_basin_boundary 传入）
          3) 再次使用 clip_file=scBasin.geojson 读取盆地边界的 bounds
          4) 若以上都失败，则退回到井点范围 / extent
        """
        import numpy as np

        # 1. 保证 target_res 合法
        if target_res is None or target_res <= 0:
            target_res = 0.03

        # 2. 若未显式指定 force_extent，则尝试用盆地边界
        if force_extent is None:
            # 2.1 优先用 Interpolator 内部保存的盆地几何（如果有人调用过 set_basin_boundary）
            if self.basin_union is not None:
                try:
                    minx, miny, maxx, maxy = self.basin_union.bounds
                    force_extent = (minx, maxx, miny, maxy)
                except Exception:
                    force_extent = None

            # 2.2 其次尝试用 clip_file（默认 scBasin.geojson）
            if force_extent is None and clip_file is not None:
                try:
                    import fiona
                    from shapely.geometry import shape
                    from shapely.ops import unary_union

                    with fiona.open(clip_file, "r", encoding="utf-8") as src:
                        geoms = [shape(feat["geometry"]) for feat in src]

                    if geoms:
                        basin = unary_union(geoms)
                        minx, miny, maxx, maxy = basin.bounds
                        force_extent = (minx, maxx, miny, maxy)
                except Exception as e:
                    print(f"[WARN] 读取 clip_file='{clip_file}' 失败，将退回井点范围: {e}")
                    force_extent = None

        # 3. 确定最终使用的空间范围
        if force_extent is not None:
            # 使用盆地 bounds（或显式给定的 force_extent）
            minx, maxx, miny, maxy = force_extent
            # 使用盆地范围时，一般不再额外加 margin
        else:
            # 没有盆地信息 → 用 extent 或井点范围 + margin
            if extent is None:
                minx, maxx = float(np.min(lons)), float(np.max(lons))
                miny, maxy = float(np.min(lats)), float(np.max(lats))
            else:
                minx, maxx, miny, maxy = extent

            # 只有在没有 force_extent 时才加 margin
            if margin_ratio is not None:
                dx = (maxx - minx) * margin_ratio
                dy = (maxy - miny) * margin_ratio
            else:
                dx = dy = margin
            minx -= dx;
            maxx += dx
            miny -= dy;
            maxy += dy

        # 4. 按 target_res 生成规则网格
        nx = max(2, int((maxx - minx) / target_res) + 1)
        ny = max(2, int((maxy - miny) / target_res) + 1)
        grid_lon = np.linspace(minx, maxx, nx)
        grid_lat = np.linspace(miny, maxy, ny)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

        return (grid_x, grid_y), grid_lon, grid_lat

    # ================== 半变异函数优化 ==================
    def select_best_model(self, lons, lats, values, candidate_models=None, *args, **kwargs):
        """
        自动选择最优半变异函数模型（支持传入多余参数，不会报错）
        说明：
          - scikit-gstat 并不内置 'linear'，遇到 'linear' 会跳过拟合（标记为 skipped）
          - 如果所有 skgstat 模型拟合失败且 'linear' 在候选中，则回退使用 'linear'
        """
        if candidate_models is None:
            candidate_models = ['spherical', 'exponential', 'gaussian']

        coords = np.column_stack((lons, lats))
        values = np.array(values, dtype=float)

        all_results = []
        best_model = None
        best_rmse = float("inf")

        print(f"[INFO] 共 {len(candidate_models)} 个候选半变异模型待评估...")

        # 记录是否包含 linear（可能用于回退）
        contains_linear = any(m.lower() == "linear" for m in candidate_models)

        for model_name in candidate_models:
            mn = model_name.lower()
            if mn == "linear":
                # scikit-gstat 默认不支持 linear variogram — 跳过拟合，但记录
                all_results.append({
                    "model": "linear",
                    "rmse": None,
                    "params": None,
                    "note": "skipped_by_skgstat (use PyKrige linear as fallback)"
                })
                print("[SKIP] scikit-gstat 不支持 'linear' 半变异模型，已跳过（将由 PyKrige 支持）。")
                continue

            try:
                V = Variogram(coords, values, model=mn, normalize=False)
                V.fit(method='trf')

                exp_lags = getattr(V, "bins", np.array([]))
                exp_gamma = getattr(V, "experimental", np.array([]))
                fit_gamma = V.fitted_model(exp_lags) if len(exp_lags) > 0 else np.array([])

                rmse = float(np.sqrt(np.mean((exp_gamma - fit_gamma) ** 2))) if len(exp_gamma) > 0 else float("inf")

                params = getattr(V, "parameters", None)
                if params is None or len(params) < 3:
                    nugget, sill, rng = 0.0, float(np.var(values)), float((np.max(lons) - np.min(lons)) / 3.0)
                else:
                    nugget, sill, rng = float(params[0]), float(params[1]), float(params[2])

                all_results.append({
                    "model": mn,
                    "rmse": rmse,
                    "params": {"nugget": nugget, "sill": sill, "range": rng},
                })

                print(f"[MODEL] {mn:<12s} → RMSE={rmse:.6f}, nugget={nugget:.3f}, sill={sill:.3f}, range={rng:.3f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = mn

            except Exception as e:
                all_results.append({
                    "model": mn,
                    "error": str(e),
                    "rmse": None,
                    "params": None
                })
                print(f"[WARN] 模型 {mn} 拟合失败: {e}")
                continue

        # 回退逻辑：若没有找到任何 skgstat 模型，但候选里含 linear，则使用 linear（由 PyKrige 支持）
        if best_model is None and contains_linear:
            best_model = "linear"
            print("[WARN] 所有 scikit-gstat 候选拟合失败，回退至 'linear'（由 PyKrige 插值支持）")

        if best_model is None:
            print("[ERROR] 所有候选半变异模型拟合均失败（包括无可用回退）。")
        else:
            print(f"[INFO] ✅ 最优半变异模型: {best_model} (RMSE={best_rmse:.6f} if computed)")

        return best_model, all_results

    # ================== 主插值函数（按需求流程） ==================
    def interpolate(self, lons, lats, values, params: Dict[str, Any], target_res=0.03, extent=None):
        """
        完整流程：
          1) 自动选择方法（OK/UK）——除非 params 中已有 method
          2) 检验正态性并在需要时 Box-Cox 变换（记录变换信息）
          3) 标准化（用于变异函数拟合）
          4) 在候选变异模型中选择最优模型（skgstat）
          5) 使用 PyKrige（OK 或 UK）按选定变异模型进行插值，生成网格并返回
        返回包含：grid_x, grid_y, z, model_params, boxcox_info, variogram_candidates
        """
        try:
            # Step 0: 参数处理 & 验证输入
            lons = np.array(lons, dtype=float)
            lats = np.array(lats, dtype=float)
            values = np.array(values, dtype=float)
            mask = np.isfinite(values)
            lons, lats, values = lons[mask], lats[mask], values[mask]
            if len(values) == 0:
                return {"error": "❌ 无有效数据点，无法插值"}

            # Step 1: 自动选择方法（若 params 未指定）
            method_param = params.get("method")
            if not method_param or str(method_param).strip().lower() in ["none", "", "auto"]:
                # 自动判断使用 OK 或 UK
                method_info = self.suggest_kriging_method(lons, lats, values)
                kriging_method = method_info.get("suggestion", "ok").lower()
                method_reason = method_info.get("reason", "自动选择（默认 ok）")
            else:
                kriging_method = str(method_param).lower()
                method_reason = "用户指定"

            # Step 2: 检验正态性并变换（如果需要）
            values_transformed, lmbda, was_transformed, shift, shapiro_p = self.check_normality_and_transform(values)
            # Step 3: 标准化（用于变异函数拟合）
            values_std, mean_val, std_val = self.standardize(values_transformed)

            # Step 4: 选择最优半变异模型（在标准化后的数据上）
            variogram_param = params.get("variogram_model")
            if not variogram_param or str(variogram_param).strip().lower() in ["none", "", "auto"]:
                candidate_models = params.get("candidate_variograms",
                                              ['spherical', 'exponential', 'gaussian'])
                best_model, variogram_candidates = self.select_best_model(
                    lons, lats, values_std, candidate_models=candidate_models
                )
                if not best_model:
                    best_model = "spherical"
            else:
                best_model = str(variogram_param).lower()
                variogram_candidates = [{"model": best_model, "note": "用户指定"}]

            # Step 5: 生成网格（注意：使用原始 lon/lat 范围）
            (grid_x, grid_y), grid_lon, grid_lat = self.generate_grid(
                lons, lats,
                target_res=target_res,
                extent=None,  # 不允许外部传进来的 extent 生效
                clip_file="scBasin.geojson"
            )

            # Step 6: Kriging 插值（在标准化值空间执行）
            variogram_model_to_use = best_model
            variogram_parameters = None  # 让 PyKrige 自行拟合同名模型（更稳健）

            # 确定 drift_terms（仅在 UK 有效）
            drift_type = params.get("drift", "linear")
            drift_terms = None
            if kriging_method == "uk":
                if drift_type == "linear":
                    drift_terms = ["regional_linear"]
                elif drift_type == "quadratic":
                    drift_terms = ["regional_quadratic"]

            # PyKrige 支持 'linear' variogram（但 scikit-gstat 不一定支持）
            if kriging_method == "uk":
                uk = UniversalKriging(
                    lons, lats, values_std,
                    variogram_model=variogram_model_to_use,
                    variogram_parameters=variogram_parameters,
                    drift_terms=drift_terms,
                    verbose=False, enable_plotting=False
                )
                z_std, ss = uk.execute("grid", grid_lon, grid_lat, backend="vectorized")
                pykrige_fit = {
                    "model": uk.variogram_model,
                    "variogram_model_parameters": uk.variogram_model_parameters
                }
            else:
                ok = OrdinaryKriging(
                    lons, lats, values_std,
                    variogram_model=variogram_model_to_use,
                    variogram_parameters=variogram_parameters,
                    verbose=False, enable_plotting=False
                )
                z_std, ss = ok.execute("grid", grid_lon, grid_lat, backend="vectorized")
                pykrige_fit = {
                    "model": ok.variogram_model,
                    "variogram_model_parameters": ok.variogram_model_parameters
                }

            # Step 7: 反标准化与反变换（返回原始量纲）
            z_destd = self.destandardize(z_std, mean_val, std_val)
            if was_transformed:
                # 先做 Box-Cox 反变换（或 log 的反变换）
                if lmbda is not None:
                    z_back = self.inverse_boxcox(z_destd, lmbda)
                else:
                    # 如果使用 log1p 做的变换： inverse is expm1
                    z_back = np.expm1(z_destd)
                if shift != 0:
                    z_back = z_back - shift
            else:
                z_back = z_destd

            # Convert to python lists for JSON-serializable output
            return {
                "grid_x": grid_x.tolist(),
                "grid_y": grid_y.tolist(),
                "z": np.array(z_back).tolist(),
                "model_params": {
                    "method": kriging_method,
                    "method_reason": method_reason,
                    "selected_variogram": variogram_model_to_use,
                    "pykrige_fit": pykrige_fit,
                    "pretransform_stats": {"shapiro_p": shapiro_p, "was_transformed": bool(was_transformed)}
                },
                "boxcox_info": {"was_transformed": bool(was_transformed), "lambda": lmbda, "shift": shift},
                "variogram_candidates": variogram_candidates
            }

        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

    # ================== 交叉验证（保留） ==================
    def cross_validate(self, lons, lats, values, params):
        coords = np.column_stack((lons, lats))
        values = np.array(values, dtype=float)
        mask = np.isfinite(values)
        coords, values = coords[mask], values[mask]
        n_points = len(values)
        if n_points < 3:
            return {"error": "样本点太少"}

        residuals, variances = [], []

        for i in range(n_points):
            coords_train = np.delete(coords, i, axis=0)
            values_train = np.delete(values, i)
            coord_test = coords[i]
            value_test = values[i]

            try:
                res = self.interpolate(
                    lons=coords_train[:, 0],
                    lats=coords_train[:, 1],
                    values=values_train,
                    params=params,
                    target_res=0.03,
                    extent=(coord_test[0] - 0.03, coord_test[0] + 0.03,
                            coord_test[1] - 0.03, coord_test[1] + 0.03)
                )
                if "error" in res:
                    continue

                z_grid = np.array(res["z"])
                pred = z_grid[0][0]
                var_pred = np.var(values_train)
                residuals.append(pred - value_test)
                variances.append(var_pred)
            except Exception:
                continue

        if len(residuals) == 0:
            return {"error": "交叉验证未成功计算任何残差"}

        residuals = np.array(residuals)
        variances = np.array(variances)
        variances_safe = np.where(variances <= 0, 1e-6, variances)
        KRME = float(np.mean(residuals))
        KRMSE = float(np.mean(residuals ** 2 / variances_safe))

        method = params.get("method", "ok").upper()
        variogram_model = params.get("variogram_model", "spherical").lower()
        print(f"[INFO] 模型={method}-{variogram_model} -> KRME={KRME:.4f}, KRMSE={KRMSE:.4f}")

        return {"KRME": KRME, "KRMSE": KRMSE, "method": method, "variogram_model": variogram_model}