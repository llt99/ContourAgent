from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from matplotlib_scalebar.scalebar import ScaleBar
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging
from shapely.geometry import shape as shapely_shape, Polygon as ShapelyPolygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from geojson import FeatureCollection, Feature, Polygon as GeoJSONPolygon
from scipy.ndimage import gaussian_filter
import numpy as np
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import palettable.colorbrewer.diverging
import json
import os
import traceback
from matplotlib.path import Path
from shapely.geometry import shape
import jenkspy
from skimage import measure
import fiona
from shapely.geometry import mapping
from fiona.crs import from_epsg
import zipfile
from skgstat import Variogram
from pyproj import Transformer
from shapely.ops import transform
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def load_basin_union(path):
    if not os.path.exists(path):
        print(f"⚠️ 未找到 {path}，跳过裁剪")
        return None
    with open(path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    geoms = [shapely_shape(feat["geometry"]) for feat in geojson_data["features"]]
    return unary_union(geoms)

basin_union = load_basin_union("scBasin.geojson")

def extract_polygons(geom):
    if geom.is_empty:
        return []
    elif geom.geom_type == 'Polygon':
        return [geom]
    elif geom.geom_type == 'MultiPolygon':
        return list(geom.geoms)
    elif geom.geom_type == 'GeometryCollection':
        return [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
    return []

# 裁剪
def mask_outside_boundary(grid_x, grid_y, z, boundary_geom):
    if boundary_geom is None:
        return z

    paths = []
    for poly in extract_polygons(boundary_geom):
        coords = np.array(poly.exterior.coords)
        if len(coords) >= 3:
            paths.append(Path(coords))

    points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    mask = np.zeros(len(points), dtype=bool)
    for path in paths:
        mask |= path.contains_points(points)

    mask = mask.reshape(z.shape)
    z[~mask] = np.nan
    return z

def save_features_to_shapefile(features, shp_path):
    schema = {
        'geometry': 'Polygon',
        'properties': {
            'value': 'float',
            'min_value': 'float',  # ✅ 改成和 props 一致
            'max_value': 'float'
        }
    }

    with fiona.open(shp_path, 'w',
                    driver='ESRI Shapefile',
                    crs=from_epsg(4326),
                    schema=schema,
                    encoding='utf-8') as sink:
        for feat in features:
            geom = shape(feat.geometry)
            props = {
                'value': feat.properties['value'],
                'min_value': feat.properties['min_value'],
                'max_value': feat.properties['max_value']
            }
            sink.write({
                'geometry': mapping(geom),
                'properties': props
            })

def zip_shapefile(shp_path):
    base = os.path.splitext(shp_path)[0]
    exts = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
    files = []
    for ext in exts:
        f = base + ext
        if os.path.exists(f) and os.path.getsize(f) > 0:
            files.append(f)
        else:
            print(f"⚠️ 文件缺失或为空: {f}")

    if not files:
        raise RuntimeError("没有有效的 shapefile 文件可压缩")

    zip_path = base + '.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for f in files:
            print(f"添加到zip: {f}")
            zipf.write(f, arcname=os.path.basename(f))
    print(f"✅ 压缩文件生成: {zip_path}")
    return zip_path

# 交叉验证
def cross_validation_uk(x, y, values, variogram_model, drift_terms):
    n = len(values)
    residuals = []

    for i in range(n):
        x_train = np.delete(x, i)
        y_train = np.delete(y, i)
        val_train = np.delete(values, i)

        x_val = x[i]
        y_val = y[i]
        true_val = values[i]

        uk = UniversalKriging(
            x_train, y_train, val_train,
            variogram_model=variogram_model,
            drift_terms=drift_terms,
            verbose=False,
            enable_plotting=False
        )
        pred, ss = uk.execute("points", np.array([x_val]), np.array([y_val]))
        pred_val = pred[0]
        residuals.append(true_val - pred_val)

    residuals = np.array(residuals)
    RSS = np.sum(residuals ** 2)
    TSS = np.sum((values - np.mean(values)) ** 2)
    R2 = 1 - RSS / TSS if TSS != 0 else float('nan')
    return RSS, R2

# 半变异函数图像
def generate_variogram_plot(V, kriging_model, title_suffix=""):
    import io
    import matplotlib.pyplot as plt
    import base64
    import numpy as np
    from matplotlib import rcParams

    rcParams['font.sans-serif'] = ['SimSun']  # 中文字体
    rcParams['axes.unicode_minus'] = False  # 负号正常显示

    exp_lags = V.bins
    exp_gamma = V.experimental
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

    ax.scatter(exp_lags, exp_gamma, color='blue', label='实验点')

    x_model = np.linspace(min(exp_lags), max(exp_lags), 100)
    y_model = V.fitted_model(x_model)
    ax.plot(x_model, y_model, color='red', label=f'拟合模型: {kriging_model}')

    ax.set_title(f'半变异函数图 {title_suffix}')
    ax.set_xlabel('距离 (m)')
    ax.set_ylabel('半变异值 γ(h)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def spherical_model(h, nugget, partial_sill, range_):
    h = np.asarray(h)
    y = np.piecewise(
        h,
        [h <= range_, h > range_],
        [
            lambda x: nugget + partial_sill * (1.5 * (x / range_) - 0.5 * (x / range_) ** 3),
            lambda x: nugget + partial_sill,
        ]
    )
    return y

def exponential_model(h, nugget, partial_sill, range_):
    h = np.asarray(h)
    return nugget + partial_sill * (1 - np.exp(-h / range_))

def gaussian_model(h, nugget, partial_sill, range_):
    h = np.asarray(h)
    return nugget + partial_sill * (1 - np.exp(-(h / range_) ** 2))

def linear_model(h, nugget, slope, _=None):
    h = np.asarray(h)
    return nugget + slope * h

def get_model_function(model):
    return {
        'spherical': spherical_model,
        'exponential': exponential_model,
        'gaussian': gaussian_model,
        'linear': linear_model,
    }.get(model)

def optimize_single_model(model_name, coords, values):
    V = Variogram(coords, values, model=model_name, normalize=False)
    print("拟合前参数：", V.parameters)

    V.fit(method='trf')
    print("拟合后参数：", V.parameters)

    sill, range_, nugget = V.parameters
    partial_sill = sill - nugget
    nugget_ratio = nugget / sill if sill != 0 else 0
    exp_lags = V.bins
    exp_gamma = V.experimental
    fit_gamma = V.fitted_model(exp_lags)

    rss = np.sum((exp_gamma - fit_gamma) ** 2)
    r2 = 1 - rss / np.sum((exp_gamma - np.mean(exp_gamma)) ** 2)

    return {
        "model": model_name,
        "nugget": nugget,
        "sill": sill,
        "partial_sill": partial_sill,
        "nugget_ratio": nugget_ratio,
        "range": range_,
        "rss": rss,
        "r2": r2,
        "exp_lags": exp_lags,
        "exp_gamma": exp_gamma,
        "fit_gamma": fit_gamma
    }

# 生成半变异函数图像
def generate_variogram_plot_from_data(h, gamma, fit_vals, model, title_suffix=""):
    import io
    import matplotlib.pyplot as plt
    import base64

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.scatter(h, gamma, color='blue', label='实验点')
    ax.plot(h, fit_vals, color='red', label=f'拟合模型: {model}')
    ax.set_title(f'半变异函数图 {title_suffix}')
    ax.set_xlabel('距离 (m)')
    ax.set_ylabel('半变异值 γ(h)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    img_buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(img_buf.read()).decode('utf-8')}"

# 矩阵分解求逆矩阵，然后利用逆矩阵对角元素快速算留一交叉验证
def compute_covariance_matrix_ok(x, y, sill, nugget, range_, variogram_model):
    coords = np.column_stack((x, y))
    n = len(coords)

    # ===== 半变异函数模型定义 =====
    def spherical_model(h, nugget, psill, range_):
        h = np.minimum(h, range_)
        return nugget + psill * (1.5 * (h / range_) - 0.5 * (h / range_) ** 3)

    def exponential_model(h, nugget, psill, range_):
        return nugget + psill * (1 - np.exp(-h / range_))

    def gaussian_model(h, nugget, psill, range_):
        return nugget + psill * (1 - np.exp(-(h / range_)**2))

    def linear_model(h, nugget, slope, _=None):
        return nugget + slope * h

    model_funcs = {
        "spherical": spherical_model,
        "exponential": exponential_model,
        "gaussian": gaussian_model,
        "linear": linear_model
    }

    model_func = model_funcs.get(variogram_model)
    if model_func is None:
        raise ValueError(f"不支持的半变异函数模型: {variogram_model}")

    # ===== 计算距离矩阵（矢量化） =====
    dx = coords[:, np.newaxis, 0] - coords[np.newaxis, :, 0]
    dy = coords[:, np.newaxis, 1] - coords[np.newaxis, :, 1]
    h = np.sqrt(dx**2 + dy**2)  # 距离矩阵 h(i,j)

    partial_sill = sill - nugget
    gamma = model_func(h, nugget, partial_sill, range_)
    K = sill - gamma

    # 数值稳定：加入微小正则项（防止病态矩阵）
    K += np.eye(n) * 1e-10

    return K

# 利用协方差矩阵逆矩阵计算OK留一交叉验证残差，返回RSS和R2
def loo_cross_validation_ok(x, y, values, sill, nugget, range_, variogram_model,
                            auto_optimize_nugget=False,
                            nugget_ratios=np.linspace(0.01, 0.5, 30),
                            target_krmse=1.0,
                            tol=0.01):
    """
    如果 auto_optimize_nugget=True，则自动调整 nugget（作为 sill 的比例）
    使得 KRMSE 趋近 target_krmse，否则直接用传入的 nugget计算。

    返回：RSS, R2, KRME, KRMSE, 以及最佳 nugget（当优化时）
    """
    if not auto_optimize_nugget:
        # 直接计算，返回固定 nugget
        K = compute_covariance_matrix_ok(x, y, sill, nugget, range_, variogram_model)
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            raise RuntimeError("协方差矩阵不可逆，无法做交叉验证")

        diag = np.diag(K_inv)
        if np.any(np.abs(diag) < 1e-10):
            raise RuntimeError(f"K⁻¹ 对角元素过小，最小值: {np.min(np.abs(diag)):.2e}")

        K_inv_z = K_inv @ values
        residuals = K_inv_z / diag
        preds = values - residuals

        RSS = np.sum(residuals ** 2)
        TSS = np.sum((values - np.mean(values)) ** 2)
        R2 = 1 - RSS / TSS if TSS != 0 else float('nan')

        std_y = np.std(values)
        krme = np.mean(preds - values)
        krmse = np.mean(((preds - values) / std_y) ** 2)

        # mean_y = np.mean(values)
        # krme = np.mean(preds - values) / mean_y
        # krmse = np.sqrt(np.mean((preds - values) ** 2)) / mean_y

        return RSS, R2, krme, krmse, nugget

    else:
        best_nugget = None
        best_krmse = None
        best_rss = None
        best_r2 = None
        best_krme = None
        best_diff = float("inf")

        for ratio in nugget_ratios:
            nugget_try = ratio * sill
            try:
                K = compute_covariance_matrix_ok(x, y, sill, nugget_try, range_, variogram_model)
                print(f"尝试 nugget_ratio={ratio:.3f}, nugget={nugget_try:.4f}, "
                      f"K matrix stats: min={K.min():.4f}, max={K.max():.4f}, mean={K.mean():.4f}")

                K_inv = np.linalg.inv(K)
                diag = np.diag(K_inv)
                if np.any(np.abs(diag) < 1e-10):
                    continue
                K_inv_z = K_inv @ values
                residuals = K_inv_z / diag
                preds = values - residuals
                RSS = np.sum(residuals ** 2)
                TSS = np.sum((values - np.mean(values)) ** 2)
                R2 = 1 - RSS / TSS if TSS != 0 else float('nan')
                # mean_y = np.mean(values)
                # krme = np.mean(preds - values) / mean_y
                # krmse = np.sqrt(np.mean((preds - values) ** 2)) / mean_y

                std_y = np.std(values)
                krme = np.mean(preds - values)
                krmse = np.mean(((preds - values) / std_y) ** 2)


            except Exception:
                continue

            diff = abs(krmse - target_krmse)
            print(f"尝试 nugget_ratio={ratio:.3f}, nugget={nugget_try:.4f}, KRMSE={krmse:.4f}, 差值={diff:.4f}")

            if diff < best_diff:
                best_diff = diff
                best_nugget = nugget_try
                best_krmse = krmse
                best_rss = RSS
                best_r2 = R2
                best_krme = krme

            if diff <= tol:
                print("达到容忍度，停止搜索")
                break

        if best_nugget is None:
            raise RuntimeError("自动优化未找到合适的 nugget")

        return best_rss, best_r2, best_krme, best_krmse, best_nugget

# K-fold 交叉验证
def kfold_uk_cross_validation(lons, lats, values, model, drift_terms, n_splits=5):
    from sklearn.model_selection import KFold
    y_true, y_pred = [], []

    coords = np.column_stack((lons, lats))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(coords):
        train_coords = coords[train_idx]
        train_values = values[train_idx]
        test_coords = coords[test_idx]
        test_values = values[test_idx]

        # 拆分
        lon_train, lat_train = train_coords[:, 0], train_coords[:, 1]
        lon_test, lat_test = test_coords[:, 0], test_coords[:, 1]

        # 模型
        uk = UniversalKriging(
            lon_train, lat_train, train_values,
            variogram_model=model,
            drift_terms=drift_terms,
            verbose=False,
            enable_plotting=False
        )
        pred, _ = uk.execute("points", lon_test, lat_test)

        y_true.extend(test_values)
        y_pred.extend(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    RSS = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    R2 = 1 - RSS / ss_tot if ss_tot != 0 else float("nan")
    # krme = np.mean(y_pred - y_true) / np.mean(y_true)
    # krmse = np.sqrt(np.mean((y_pred - y_true) ** 2)) / np.mean(y_true)

    krme = np.mean(y_pred - y_true)  # 新公式 (3)
    std_y = np.std(y_true)
    krmse = np.mean(((y_pred - y_true) ** 2) / (std_y ** 2))

    print(f"RSS: {RSS:.2f}")
    print(f"R²: {R2:.4f}")
    print(f"KRME: {krme:.4f}")
    print(f"KRMSE: {krmse:.4f}")

    return RSS, R2, krme, krmse


def dms_formatter(x, pos=None, is_lat=False):
    # 将度数x格式化成 度°分'秒'' + E/N或W/S方向,例如：100°40′0″ E 或 30°40′0″ N

    deg = int(x)
    min_float = abs((x - deg) * 60)
    minute = int(min_float)
    second = int(round((min_float - minute) * 60))

    # 修正秒分进位
    if second == 60:
        second = 0
        minute += 1
    if minute == 60:
        minute = 0
        deg += 1

    # 方向判断
    if is_lat:
        direction = 'N' if x >= 0 else 'S'
    else:
        direction = 'E' if x >= 0 else 'W'

    deg_abs = abs(deg)
    return f"{deg_abs}°{minute}′ {direction}"

def draw_north_arrow(ax, x=0.95, y=0.85, size=0.08):

    # 在 ax 图的坐标轴比例位置 (x, y) 画一个北箭头（默认右上角偏下）
    ax.annotate('N',
                xy=(x, y), xytext=(x, y - size),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=14,
                xycoords=ax.transAxes)

def save_kriging_contour_plot(
    grid_x, grid_y, z, contour_levels,
    sample_points = None,
    filename="kriging_contour_map.png", output_dir=".\output", dpi=300):
    """
    绘制克里金插值等值图并保存为 PNG 文件，含色带、度分秒坐标、比例尺、指南针与边界框。

    参数:
    - grid_x, grid_y: 网格坐标（np.meshgrid）
    - z: 插值结果二维数组
    - contour_levels: 等值线分级
    - basin_boundary: 多边形边界（Shapely Polygon 或 MultiPolygon）
    - filename: 输出文件名
    - output_dir: 保存目录
    - dpi: 图像分辨率
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    try:
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

        # 插值等值图
        cs = ax.contourf(grid_x, grid_y, z, levels=contour_levels, cmap=palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap)
        cbar = fig.colorbar(cs, ax=ax, orientation="vertical", shrink=0.4, pad=0.03)
        cbar.set_label("地层厚度(m)", fontsize=12)

        # 绘制原始钻井点（支持 list[dict] 或 list[tuple] 格式）
        if sample_points:
            try:
                # 如果是 dict，则提取坐标字段
                if isinstance(sample_points[0], dict):
                    if "lng" in sample_points[0] and "lat" in sample_points[0]:
                        sample_points = [(pt["lng"], pt["lat"]) for pt in sample_points]
                    elif "lon" in sample_points[0] and "lat" in sample_points[0]:
                        sample_points = [(pt["lon"], pt["lat"]) for pt in sample_points]
                    elif "x" in sample_points[0] and "y" in sample_points[0]:
                        sample_points = [(pt["x"], pt["y"]) for pt in sample_points]
                    else:
                        raise ValueError("sample_points 中的字段名不支持，需包含 lng/lat, lon/lat 或 x/y")
                sample_points = np.array(sample_points)
                ax.scatter(sample_points[:, 0], sample_points[:, 1],
                           c='blue', s=10, label='钻井点', zorder=5)
            except Exception as point_err:
                print(f"⚠️ 无法绘制钻井点: {point_err}")

        # 经纬度格式化
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: dms_formatter(val, pos, is_lat=False)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: dms_formatter(val, pos, is_lat=True)))

        # 指南针
        draw_north_arrow(ax, x=0.92, y=0.92, size=0.15)

        # 设置坐标范围
        ax.set_xlim(102, 111)
        ax.set_ylim(27, 33)
        ax.set_aspect('equal', adjustable='box')

        # 设置刻度
        xticks = np.arange(102, 112, 1)
        yticks = np.arange(27, 34, 1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        # 横坐标隔一个显示一个标签
        xtick_labels = [f"{x}°E" if i % 2 == 0 else "" for i, x in enumerate(xticks)]
        ytick_labels = [f"{y}°N" for y in yticks]

        ax.set_xticklabels(xtick_labels, fontsize=10)
        ax.set_yticklabels(ytick_labels, fontsize=10)

        # 保存图像
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, format="png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"✅ 插值图已保存到: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ 插值图保存失败: {e}")
        return None

def compute_krme_krmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # mean_y = np.mean(y_true)
    # krme = np.mean(y_pred - y_true) / mean_y
    # krmse = np.sqrt(np.mean((y_pred - y_true) ** 2)) / mean_y

    krme = np.mean(y_pred - y_true)  # 新公式 (3)
    std_y = np.std(y_true)
    krmse = np.mean(((y_pred - y_true) ** 2) / (std_y ** 2))

    return krme, krmse


@app.post("/kriging/vector")
async def kriging_vector(request: Request):
    try:
        data = await request.json()
        features = data.get("features", [])
        print("📥 接收到前端数据，点数:", len(features))

        if len(features) < 3:
            return {"error": "至少需要3个数据点进行插值"}
        # 获取插值参数
        params = data.get("krigingParams", {})
        kriging_method = params.get("model", "ok").lower()
        kriging_model_input = params.get("variogram_model", "spherical").lower()
        auto_optimize = params.get("autoOptimizeModel", False)
        sigma = float(params.get("smoothSigma", 0))
        property_field = params.get("property", "level")
        output_path = params.get("outputPath", "")

        # 提取有效点
        valid_points = [
            (f["geometry"]["coordinates"][0],
             f["geometry"]["coordinates"][1],
             f["properties"].get(property_field))
            for f in features
            if f["properties"].get(property_field) is not None
        ]

        if len(valid_points) < 3:
            return {"error": f"有效的 {property_field} 数据不足，无法插值"}

        # 拆分经纬度与值
        lons, lats, values = zip(*valid_points)
        lons = np.array(lons)
        lats = np.array(lats)
        values = np.array(values, dtype=float)
        coords = np.column_stack((lons, lats))

        # 坐标投影至米制平面（EPSG:3857）
        transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        projected_coords = np.array([transformer.transform(lon, lat) for lon, lat in coords])

        if auto_optimize:
            # 只优化前端传入的模型类型的参数，不遍历候选模型
            best_result = optimize_single_model(kriging_model_input, projected_coords, values)
            if best_result is None:
                print("❌ 优化失败，无法拟合有效的半变异函数")
                return {"error": "自动优化失败，无法拟合有效的半变异函数"}

            kriging_model = best_result["model"]
            nugget = max(best_result["nugget"], 0.05 * best_result["sill"])
            partial_sill = best_result["partial_sill"]
            sill = nugget + partial_sill
            nugget_ratio = nugget / sill if sill != 0 else 0
            range_ = best_result["range"]
            RSS = best_result["rss"]
            R2 = best_result["r2"]
            exp_lags = best_result["exp_lags"]
            exp_gamma = best_result["exp_gamma"]
            fit_gamma = best_result["fit_gamma"]

            variogram_plot_base64 = generate_variogram_plot_from_data(
                exp_lags, exp_gamma, fit_gamma, kriging_model,
                title_suffix=f"({kriging_method.upper()})"
            )

            print("🟢 自动优化后模型参数：")
            print(f"模型: {kriging_model}")
            print(f"块金值 Nugget: {nugget:.4f}")
            print(f"基台值 Sill: {sill:.4f}")
            print(f"偏基台 Partial Sill: {partial_sill:.4f}")
            print(f"块金比例 Nugget Ratio: {nugget_ratio:.4f}")
            print(f"变程 Range: {range_:.4f}")
            print(f"残差平方和 RSS: {RSS:.6f}")
            print(f"决定系数 R²: {R2:.4f}")


        else:

            # ✅ 用户自选模型，使用 skgstat Variogram 拟合

            kriging_model = kriging_model_input
            V = Variogram(projected_coords, values, model=kriging_model, normalize=False, fit_method="trf",
                          fit_range=(0, 300_000))
            variogram_plot_base64 = generate_variogram_plot(V, kriging_model,
                                                            title_suffix=f"({kriging_method.upper()})")
            try:
                sill, range_, nugget = V.parameters
                if any(param <= 0 for param in [sill, range_]):
                    raise ValueError("拟合参数异常")
            except Exception:
                sill = np.var(values)
                range_ = max(lons.max() - lons.min(), lats.max() - lats.min()) / 3
                nugget = 1.2 * sill
            partial_sill = sill - nugget
            nugget_ratio = nugget / sill if sill != 0 else 0
            exp_lags = V.bins
            exp_gamma = V.experimental
            fit_gamma = V.fitted_model(exp_lags)
            RSS = np.sum((exp_gamma - fit_gamma) ** 2)
            SS_tot = np.sum((exp_gamma - np.mean(exp_gamma)) ** 2)
            R2 = 1 - RSS / SS_tot if SS_tot != 0 else float('nan')
            print("🟡 未优化模型参数（使用用户选择模型）:")
            print(f"模型: {kriging_model}")
            print(f"块金值 Nugget: {nugget:.4f}")
            print(f"基台值 Sill: {sill:.4f}")
            print(f"偏基台 Partial Sill: {partial_sill:.4f}")
            print(f"块金比例 Nugget Ratio: {nugget_ratio:.4f}")
            print(f"变程 Range: {range_:.4f}")
            print(f"残差平方和 RSS: {RSS:.6f}")
            print(f"决定系数 R²: {R2:.4f}")
        if basin_union:
            minx, miny, maxx, maxy = basin_union.bounds
        else:
            minx, maxx = lons.min(), lons.max()
            miny, maxy = lats.min(), lats.max()

        expand_x = (maxx - minx) * 0.5
        expand_y = (maxy - miny) * 0.5
        minx -= expand_x
        maxx += expand_x
        miny -= expand_y
        maxy += expand_y

        grid_res = 400
        grid_lon = np.linspace(minx, maxx, grid_res)
        grid_lat = np.linspace(miny, maxy, grid_res)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

        # ✅ 校验模型合法性
        valid_models_ok = {"spherical", "exponential", "gaussian", "linear", "circular"}
        valid_models_uk = {"spherical", "exponential", "gaussian", "linear"}

        if kriging_method == "ok":
            if kriging_model not in valid_models_ok:
                return {"error": f"普通克里金法不支持模型 '{kriging_model}'"}
        elif kriging_method == "uk":
            if kriging_model not in valid_models_uk:
                return {"error": f"泛克里金法不支持模型 '{kriging_model}'"}
        else:
            return {"error": f"未知的克里金方法 '{kriging_method}'"}

        if kriging_method == "uk":
            drift_type = (params.get("drift") or "linear").lower()
            if drift_type == "linear":
                drift_terms = ["regional_linear"]
            elif drift_type == "quadratic":
                drift_terms = ["regional_quadratic"]
            else:
                raise ValueError(f"不支持的泛克里金漂移类型: {drift_type}")

            # ✅ 输出使用的配置
            print(f"✅ 使用方法: Universal Kriging（泛克里金）")
            print(f"✅ 漂移类型: {drift_type}")
            print(f"✅ 半变异模型: {kriging_model}")

            import io, sys, re
            buffer = io.StringIO()
            sys_stdout = sys.stdout
            sys.stdout = buffer  # 重定向 stdout 捕获 pykrige 输出

            try:
                kriging = UniversalKriging(
                    lons, lats, values,
                    variogram_model=kriging_model,
                    drift_terms=drift_terms,
                    verbose=True,
                    enable_plotting=False
                )
                z_raw, ss = kriging.execute("grid", grid_lon, grid_lat)

                # 交叉验证计算RSS和R2
                # RSS, R2 = cross_validation_uk(lons, lats, values, kriging_model, drift_terms)

                # K折交叉验证
                RSS, R2, KRME, KRMSE = kfold_uk_cross_validation(
                    lons, lats, values, kriging_model, drift_terms, 5
                )
                print(f"泛克里金交叉验证残差平方和 RSS: {RSS:.4f}")
                print(f"泛克里金交叉验证决定系数 R²: {R2:.4f}")
            finally:
                sys.stdout = sys_stdout  # 恢复 stdout

            # 提取 pykrige 输出中的拟合参数
            output = buffer.getvalue()
            print(output)

            nugget_match = re.search(r"Nugget:\s*([0-9.eE+-]+)", output)
            sill_match = re.search(r"Full Sill:\s*([0-9.eE+-]+)", output)
            psill_match = re.search(r"Partial Sill:\s*([0-9.eE+-]+)", output)
            range_match = re.search(r"Range:\s*([0-9.eE+-]+)", output)

            if all([nugget_match, sill_match, psill_match, range_match]):
                nugget_fitted = float(nugget_match.group(1))
                sill_fitted = float(sill_match.group(1))
                psill_fitted = float(psill_match.group(1))
                range_fitted = float(range_match.group(1))
                nugget_ratio = nugget_fitted / sill_fitted if sill_fitted else 0

                print("📊 泛克里金实际使用模型参数：")
                print(f"块金值 Nugget: {nugget_fitted:.4f}")
                print(f"基台值 Sill: {sill_fitted:.4f}")
                print(f"偏基台 Partial Sill: {psill_fitted:.4f}")
                print(f"块金比例 Nugget Ratio: {nugget_ratio:.4f}")
                print(f"变程 Range (m): {range_fitted:.4f}")
            else:
                print("⚠️ 未能从 pykrige 输出中提取模型参数")

        elif kriging_method == "ok":
            print(f"✅ 使用方法: Ordinary Kriging（普通克里金）")
            print(f"✅ 半变异模型: {kriging_model}")
            print(f"✅ 使用参数: sill={sill}, range={range_}, nugget={nugget}")

            kriging = OrdinaryKriging(
                lons, lats, values,
                variogram_model=kriging_model,
                variogram_parameters={"sill": sill, "range": range_, "nugget": nugget},
                verbose=True,
                enable_plotting=False
            )
            z_raw, ss = kriging.execute("grid", grid_lon, grid_lat, backend="vectorized")
            # 调用留一交叉验证函数
            try:
                RSS, R2, KRME, KRMSE, best_nugget = loo_cross_validation_ok(
                    lons, lats, values, sill, nugget, range_, kriging_model, auto_optimize_nugget=True
                )
                print(f"普通克里金优化后 nugget = {best_nugget:.4f}")
                print(f"普通克里金：RSS = {RSS:.4f}, R2 = {R2:.4f}, KRME = {KRME:.4f}, KRMSE = {KRMSE:.4f}")
            except Exception as e:
                print(f"⚠️ 快速交叉验证失败: {e}")
                RSS, R2 = None, None

        else:
            return {"error": f"未知的克里金方法 '{kriging_method}'"}

        # 平滑+裁剪
        z_filtered = gaussian_filter(z_raw, sigma=sigma) if sigma > 0 else z_raw
        z = mask_outside_boundary(grid_x, grid_y, z_filtered, basin_union)

        # 检查插值有效性
        if np.isclose(np.nanmin(z), np.nanmax(z)):
            return {"error": "插值结果无差异", "value": float(np.nanmin(z))}

        z_flat = z[~np.isnan(z)]
        if len(z_flat) < 11:
            return {"error": "有效区域 z 值不足，无法使用 Jenks 分级"}

        # 生成等值线
        contour_levels = jenkspy.jenks_breaks(z_flat, 11)
        contour_levels = np.unique(contour_levels)
        if len(contour_levels) < 2:
            return {"error": "等值线级别不足，无法绘制等值线"}

        # 从 features 中提取坐标元组列表 (lng, lat)
        sample_points = [
            (f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1])
            for f in features
            if f.get("geometry") and f["geometry"]["type"].lower() == "point"
        ]
        contour_img_path = save_kriging_contour_plot(grid_x, grid_y, z, contour_levels, sample_points)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        cs = ax.contourf(grid_x, grid_y, z, levels=contour_levels, cmap=palettable.colorbrewer.diverging.RdYlBu_11_r.mpl_colormap)
        plt.close(fig)

        level_polygons = []
        for i in range(len(contour_levels) - 1):
            polys = []
            if i >= len(cs.allsegs):
                level_polygons.append(None)
                continue
            for seg in cs.allsegs[i]:
                if len(seg) < 3:
                    continue
                if not np.array_equal(seg[0], seg[-1]):
                    seg = np.vstack([seg, seg[0]])
                poly = ShapelyPolygon(seg)
                if poly.is_valid and poly.area > 1e-10:
                    polys.append(poly)
            if polys:
                level_polygons.append(unary_union(polys))
            else:
                level_polygons.append(None)

        features_result = []
        valid_contour_levels = []
        for i in range(len(level_polygons)):
            current_poly = level_polygons[i]
            if current_poly is None or current_poly.is_empty:
                continue
            next_poly = level_polygons[i + 1] if i + 1 < len(level_polygons) else None
            band = current_poly
            if next_poly and not next_poly.is_empty:
                band = current_poly.difference(next_poly)
            if basin_union:
                band = band.intersection(basin_union)
            if band.is_empty:
                continue
            geoms = extract_polygons(band)
            for geom in geoms:
                if not geom.is_valid or geom.area < 1e-6:
                    continue
                coords = list(geom.exterior.coords)
                coords_rounded = [(round(x, 6), round(y, 6)) for x, y in coords]
                gj_poly = GeoJSONPolygon([coords_rounded])
                features_result.append(Feature(geometry=gj_poly, properties={
                    "value": float(contour_levels[i]),
                    "min_value": float(contour_levels[i]),
                    "max_value": float(contour_levels[i + 1])
                }))
            valid_contour_levels.append(contour_levels[i])

        # 生成shapefile及zip，并返回下载地址
        zip_download_url = None
        if output_path:
            try:
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                save_features_to_shapefile(features_result, output_path)
                zip_path = zip_shapefile(output_path)
                zip_filename = os.path.basename(zip_path)
                zip_download_url = f"/download/{zip_filename}"
                print(f"✅ Shapefile 及 zip 已保存，下载地址: {zip_download_url}")
            except Exception as e:
                print(f"❌ 保存 Shapefile 或压缩 zip 失败: {e}")

        return {
            "type": "FeatureCollection",
            "features": features_result,
            "properties": {
                "contour_levels": valid_contour_levels,
                "contour_image_path": contour_img_path,
                "download_url": zip_download_url,
                "variogram_plot_base64": variogram_plot_base64,
                "used_kriging_model": kriging_model,
                "nugget": nugget,
                "sill": sill,
                "partial_sill": partial_sill,
                "nugget_ratio": nugget_ratio,
                "range": range_,
                "fit_rss": best_result["rss"] if auto_optimize else None
            }
        }

    except Exception as e:
        tb = traceback.format_exc()
        print("❌ Kriging error:", e)
        print(tb)
        return {"error": str(e), "traceback": tb, "details": "插值过程发生错误"}


@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join("output", filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path, media_type="application/zip", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="文件未找到")



import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)



# 判断是否正态分布
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
#
# # === 1. 读取 Excel 文件 ===
# # 替换为你的文件路径和 Sheet 名称
# file_path = 'stratum.xlsx'
# sheet_name = 'Sheet1'
# column_name = 'start_depth'
#
# # 读取数据
# df = pd.read_excel(file_path, sheet_name=sheet_name)
#
# # 去除缺失值，仅保留有效数值
# data = df[column_name].dropna().values
#
# # === 2. 绘制直方图 ===
# plt.figure(figsize=(8, 6))
# plt.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7, density=True)
#
# # 正态分布拟合曲线
# mu, std = norm.fit(data)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'r--', linewidth=2, label=f'Normal fit: μ={mu:.2f}, σ={std:.2f}')
#
# # 标注图例和标题
# plt.title(f"{column_name} 的直方图（含正态拟合）", fontsize=14)
# plt.xlabel("值")
# plt.ylabel("频率密度")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
