import io
import os
import base64
import fiona
import numpy as np
import jenkspy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from adjustText import adjust_text
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects



matplotlib.rcParams['font.sans-serif'] = ['SimHei']


# ================= 图件核心类 =================
class MapRenderer:
    def lighten_cmap(self, cmap, factor=1.3):
        colors = cmap(np.linspace(0, 1, cmap.N))
        colors[:, :3] = np.clip(colors[:, :3] * (1/factor) + (1 - 1/factor), 0, 1)
        return mcolors.ListedColormap(colors)

    async def load_boundary_from_geojson(self, filepath="scBasin.geojson"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"未找到 {filepath}")
        with fiona.open(filepath, "r", encoding="utf-8") as src:
            geoms = [shape(feat["geometry"]) for feat in src]
        if len(geoms) == 1:
            return geoms[0]
        return MultiPolygon([g for g in geoms if isinstance(g, Polygon)])

    async def mask_z_by_boundary(self, grid_x, grid_y, z, boundary_geom=None):
        boundary_geom = boundary_geom or await self.load_boundary_from_geojson()
        if isinstance(boundary_geom, Polygon):
            polys = [boundary_geom]
        elif isinstance(boundary_geom, MultiPolygon):
            polys = list(boundary_geom.geoms)
        else:
            polys = []

        paths = [Path(np.array(p.exterior.coords)) for p in polys if len(p.exterior.coords) >= 3]
        points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
        mask = np.zeros(len(points), dtype=bool)
        for path in paths:
            mask |= path.contains_points(points)
        mask = mask.reshape(z.shape)
        return np.where(mask, z, np.nan)

    def dms_formatter(self, x, pos=None, is_lat=False):
        deg = int(x)
        min_float = abs((x - deg) * 60)
        minute = int(min_float)
        second = int(round((min_float - minute) * 60))
        if second == 60:
            second = 0
            minute += 1
        if minute == 60:
            minute = 0
            deg += 1
        direction = 'N' if is_lat and x >= 0 else 'S' if is_lat else 'E' if x >= 0 else 'W'
        return f"{abs(deg)}°{minute}′ {direction}"

    def draw_north_arrow(self, ax, x=0.95, y=0.75, size=0.08):
        ax.annotate('N',
                    xy=(x, y), xytext=(x, y - size),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=14,
                    xycoords=ax.transAxes)

    async def _render_map_image(self, grid_x, grid_y, z, points=None, boundary_geom=None,
                                task_text=None, variable="thickness", lithology=None,
                                smooth_sigma=0, n_classes=11, colormap="RdYlBu",
                                lighten=False):

        z_masked = await self.mask_z_by_boundary(np.array(grid_x), np.array(grid_y), np.array(z), boundary_geom)

        if smooth_sigma > 0:
            z_masked = gaussian_filter(z_masked, sigma=smooth_sigma)

        z_flat = z_masked[~np.isnan(z_masked)]
        if len(z_flat) == 0:
            levels = np.linspace(0, 1, 6)
        else:
            # 使用 jenkspy 确保与矢量化使用相同的分级
            if n_classes is None:
                n_classes = 11  # 保持默认值
            levels = jenkspy.jenks_breaks(z_flat, n_classes)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        base_cmap = plt.get_cmap(colormap + "_r")
        cmap = self.lighten_cmap(base_cmap, factor=1.3) if lighten else base_cmap

        cs = ax.contourf(grid_x, grid_y, z_masked, levels=levels, cmap=cmap)
        cbar = fig.colorbar(cs, ax=ax, orientation="vertical", shrink=0.4, pad=0.03)

        # ----------------------------
        # 设置 colorbar 标签
        # ----------------------------
        is_percentage = variable.endswith("占比") or variable.lower() == "ratio"

        if variable == "地层厚度":
            cbar.set_label("地层厚度 (m)", fontsize=12)
        elif is_percentage:
            cbar.set_label(f"{lithology or variable} (%)", fontsize=12)
            cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        else:
            cbar.set_label(variable, fontsize=12)

        # ----------------------------
        # 绘制钻井点
        # ----------------------------
        if points:
            lons, lats = [], []
            texts = []
            y_offset = 0.02

            for p in points:
                lon = (
                    p.get("lon") or p.get("lng") or p.get("x") or
                    p.get("longitude") or p.get("geo_X")
                )
                lat = (
                    p.get("lat") or p.get("y") or p.get("latitude") or
                    p.get("geo_Y")
                )
                name = (
                    p.get("name") or p.get("well_name") or
                    p.get("井名") or ""
                )

                if lon is not None and lat is not None:
                    try:
                        lon = float(lon)
                        lat = float(lat)
                    except Exception:
                        continue

                    lons.append(lon)
                    lats.append(lat)

                    # 绘制井名
                    txt = ax.text(
                        lon,
                        lat + y_offset,
                        str(name),
                        fontsize=5,
                        ha="center",
                        va="bottom",
                        color="black",
                        zorder=6,
                        path_effects=[
                            path_effects.withStroke(linewidth=2, foreground="white")
                        ],
                    )
                    texts.append(txt)

            if lons and lats:
                ax.scatter(
                    lons, lats,
                    c="red",
                    s=20,
                    edgecolors="white",
                    linewidths=0.8,
                    label="钻井点",
                    zorder=5
                )

            # 自动调整注记，避免堆叠
            if texts:
                adjust_text(
                    texts,
                    only_move={'points': 'y', 'texts': 'y'},  # 只在垂直方向调整
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)  # 可选：加指向箭头
                )

        # ----------------------------
        # 绘制边界
        # ----------------------------
        try:
            basin_geom = boundary_geom or await self.load_boundary_from_geojson()
            polys = [basin_geom] if isinstance(basin_geom, Polygon) else list(basin_geom.geoms)
            for poly in polys:
                x, y = poly.exterior.xy
                ax.plot(x, y, color="black", linewidth=1.2)
        except Exception as e:
            print("⚠️ 无法加载边界:", e)

        # ----------------------------
        # 坐标轴格式化
        # ----------------------------
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: self.dms_formatter(val, pos, is_lat=False)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: self.dms_formatter(val, pos, is_lat=True)))
        ax.set_xlim(102, 111)
        ax.set_ylim(27, 33)
        ax.set_aspect("equal", adjustable="box")

        # ----------------------------
        # 绘制北箭头
        # ----------------------------
        self.draw_north_arrow(ax, x=0.92, y=0.92, size=0.12)

        # ----------------------------
        # 设置标题
        # ----------------------------
        if task_text:
            title = task_text[2:].strip() if task_text.startswith("绘制") else task_text.strip()
        else:
            if is_percentage:
                title = f"{variable}"
            elif variable == "地层厚度":
                title = f"地层厚度分布图"
            else:
                title = f"{variable}"

        ax.set_title(title, fontsize=14, fontweight="bold")

        # ----------------------------
        # 输出为 base64
        # ----------------------------
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    async def generate_contours_polygons_geojson(
            self, grid_x, grid_y, z, n_classes=None, boundary_geom=None,
            colormap="RdYlBu", lighten=False
    ):
        """
        将栅格插值结果转成等值面 GeoJSON，并按四川盆地边界裁剪
        """
        from shapely.ops import unary_union

        # ===========================
        # 1. 准备盆地边界几何
        # ===========================
        if boundary_geom is None:
            try:
                # 如果外部没传，就默认加载 scBasin.geojson
                boundary_geom = await self.load_boundary_from_geojson()
            except Exception as e:
                print("⚠️ 无法加载边界，等值面将不进行几何裁剪:", e)
                boundary_geom = None

        if boundary_geom is not None:
            # MultiPolygon 统一合并成一个几何，避免很多小块
            if isinstance(boundary_geom, MultiPolygon):
                basin_geom = unary_union(boundary_geom)
            else:
                basin_geom = boundary_geom
        else:
            basin_geom = None

        # ===========================
        # 2. 按盆地边界对栅格做 mask
        # ===========================
        z_masked = await self.mask_z_by_boundary(
            np.array(grid_x), np.array(grid_y), np.array(z), boundary_geom
        )
        z_flat = z_masked[~np.isnan(z_masked)]

        # ===========================
        # 3. 等级分级（Jenks 或默认）
        # ===========================
        if len(z_flat) == 0:
            levels = np.linspace(0, 1, 6).tolist()
        else:
            if n_classes is None:
                n_classes = min(11, max(5, len(z_flat) // 10))
            levels = jenkspy.jenks_breaks(z_flat, n_classes)

        # ===========================
        # 4. 配置颜色映射
        # ===========================
        base_cmap = plt.get_cmap(colormap + "_r")
        cmap = self.lighten_cmap(base_cmap, factor=1.3) if lighten else base_cmap
        norm = mcolors.BoundaryNorm(levels, cmap.N)

        # ===========================
        # 5. 生成等值面（仍然是矩形网格）
        # ===========================
        fig = plt.figure()
        cs = plt.contourf(grid_x, grid_y, z_masked, levels=levels, cmap=cmap)
        plt.close(fig)

        # ===========================
        # 6. 提取等值面 polygon，并与盆地相交
        # ===========================
        features = []

        for i, segs in enumerate(cs.allsegs):
            level = cs.levels[i]
            color = mcolors.to_hex(cmap(norm(level)))  # 当前等级对应颜色

            for seg in segs:
                if len(seg) < 3:
                    continue

                poly = Polygon(seg)
                if not poly.is_valid or poly.is_empty:
                    continue

                # ★ 核心：与盆地边界做几何裁剪 ★
                if basin_geom is not None:
                    poly = poly.intersection(basin_geom)
                    if poly.is_empty:
                        continue

                features.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "value": float(level),
                        "fill": color
                    }
                })

        # ===========================
        # 7. 打包成 GeoJSON 输出
        # ===========================
        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "contour_levels": [float(l) for l in levels],
                "colormap": colormap,
                "lighten": lighten
            }
        }

    async def render_map(self, grid_x, grid_y, z, points=None, boundary_geom=None,
                         task_text=None, variable="thickness", lithology=None,
                         smooth_sigma=0, n_classes=None, colormap="RdYlBu",
                         lighten=False):
        image_base64 = await self._render_map_image(grid_x, grid_y, z, points, boundary_geom,
                                                    task_text, variable, lithology, smooth_sigma,
                                                    n_classes, colormap, lighten)
        geojson = await self.generate_contours_polygons_geojson(grid_x, grid_y, z, n_classes, boundary_geom, colormap, lighten)
        return {
            "image_base64": image_base64,
            "geojson": geojson,
            "colormap": colormap,
            "lighten": lighten
        }
