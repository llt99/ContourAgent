import asyncio
from collections import defaultdict
import logging

import numpy as np

from kriging import Interpolator
from mcp_server import mcp_server
from pprint import pformat
from prompt import SYSTEM_TO_STRATA
# ------------------------
# 抽象 Agent
# ------------------------
class Agent:
    async def run(self, ctx, context: dict) -> dict:
        """必须实现 run 方法"""
        raise NotImplementedError("Agent 必须实现 run 方法")

# ------------------------
# DummyContext，用于测试或无 MCP 模式
# ------------------------
class DummyContext:
    async def info(self, msg: str, **kwargs):
        logging.info(msg)

    async def report_progress(self, progress, total=1.0, message=""):
        logging.info(f"[PROGRESS] {progress*100:.1f}% - {message}")

    async def error(self, msg: str, **kwargs):
        logging.error(msg)

class ExtendedContext:
    def __init__(self, ctx=None):
        self.ctx = ctx

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx)

    async def info(self, msg: str):
        if hasattr(self.ctx, "info"):
            try:
                await self.ctx.info(msg)
                return
            except Exception:
                pass
        logging.info(msg)

    async def error(self, msg: str):
        if hasattr(self.ctx, "error"):
            try:
                await self.ctx.error(msg)
                return
            except Exception:
                pass
        logging.error(msg)

    async def report_progress(self, progress: float, total: float = 1.0, message: str = ""):
        if hasattr(self.ctx, "report_progress"):
            try:
                await self.ctx.report_progress(progress, total, message)
                return
            except Exception:
                pass
        logging.info(f"[PROGRESS] {progress*100:.1f}% {message}")

    async def call_tool(self, name: str, **kwargs):
        """优先使用 ctx.call_tool，否则 fallback 到全局 mcp_server"""
        # 确保 ctx 参数始终传入
        if "ctx" not in kwargs:
            kwargs["ctx"] = self  # self 是 ExtendedContext

        # 先尝试 MCP Context 内调用
        if hasattr(self, "_ctx") and hasattr(self._ctx, "call_tool"):
            try:
                return await self._ctx.call_tool(name, **kwargs)
            except Exception:
                logging.warning(f"⚠️ MCP Context 调用 {name} 失败，尝试全局 mcp_server")

        # fallback 全局 mcp_server
        tool = mcp_server._local_tools.get(name)
        if not tool:
            raise RuntimeError(f"工具 {name} 未注册")

        if asyncio.iscoroutinefunction(tool):
            return await tool(**kwargs)
        else:
            return tool(**kwargs)


# ------------------------
# NLP Agent
# ------------------------
class NLPAgent(Agent):
    async def run(self, ctx: ExtendedContext, context: dict) -> dict:
        text = context.get("text")
        if not text:
            context.setdefault("errors", []).append("缺少 text")
            context["plan"] = {"pipeline": []}
            return context

        try:
            # 1. 调用 NLP 工具解析
            result = await ctx.call_tool(
                "parse_text_tool_mcp",
                user_text=text,
                context=context
            )
            task = context.get("task", {}) or {}

            # 2. 获取 MCPContext 中已有的参数
            last_params = getattr(mcp_server, "context", {}).params or {}
            last_task = getattr(mcp_server, "context", {}).task or {}

            # 3. 必要参数清单
            required_keys = ["region", "stratum", "variable", "plot"]

            # 4. 遍历缺参情况，尝试用历史参数补齐
            for key in required_keys:
                if not task.get(key):
                    if last_params.get(key):
                        task[key] = last_params[key]
                        task.setdefault("warnings", []).append(
                            f"参数 {key} 缺失，已自动继承上一次的值: {last_params[key]}"
                        )
                    elif last_task.get(key):
                        task[key] = last_task[key]
                        task.setdefault("warnings", []).append(
                            f"参数 {key} 缺失，已自动继承上一次的值: {last_task[key]}"
                        )
                    else:
                        task.setdefault("warnings", []).append(f"缺少必要参数: {key}")

            # 5. 更新 context
            context["task"] = task
            context["params"] = {**last_params, **task}  # 合并覆盖

            # 6. 写回 MCPContext（关键！）
            mcp_server.context.task.update(task)
            mcp_server.context.params.update(context["params"])
            # 将 plan 也写入 context，确保可追溯
            mcp_server.context.task["plan"] = context.get("plan")

            # 7. 输出状态
            await ctx.info("� NLPAgent 执行后，全局 MCPContext 状态：")
            await ctx.info(pformat({
                "task": mcp_server.context.task,
                "params": mcp_server.context.params
            }, width=80))

        except Exception as e:
            context.setdefault("errors", []).append(str(e))
            await ctx.error(f"NLP 解析失败: {e}")

        return context


# ------------------------
# Feedback Agent
# ------------------------
class FeedbackAgent(Agent):
    async def run(self, ctx: ExtendedContext, context: dict) -> dict:
        feedback_text = context.get("feedback")
        if not feedback_text:
            await ctx.info("⚠️ 无用户反馈，跳过 FeedbackAgent")
            return context
        try:
            # 修正：确保所有参数都通过关键字传递，避免位置参数冲突
            result = await ctx.call_tool(
                "parse_user_feedback_tool",
                feedback_text=feedback_text,
                context=context
            )
            
            # 更新 MCP 上下文中的核心参数
            if "params" in result.get("mcp_context", {}):
                mcp_server.context.params.update(result["mcp_context"]["params"])

            # 将解析出的参数也更新到当前任务的本地上下文中
            context.update(result["mcp_context"]["params"])
            await ctx.info("✅ 用户反馈已更新到 MCPContext")
        except Exception as e:
            context.setdefault("errors", []).append(str(e))
            mcp_server.context.add_error(str(e))
            await ctx.error(f"反馈解析失败: {e}")
        return context



# ------------------------
# Data Agent
# ------------------------
class DataAgent(Agent):
    async def run(self, ctx: ExtendedContext, context: dict) -> dict:
        # 🟦 0. 检查是否传入了Excel数据
        excel_data = context.get("excel_data")
        if excel_data:
            await ctx.info(f"🔄 检测到传入的Excel数据: {len(excel_data)} 条记录")
            
            # 将Excel数据转换为标准格式
            data_points = []
            for row in excel_data:
                try:
                    # 尝试找到坐标和值字段
                    lon = row.get("lon") or row.get("经度") or row.get("x")
                    lat = row.get("lat") or row.get("纬度") or row.get("y")
                    value = row.get("value") or row.get("thickness") or row.get("stratum_thickness") or row.get("厚度") or 0
                    
                    if lon is not None and lat is not None:
                        data_points.append({
                            "lon": float(lon),
                            "lat": float(lat),
                            "value": float(value),
                            "well_name": row.get("well_name") or row.get("井名") or ""
                        })
                except Exception as e:
                    await ctx.info(f"⚠️ 跳过无效数据行: {row}, 错误: {e}")
                    continue
            
            if not data_points:
                await ctx.error("❌ Excel数据转换后无有效数据点")
                return context
            
            await ctx.info(f"✅ Excel数据转换完成: {len(data_points)} 个有效井点")
            
            # 写入本地上下文
            context["data_points"] = {"rows": data_points}
            
            # 写入MCPContext
            mcp_server.context.data["data_points"] = data_points
            mcp_server.context.data["query_result"] = {"rows": data_points}
            mcp_server.context.data["task_params"] = {
                "region": context.get("task", {}).get("region"),
                "stratum": context.get("task", {}).get("stratum"),
                "variable": context.get("task", {}).get("variable"),
            }
            
            await ctx.info("✅ Excel数据已写入 MCPContext.data")
            return context

        # 🟦 1. 检查缓存有效性并尝试复用
        current_task = context.get("task", {})
        cached_data = mcp_server.context.data
        
        if cached_data.get("data_points"):
            cached_params = cached_data.get("task_params", {})
            critical_fields = ["region", "stratum", "variable"]
            
            can_reuse = True
            for field in critical_fields:
                if (current_task.get(field) or "") != (cached_params.get(field) or ""):
                    can_reuse = False
                    await ctx.info(f"🔄 缓存参数 '{field}' 不匹配: '{cached_params.get(field)}' vs '{current_task.get(field)}'")
                    break
            
            if can_reuse:
                await ctx.info("✅ 缓存参数匹配，正在复用 MCPContext 数据...")
                context["data_points"] = cached_data["query_result"]
                return context
            else:
                await ctx.info("🔥 缓存无效，正在清空并重新获取数据...")
                mcp_server.context.data.clear()
                mcp_server.context.results.clear() # Also clear results dependent on data

        # 🟦 2. 如果本地 context 已存在数据，也跳过
        if context.get("data_points"):
            await ctx.info("🔹 数据已在当前任务流中存在，跳过 DataAgent")
            return context

        task = context.get("task")
        if not task:
            await ctx.error("缺少 task，DataAgent 无法执行")
            return context

        # 🟦 3. 判断任务层级（系统层级或地层层级）
        system = task.get("system")
        stratum = task.get("stratum")
        variable = task.get("variable")

        # 🟦 4. 构造查询文本
        if system and not stratum:
            # 如果是系统层级任务，例如 “绘制川东二叠系粉砂岩分布图”
            query_text = f"查询 {system} 各井 {variable} 数据（包含全部子地层）"
        else:
            # 普通地层任务，例如 “绘制龙潭组煤岩分布图”
            query_text = f"查询 {stratum} 各井 {variable} 数据"

        await ctx.info(f"🧭 数据检索任务: {query_text}")

        # 🟦 5. 执行 SQL 查询
        try:
            query_result = await ctx.call_tool("text_to_sql_query_tool", query=query_text)
        except Exception as e:
            context.setdefault("errors", []).append(str(e))
            await ctx.error(f"数据查询失败: {e}")
            return context

        # 🟦 6. 若返回结果为空
        if not query_result or not query_result.get("rows"):
            await ctx.error("❌ 未获取到有效井点数据")
            return context

        rows = query_result["rows"]

        # 🟦 7. 如果是系统层级任务：合并所有子层点为一个统一点集
        if system and not stratum:
            all_points = []
            for row in rows:
                try:
                    all_points.append({
                        "lon": float(row.get("lon") or row.get("lng") or row.get("x")),
                        "lat": float(row.get("lat") or row.get("y")),
                        "value": float(row.get("value") or row.get("thickness") or row.get("ratio") or 0),
                    })
                except Exception:
                    continue
            await ctx.info(f"🧩 系统层级任务 {system}，已合并 {len(all_points)} 个井点用于统一插值")
            context["data_points"] = {"rows": all_points}
            # 将合并后的数据点也写入 MCPContext
            mcp_server.context.data["query_result"] = {"rows": all_points}
            mcp_server.context.data["data_points"] = all_points
            await ctx.info("✅ 系统层级数据已写入 MCPContext.data")
            return context

        # 🟦 8. 普通地层任务：直接返回结果
        context["data_points"] = query_result
        
        # 🟦 9. 将数据和任务参数写入 MCPContext
        mcp_server.context.data["query_text"] = query_text
        mcp_server.context.data["query_sql"] = query_result.get("sql")
        mcp_server.context.data["query_result"] = query_result
        mcp_server.context.data["data_points"] = query_result.get("rows", [])
        mcp_server.context.data["task_params"] = {
            "region": task.get("region"),
            "stratum": task.get("stratum"),
            "variable": task.get("variable"),
        }
        
        await ctx.info(f"✅ 获取 {len(rows)} 个井点数据，用于 {stratum or system} 插值计算")
        await ctx.info("✅ 数据及关联参数已写入 MCPContext.data")
        
        return context


# ------------------------
# Kriging Agent
# ------------------------
from collections import defaultdict


class KrigingAgent(Agent):
    async def run(self, ctx: ExtendedContext, context: dict) -> dict:
        rows = context.get("data_points", {}).get("rows", [])
        if not rows:
            await ctx.info("⚠️ 无数据点，跳过 KrigingAgent")
            return context

        points_by_stratum = defaultdict(list)
        task = context.get("task", {})
        kriging_results = {}

        # ---- 整理数据点 ----
        target_stratum = task.get("stratum", "")
        is_system_level = target_stratum in SYSTEM_TO_STRATA

        for p in rows:
            stratum = target_stratum if is_system_level else (p.get("stratum_name") or target_stratum)
            if not stratum:
                continue
            lon = p.get("lon") or p.get("geo_X")
            lat = p.get("lat") or p.get("geo_Y")
            value = p.get("value") or p.get("thickness") or p.get("ratio") or p.get("content")
            if None in (lon, lat, value):
                continue
            points_by_stratum[stratum].append({
                "lon": float(lon),
                "lat": float(lat),
                "value": float(value),
            })

        # ---- 执行插值 ----
        for stratum, pts in points_by_stratum.items():
            unique_points = pts

            # 系统级任务处理
            if is_system_level and stratum == target_stratum:
                await ctx.info(f"🧩 系统级任务 '{target_stratum}'，已合并 {len(pts)} 个数据点进行统一插值")

                seen_coords = set()
                cleaned = []
                for p in pts:
                    coords = (p["lon"], p["lat"])
                    if coords not in seen_coords:
                        cleaned.append(p)
                        seen_coords.add(coords)
                unique_points = cleaned

                if len(unique_points) < len(pts):
                    await ctx.info(f"ℹ️ 为 '{stratum}' 移除了 {len(pts) - len(unique_points)} 个重复坐标的数据点")

                if len(unique_points) < 5:
                    await ctx.error(f"❌ {stratum} 数据点过少 ({len(unique_points)} 个)，无法执行插值")
                    continue

            try:
                # ---- 参数提取与标准化 ----
                params = mcp_server.context.params
                method_raw = params.get("method") or task.get("method_code") or "auto"
                model_raw = params.get("variogram_model") or task.get("model_code") or "auto"

                # 方法映射（兼容多种写法）
                method_map = {
                    "普通克里金": "ok",
                    "泛克里金": "uk",
                    "universal_kriging": "uk",
                    "ordinary_kriging": "ok",
                }
                method = method_map.get(str(method_raw).lower(), str(method_raw).lower())

                # 半变异模型与优化控制
                candidate_models = params.get("candidate_models", ["spherical", "exponential", "gaussian"])
                auto_optimize = params.get("auto_optimize", True)
                drift = params.get("drift", "linear")

                # ✅ 如果用户指定了模型，则锁定模型并禁用自动优选
                if model_raw and model_raw != "auto":
                    candidate_models = [model_raw]
                    auto_optimize = False
                    await ctx.info(f"🎯 用户指定半变异函数模型: {model_raw}，已禁用自动模型优选")

                await ctx.info(f"🎯 {stratum} 调用 kriging_interpolate 工具执行插值...")
                await ctx.info(f"⚙️ 参数: method={method}, models={candidate_models}, drift={drift}")

                # ---- 执行插值 ----
                interp_result = await ctx.call_tool(
                    "kriging_interpolate",
                    points=unique_points,
                    method=method,
                    candidate_models=candidate_models,
                    autoOptimizeModel=auto_optimize,
                    drift=drift,
                )

                if not interp_result or "error" in interp_result:
                    raise Exception(interp_result.get("error", "插值返回空结果"))

                kriging_results[stratum] = interp_result
                await ctx.info(
                    f"✅ {stratum} 插值完成 | 最优模型={interp_result.get('best_model')} "
                    f"| 方法={interp_result.get('selected_method')} "
                    f"| RMSE={interp_result.get('cv_results', {}).get(interp_result.get('best_model', ''), {}).get('KRMSE', 'N/A')}"
                )

            except Exception as e:
                await ctx.error(f"❌ {stratum} 插值失败: {e}")
                kriging_results[stratum] = {"error": str(e)}

        # ---- 将结果和参数写入本地和全局上下文 ----
        context["kriging_result"] = kriging_results
        mcp_server.context.results["kriging"] = kriging_results
        
        # 提取第一个有效结果的参数用于回写
        first_valid_result = next((res for res in kriging_results.values() if "error" not in res), None)
        
        final_kriging_params = {
            "method": first_valid_result.get("selected_method") if first_valid_result else None,
            "variogram_model": first_valid_result.get("best_model") if first_valid_result else None,
            "drift": drift, # 记录 drift 等其他重要参数
        }
        
        # 更新全局参数，只更新非 None 的值
        mcp_server.context.params.update({k: v for k, v in final_kriging_params.items() if v is not None})

        await ctx.info("✅ 插值结果和最终使用参数已写入 MCPContext")
        
        return context


# ------------------------
# MapRender Agent
# ------------------------
class MapRenderAgent(Agent):
    async def run(self, ctx: ExtendedContext, context: dict) -> dict:

        # await ctx.info(f"🎯 绘图前 MCPContext params: {mcp_server.context.params}")

        kriging_results = context.get("kriging_result") or mcp_server.context.results
        if not kriging_results:
            await ctx.info("⚠️ 无插值结果，跳过 MapRenderAgent")
            return context

        # 查找第一个有效的插值结果（包含grid_x, grid_y, z）
        first_result = None
        for result in kriging_results.values():
            if isinstance(result, dict) and all(k in result for k in ["grid_x", "grid_y", "z"]):
                first_result = result
                break
        
        if not first_result:
            await ctx.error("❌ 无有效的插值结果（缺少 grid_x, grid_y, z）")
            context.setdefault("errors", []).append("无有效的插值结果")
            return context

        try:
            params = mcp_server.context.params
            res = await ctx.call_tool(
                "render_map_tool",
                grid_x=first_result["grid_x"],
                grid_y=first_result["grid_y"],
                z=first_result["z"],
                points=context.get("data_points", {}).get("rows", []),
                variable=context.get("task", {}).get("variable"),
                colormap=params.get("colormap", "RdYlBu"),
                n_classes=params.get("n_classes"),
                smooth_sigma=params.get("smooth_sigma", 0),
                lighten=params.get("lighten", False)
            )
            # ---- 将结果和参数写入本地和全局上下文 ----
            image_results = {"map": res.get("image_base64")}
            geojson_results = {"map": res.get("geojson")}
            
            context["image_results"] = image_results
            context["geojson_results"] = geojson_results
            mcp_server.context.results["image"] = image_results
            mcp_server.context.results["geojson"] = geojson_results

            # ---- 回写最终使用的渲染参数到全局上下文 ----
            final_render_params = {
                "colormap": params.get("colormap", "RdYlBu"),
                "n_classes": params.get("n_classes"),
                "smooth_sigma": params.get("smooth_sigma", 0),
                "lighten": params.get("lighten", False)
            }
            mcp_server.context.params.update(final_render_params)

            await ctx.info("✅ 渲染结果和最终使用参数已写入 MCPContext")
        except Exception as e:
            await ctx.error(f"❌ 地图渲染失败: {e}")
            context.setdefault("errors", []).append(str(e))
            mcp_server.context.add_error(str(e))

        return context
