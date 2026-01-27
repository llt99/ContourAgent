from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from decimal import Decimal
import math
import traceback
from fastapi.middleware.cors import CORSMiddleware

# ------------------------
# MCP 引入
# ------------------------
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp_tool import *
from agent import NLPAgent, DataAgent, KrigingAgent, MapRenderAgent, FeedbackAgent, ExtendedContext
from context_schema import MCPContextSchema # 导入 Schema 用于重置

# ------------------------
# FastAPI 初始化
# ------------------------
app = FastAPI()

# ------------------------
# CORS 配置
# ------------------------
# 允许所有来源的跨域请求（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

mcp_server = FastMCP(name="Interpolation Pipeline")

# ------------------------
# MCP Controller + 调度
# ------------------------
class MCPController:
    def __init__(self):
        self.last_context = {}
        self.agents = {
            "nlp": NLPAgent(),
            "data": DataAgent(),
            "kriging": KrigingAgent(),
            "image": MapRenderAgent(),
            "feedback": FeedbackAgent(),
        }
        self.history = []  # 历史记录列表

    async def run_pipeline(self, context, ctx):
        extended_ctx = ExtendedContext.from_context(ctx)

        # --- 步骤 1: 判断运行模式 (新任务 vs 反馈) ---
        is_feedback_run = context.get("is_feedback_run", False)

        if is_feedback_run:
            # 反馈模式: 先运行 FeedbackAgent
            context = await self.agents["feedback"].run(extended_ctx, context)
            feedback_params = context.get("feedbackParsed", {})

            # 决定重启点
            # 如果修改了插值相关参数，则从 kriging 开始
            if any(k in feedback_params for k in ["method", "variogram_model"]):
                start_agent = "kriging"
            # 否则，只修改了渲染参数，从 image 开始
            else:
                start_agent = "image"
            
            # 继承上一次的 plan，并截取
            original_pipeline = self.last_context.get("plan", {}).get("pipeline", [])
            try:
                start_index = original_pipeline.index(start_agent)
                context["plan"]["pipeline"] = original_pipeline[start_index:]
            except ValueError:
                context["plan"]["pipeline"] = ["kriging", "image"] # 容错

        else:
            # 新任务模式: 运行 NLPAgent
            context = await self.agents["nlp"].run(extended_ctx, context)

        # --- 步骤 2: 动态执行 Pipeline ---
        pipeline = context.get("plan", {}).get("pipeline", [])
        if not pipeline:
            await extended_ctx.error("未能生成或继承有效的执行计划 (pipeline)")
            context.setdefault("errors", []).append("未能生成执行计划")
            return context

        await extended_ctx.info(f"动态执行计划: {' -> '.join(pipeline)}")

        for agent_name in pipeline:
            # 在反馈模式下，nlp agent 不应再执行
            if is_feedback_run and agent_name == "nlp":
                continue
            
            agent = self.agents.get(agent_name)
            if agent:
                context = await agent.run(extended_ctx, context)
            else:
                await extended_ctx.error(f"未找到名为 '{agent_name}' 的 Agent")

        self.last_context = context

        # 保存历史记录（包括参数）
        history_entry = {
            "text": context.get("text"),
            "feedback": context.get("feedback"),
            "params": {
                "kriging": context.get("kriging_params", {}),
                "render": context.get("render_params", {}),
            },
            "dataResult": convert_to_json_serializable(context.get("data_points")),
            "krigingResult": convert_to_json_serializable(context.get("kriging_result")),
            "imageResult": convert_to_json_serializable(context.get("image_results")),
            "geojsonResult": convert_to_json_serializable(context.get("geojson_results")),
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(history_entry)

        # 限制历史长度，防止内存过大
        if len(self.history) > 20:
            self.history.pop(0)

        return context


mcp_controller = MCPController()

# ------------------------
# 请求模型
# ------------------------
class TaskRequest(BaseModel):
    text: str | None = None
    feedback: str | None = None  # 用户反馈是字符串
    excelData: list | None = None  # Excel数据，直接传入

# ------------------------
# 工具函数：JSON 可序列化转换（处理 NaN / Inf / Decimal / np.ndarray）
# ------------------------
def convert_to_json_serializable(obj):
    if isinstance(obj, list):
        return [convert_to_json_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return convert_to_json_serializable(obj.tolist())
    else:
        return obj

# ------------------------
# MCP 任务执行
# ------------------------
async def run_task_mcp(text: str | None = None, feedback: str | None = None, excel_data: list | None = None) -> dict:
    ctx = Context[ServerSession, None](session=None)

    # --- 启发式规则：检测可能错发到 'text' 字段的反馈指令 ---
    if not feedback and text:
        feedback_keywords = ["修改", "更改", "换成", "使用", "渲染", "颜色", "方法", "模型", "色带", "克里金", "高斯", "球状", "指数", "红黄绿"]
        # 假设反馈指令通常较短，且包含关键词
        if any(keyword in text for keyword in feedback_keywords) and len(text.split()) < 15:
            feedback = text  # 将 text 内容视为 feedback
            text = None      # 清空 text，强制进入反馈模式

    # --- 判断是新任务还是反馈 ---
    if feedback:
        # 反馈模式
        if not mcp_controller.last_context:
            return {"error": "No previous task context available to apply feedback."}
        
        # 继承上一次的上下文，并加入新的反馈
        context = mcp_controller.last_context.copy()
        context["feedback"] = feedback
        context["is_feedback_run"] = True
        
    else:
        # 新任务模式
        print("✨ New task detected, applying soft reset to MCPContext.")
        
        # 软重置：保留渲染参数，清空其他所有内容
        render_params_to_keep = {}
        if hasattr(mcp_server, 'context'):
            last_params = mcp_server.context.params or {}
            render_params_to_keep = {
                "colormap": last_params.get("colormap"),
                "n_classes": last_params.get("n_classes"),
                "smooth_sigma": last_params.get("smooth_sigma"),
                "lighten": last_params.get("lighten"),
            }
            # 过滤掉值为 None 的参数
            render_params_to_keep = {k: v for k, v in render_params_to_keep.items() if v is not None}

        # 执行重置
        mcp_server.context = MCPContextSchema()
        
        # 重新应用保留的渲染参数
        if render_params_to_keep:
            mcp_server.context.params.update(render_params_to_keep)
            print(f"🎨 Kept render params: {render_params_to_keep}")

        context = {
            "text": text,
            "feedback": None,
            "is_feedback_run": False,
            "excel_data": excel_data  # 传递Excel数据
        }
        
        # 如果有Excel数据，确保任务参数被正确设置
        if excel_data:
            # 即使没有text，也需要设置基本的任务信息
            if not text:
                context["text"] = "绘制四川盆地龙潭组煤岩分布图"  # 默认任务描述
    
    return await mcp_controller.run_pipeline(context, ctx=ctx)

# ------------------------
# FastAPI 接口
# ------------------------
@app.post("/task")
async def run_task(req: TaskRequest):
    try:
        result_context = await run_task_mcp(req.text, req.feedback, req.excelData)

        response_content = {
            "nlpResult": result_context.get("task"),
            "plan": result_context.get("plan"),
            "dataResult": convert_to_json_serializable(result_context.get("data_points")),
            "krigingResult": convert_to_json_serializable(result_context.get("kriging_result")),
            "imageResult": convert_to_json_serializable(result_context.get("image_results")),
            "geojsonResult": convert_to_json_serializable(result_context.get("geojson_results")),
            "feedbackParsed": result_context.get("feedbackParsed"),
            "history": mcp_controller.history  # 返回历史记录，包括参数和结果
        }

        return JSONResponse(response_content)
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback_str}
        )


# ------------------------
# FastAPI 接口：获取历史记录
# ------------------------
@app.get("/history")
async def get_history(limit: int = Query(20, ge=1)):
    """
    返回最近 limit 条历史记录
    """
    try:
        # 截取最近 limit 条
        history_slice = mcp_controller.history[-limit:]
        return JSONResponse({"history": history_slice})
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback_str}
        )

# ------------------------
# 启动入口
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
