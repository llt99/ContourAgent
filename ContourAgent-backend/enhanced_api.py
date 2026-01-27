"""
增强的API接口
集成增强的自然语言解析和多Agent协作框架
"""
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import traceback
import asyncio

from fastapi.middleware.cors import CORSMiddleware

# 导入增强的模块
from enhanced_nlp_processor import (
    EnhancedNLPProcessor, 
    FeedbackProcessor,
    parse_text_enhanced,
    parse_feedback_enhanced
)
from enhanced_agent_framework import (
    setup_agents,
    execute_user_task,
    get_execution_history,
    orchestrator
)

# 导入现有的工具函数（用于兼容）
from mcp_tool import text_to_sql_query_tool, kriging_interpolate, render_map_tool


# ------------------------
# FastAPI 初始化
# ------------------------
app = FastAPI(title="ContourAgent Enhanced API", version="2.0.0")

# ------------------------
# CORS 配置
# ------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# 全局处理器实例
# ------------------------
enhanced_nlp = EnhancedNLPProcessor()
feedback_processor = FeedbackProcessor()


# ------------------------
# 请求模型
# ------------------------
class TaskRequest(BaseModel):
    text: str | None = None
    feedback: str | None = None
    context: Dict[str, Any] | None = None  # 新增：支持传递上下文


class EnhancedTaskRequest(BaseModel):
    """增强的任务请求模型"""
    text: str
    context: Optional[Dict[str, Any]] = None
    use_enhanced_nlp: bool = True  # 是否使用增强NLP
    use_agent_framework: bool = True  # 是否使用Agent框架


class FeedbackRequest(BaseModel):
    """反馈请求模型"""
    feedback_text: str
    context: Dict[str, Any]


# ------------------------
# 工具函数
# ------------------------
def convert_to_json_serializable(obj):
    """JSON可序列化转换"""
    if isinstance(obj, list):
        return [convert_to_json_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        import math
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj


# ------------------------
# 增强的NLP解析接口
# ------------------------
@app.post("/enhanced/parse")
async def enhanced_parse(request: EnhancedTaskRequest):
    """
    增强的自然语言解析接口
    支持更智能的意图识别和参数提取
    """
    try:
        if request.use_enhanced_nlp:
            # 使用增强的NLP处理器
            result = enhanced_nlp.parse_text(request.text, request.context)
        else:
            # 兼容原有的NLP处理器
            from nlp_processor import parse_text_tool
            result = parse_text_tool(request.text)
        
        return JSONResponse({
            "success": True,
            "result": result,
            "processor": "enhanced" if request.use_enhanced_nlp else "original"
        })
    
    except Exception as e:
        traceback_str = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "trace": traceback_str
            }
        )


# ------------------------
# 增强的反馈解析接口
# ------------------------
@app.post("/enhanced/feedback")
async def enhanced_feedback(request: FeedbackRequest):
    """
    增强的反馈解析接口
    支持更精确的参数修改识别
    """
    try:
        result = feedback_processor.parse_feedback(
            request.feedback_text, 
            request.context
        )
        
        return JSONResponse({
            "success": True,
            "result": result
        })
    
    except Exception as e:
        traceback_str = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "trace": traceback_str
            }
        )


# ------------------------
# Agent框架执行接口
# ------------------------
@app.post("/agent/execute")
async def agent_execute(request: EnhancedTaskRequest):
    """
    使用Agent框架执行任务
    支持多Agent协作和智能任务调度
    """
    try:
        # 确保Agent已设置
        if not orchestrator.agents:
            setup_agents(
                query_func=text_to_sql_query_tool,
                kriging_func=kriging_interpolate,
                render_func=render_map_tool
            )
        
        # 执行任务
        result = await execute_user_task(request.text, request.context)
        
        # 格式化返回结果
        response = {
            "success": result.get("success", False),
            "pipeline": result.get("pipeline", []),
            "agent_results": result.get("agent_results", []),
            "total_time": result.get("total_time", 0),
            "errors": result.get("errors", []),
            "warnings": result.get("warnings", []),
            "final_context": convert_to_json_serializable(result.get("final_context", {}))
        }
        
        # 提取传统格式的结果（兼容前端）
        final_context = result.get("final_context", {})
        if final_context:
            response.update({
                "nlpResult": final_context.get("parsed_intent"),
                "dataResult": final_context.get("data_points"),
                "krigingResult": final_context.get("kriging_results"),
                "imageResult": final_context.get("render_results"),
                "geojsonResult": final_context.get("render_results", {}).get("geojson")
            })
        
        return JSONResponse(response)
    
    except Exception as e:
        traceback_str = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "trace": traceback_str
            }
        )


# ------------------------
# 混合执行接口（兼容原有流程）
# ------------------------
@app.post("/task")
async def run_task(req: TaskRequest):
    """
    混合执行接口
    - 如果有feedback，使用反馈处理流程
    - 如果有text，使用增强的NLP解析
    - 兼容原有的API格式
    """
    try:
        # 确保Agent已设置
        if not orchestrator.agents:
            setup_agents(
                query_func=text_to_sql_query_tool,
                kriging_func=kriging_interpolate,
                render_func=render_map_tool
            )
        
        # 判断是新任务还是反馈
        if req.feedback:
            # 反馈模式：使用Agent框架
            # 需要上下文支持
            if not req.context:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "反馈模式需要提供上下文(context)"
                    }
                )
            
            # 使用Agent框架处理反馈
            result = await execute_user_task(req.feedback, req.context)
            
        elif req.text:
            # 新任务模式：使用增强NLP + Agent框架
            result = await execute_user_task(req.text, req.context)
            
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "需要提供text或feedback"
                }
            )
        
        # 格式化返回结果
        response = {
            "nlpResult": result.get("final_context", {}).get("parsed_intent"),
            "plan": result.get("final_context", {}).get("parsed_intent", {}).get("plan"),
            "dataResult": result.get("final_context", {}).get("data_points"),
            "krigingResult": result.get("final_context", {}).get("kriging_results"),
            "imageResult": result.get("final_context", {}).get("render_results"),
            "geojsonResult": result.get("final_context", {}).get("render_results", {}).get("geojson"),
            "feedbackParsed": result.get("final_context", {}).get("feedback_parsed"),
            "history": get_execution_history(5),  # 返回最近5条历史
            "agent_results": result.get("agent_results", []),
            "execution_summary": {
                "success": result.get("success", False),
                "total_time": result.get("total_time", 0),
                "errors": result.get("errors", []),
                "warnings": result.get("warnings", [])
            }
        }
        
        return JSONResponse(response)
    
    except Exception as e:
        traceback_str = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback_str
            }
        )


# ------------------------
# 原有兼容接口（保持不变）
# ------------------------
@app.get("/history")
async def get_history(limit: int = Query(20, ge=1)):
    """
    获取执行历史记录
    """
    try:
        history = get_execution_history(limit)
        return JSONResponse({"history": history})
    except Exception as e:
        traceback_str = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback_str}
        )


# ------------------------
# 系统状态接口
# ------------------------
@app.get("/system/status")
async def system_status():
    """
    获取系统状态和Agent信息
    """
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "agent_framework": {
                "registered_agents": [agent_type.value for agent_type in orchestrator.agents.keys()],
                "total_executions": len(orchestrator.execution_history),
                "history_size": len(orchestrator.execution_history)
            },
            "nlp_processor": {
                "type": "enhanced",
                "confidence_threshold": enhanced_nlp.confidence_threshold,
                "entities_count": {
                    "regions": len(enhanced_nlp.REGION_KEYWORDS),
                    "stratums": len(enhanced_nlp.STRATUM_KEYWORDS),
                    "variables": len(enhanced_nlp.VARIABLE_KEYWORDS),
                    "plot_types": len(enhanced_nlp.PLOT_TYPE_KEYWORDS)
                }
            }
        }
        return JSONResponse(status)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ------------------------
# 测试接口
# ------------------------
@app.post("/test/enhanced_nlp")
async def test_enhanced_nlp(text: str):
    """
    测试增强的NLP解析器
    """
    try:
        result = enhanced_nlp.parse_text(text)
        return JSONResponse({
            "input": text,
            "result": result
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/test/feedback")
async def test_feedback(feedback_text: str, context: Dict[str, Any]):
    """
    测试反馈处理器
    """
    try:
        result = feedback_processor.parse_feedback(feedback_text, context)
        return JSONResponse({
            "feedback": feedback_text,
            "context": context,
            "result": result
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ------------------------
# 启动时初始化
# ------------------------
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化Agent框架"""
    try:
        # 设置Agent
        setup_agents(
            query_func=text_to_sql_query_tool,
            kriging_func=kriging_interpolate,
            render_func=render_map_tool
        )
        
    except Exception as e:
        pass


# ------------------------
# 启动入口
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("enhanced_api:app", host="127.0.0.1", port=8000, reload=True)
