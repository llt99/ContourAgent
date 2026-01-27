"""
增强的多Agent协作框架
提供更智能的Agent协作机制和任务调度
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from enhanced_nlp_processor import (
    EnhancedNLPProcessor, 
    FeedbackProcessor, 
    TaskType,
    parse_text_enhanced,
    parse_feedback_enhanced
)


class AgentStatus(Enum):
    """Agent状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentType(Enum):
    """Agent类型枚举"""
    NLP = "nlp"  # 自然语言处理
    DATA = "data"  # 数据查询
    KRIGING = "kriging"  # 插值计算
    RENDER = "render"  # 地图渲染
    FEEDBACK = "feedback"  # 反馈处理
    ANALYSIS = "analysis"  # 统计分析


@dataclass
class AgentContext:
    """Agent上下文，用于在Agent间传递数据"""
    task_id: str
    user_input: str
    parsed_intent: Dict[str, Any] = field(default_factory=dict)
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    kriging_results: Dict[str, Any] = field(default_factory=dict)
    render_results: Dict[str, Any] = field(default_factory=dict)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    feedback_parsed: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "user_input": self.user_input,
            "parsed_intent": self.parsed_intent,
            "data_points": self.data_points,
            "kriging_results": self.kriging_results,
            "render_results": self.render_results,
            "analysis_results": self.analysis_results,
            "feedback_parsed": self.feedback_parsed,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata
        }


@dataclass
class AgentResult:
    """Agent执行结果"""
    success: bool
    data: Dict[str, Any]
    status: AgentStatus
    execution_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class BaseAgent:
    """Agent基类"""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"Agent.{agent_type.value}")
        
    async def execute(self, context: AgentContext) -> AgentResult:
        """执行Agent任务"""
        raise NotImplementedError("子类必须实现execute方法")
    
    def log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(f"[{self.agent_type.value}] {message}")
    
    def log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(f"[{self.agent_type.value}] {message}")
    
    def log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(f"[{self.agent_type.value}] {message}")


class NLPAgent(BaseAgent):
    """自然语言处理Agent"""
    
    def __init__(self):
        super().__init__(AgentType.NLP)
        self.processor = EnhancedNLPProcessor()
    
    async def execute(self, context: AgentContext) -> AgentResult:
        self.status = AgentStatus.RUNNING
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.log_info(f"开始解析用户输入: {context.user_input}")
            
            # 使用增强的NLP处理器
            result = self.processor.parse_text(context.user_input, context.to_dict())
            
            # 检查解析结果
            task = result.get("task", {})
            if task.get("errors"):
                context.errors.extend(task["errors"])
                self.status = AgentStatus.FAILED
                return AgentResult(
                    success=False,
                    data=result,
                    status=self.status,
                    errors=task["errors"],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            
            # 更新上下文
            context.parsed_intent = result
            if task.get("warnings"):
                context.warnings.extend(task["warnings"])
            
            self.log_info(f"解析完成，置信度: {task.get('confidence', 0):.2f}")
            self.status = AgentStatus.COMPLETED
            
            return AgentResult(
                success=True,
                data=result,
                status=self.status,
                warnings=task.get("warnings", []),
                execution_time=asyncio.get_event_loop().time() - start_time
            )
            
        except Exception as e:
            self.log_error(f"解析失败: {str(e)}")
            self.status = AgentStatus.FAILED
            context.errors.append(f"NLP解析失败: {str(e)}")
            
            return AgentResult(
                success=False,
                data={},
                status=self.status,
                errors=[str(e)],
                execution_time=asyncio.get_event_loop().time() - start_time
            )


class DataAgent(BaseAgent):
    """数据查询Agent"""
    
    def __init__(self, query_func: Callable):
        super().__init__(AgentType.DATA)
        self.query_func = query_func
    
    async def execute(self, context: AgentContext) -> AgentResult:
        self.status = AgentStatus.RUNNING
        start_time = asyncio.get_event_loop().time()
        
        try:
            task = context.parsed_intent.get("task", {})
            region = task.get("region", "")
            stratum = task.get("stratum", "")
            variable = task.get("variable", "")
            
            if not all([region, stratum, variable]):
                missing = []
                if not region: missing.append("区域")
                if not stratum: missing.append("地层")
                if not variable: missing.append("变量")
                
                error_msg = f"缺少必要参数: {', '.join(missing)}"
                self.log_error(error_msg)
                self.status = AgentStatus.FAILED
                context.errors.append(error_msg)
                
                return AgentResult(
                    success=False,
                    data={},
                    status=self.status,
                    errors=[error_msg],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            
            # 构造查询文本
            query_text = f"查询 {stratum} 各井 {variable} 数据"
            self.log_info(f"执行数据查询: {query_text}")
            
            # 调用查询函数
            query_result = await self.query_func(query_text)
            
            if not query_result or not query_result.get("rows"):
                error_msg = "未获取到有效井点数据"
                self.log_error(error_msg)
                self.status = AgentStatus.FAILED
                context.errors.append(error_msg)
                
                return AgentResult(
                    success=False,
                    data={},
                    status=self.status,
                    errors=[error_msg],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            
            # 更新上下文
            context.data_points = query_result["rows"]
            
            self.log_info(f"成功获取 {len(query_result['rows'])} 个数据点")
            self.status = AgentStatus.COMPLETED
            
            return AgentResult(
                success=True,
                data=query_result,
                status=self.status,
                execution_time=asyncio.get_event_loop().time() - start_time
            )
            
        except Exception as e:
            self.log_error(f"数据查询失败: {str(e)}")
            self.status = AgentStatus.FAILED
            context.errors.append(f"数据查询失败: {str(e)}")
            
            return AgentResult(
                success=False,
                data={},
                status=self.status,
                errors=[str(e)],
                execution_time=asyncio.get_event_loop().time() - start_time
            )


class KrigingAgent(BaseAgent):
    """克里金插值Agent"""
    
    def __init__(self, kriging_func: Callable):
        super().__init__(AgentType.KRIGING)
        self.kriging_func = kriging_func
    
    async def execute(self, context: AgentContext) -> AgentResult:
        self.status = AgentStatus.RUNNING
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not context.data_points:
                error_msg = "没有可用的数据点进行插值"
                self.log_error(error_msg)
                self.status = AgentStatus.FAILED
                context.errors.append(error_msg)
                
                return AgentResult(
                    success=False,
                    data={},
                    status=self.status,
                    errors=[error_msg],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            
            # 提取插值参数
            task = context.parsed_intent.get("task", {})
            method = task.get("method_code") or task.get("method") or "auto"
            model = task.get("model_code") or task.get("model") or "auto"
            
            self.log_info(f"开始克里金插值，方法: {method}, 模型: {model}")
            
            # 调用插值函数
            result = await self.kriging_func(
                points=context.data_points,
                method=method,
                candidate_models=[model] if model != "auto" else None,
                autoOptimizeModel=(model == "auto")
            )
            
            if "error" in result:
                error_msg = result["error"]
                self.log_error(error_msg)
                self.status = AgentStatus.FAILED
                context.errors.append(error_msg)
                
                return AgentResult(
                    success=False,
                    data=result,
                    status=self.status,
                    errors=[error_msg],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            
            # 更新上下文
            context.kriging_results = result
            
            self.log_info(f"插值完成，最优模型: {result.get('best_model', 'N/A')}")
            self.status = AgentStatus.COMPLETED
            
            return AgentResult(
                success=True,
                data=result,
                status=self.status,
                execution_time=asyncio.get_event_loop().time() - start_time
            )
            
        except Exception as e:
            self.log_error(f"插值失败: {str(e)}")
            self.status = AgentStatus.FAILED
            context.errors.append(f"插值失败: {str(e)}")
            
            return AgentResult(
                success=False,
                data={},
                status=self.status,
                errors=[str(e)],
                execution_time=asyncio.get_event_loop().time() - start_time
            )


class RenderAgent(BaseAgent):
    """地图渲染Agent"""
    
    def __init__(self, render_func: Callable):
        super().__init__(AgentType.RENDER)
        self.render_func = render_func
    
    async def execute(self, context: AgentContext) -> AgentResult:
        self.status = AgentStatus.RUNNING
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not context.kriging_results:
                error_msg = "没有可用的插值结果进行渲染"
                self.log_error(error_msg)
                self.status = AgentStatus.FAILED
                context.errors.append(error_msg)
                
                return AgentResult(
                    success=False,
                    data={},
                    status=self.status,
                    errors=[error_msg],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            
            # 提取渲染参数
            task = context.parsed_intent.get("task", {})
            first_result = next(iter(context.kriging_results.values()))
            
            # 获取渲染参数，支持从反馈中继承
            render_params = context.metadata.get("render_params", {})
            colormap = render_params.get("colormap", "RdYlBu")
            n_classes = render_params.get("n_classes", 11)
            smooth_sigma = render_params.get("smooth_sigma", 0)
            lighten = render_params.get("lighten", False)
            
            self.log_info(f"开始渲染地图，色带: {colormap}, 分级: {n_classes}")
            
            # 调用渲染函数
            result = await self.render_func(
                grid_x=first_result["grid_x"],
                grid_y=first_result["grid_y"],
                z=first_result["z"],
                points=context.data_points,
                variable=task.get("variable", "thickness"),
                colormap=colormap,
                n_classes=n_classes,
                smooth_sigma=smooth_sigma,
                lighten=lighten
            )
            
            # 更新上下文
            context.render_results = {
                "image": result.get("image_base64"),
                "geojson": result.get("geojson")
            }
            
            self.log_info("地图渲染完成")
            self.status = AgentStatus.COMPLETED
            
            return AgentResult(
                success=True,
                data=result,
                status=self.status,
                execution_time=asyncio.get_event_loop().time() - start_time
            )
            
        except Exception as e:
            self.log_error(f"渲染失败: {str(e)}")
            self.status = AgentStatus.FAILED
            context.errors.append(f"渲染失败: {str(e)}")
            
            return AgentResult(
                success=False,
                data={},
                status=self.status,
                errors=[str(e)],
                execution_time=asyncio.get_event_loop().time() - start_time
            )


class FeedbackAgent(BaseAgent):
    """反馈处理Agent"""
    
    def __init__(self):
        super().__init__(AgentType.FEEDBACK)
        self.processor = FeedbackProcessor()
    
    async def execute(self, context: AgentContext) -> AgentResult:
        self.status = AgentStatus.RUNNING
        start_time = asyncio.get_event_loop().time()
        
        try:
            feedback_text = context.user_input
            self.log_info(f"处理用户反馈: {feedback_text}")
            
            # 解析反馈
            result = self.processor.parse_feedback(feedback_text, context.to_dict())
            
            # 更新上下文
            context.feedback_parsed = result
            
            # 提取修改的参数
            params = result.get("mcp_context", {}).get("params", {})
            if params:
                # 将参数存储到metadata中，供后续Agent使用
                context.metadata["render_params"] = params
                self.log_info(f"反馈解析完成，修改参数: {list(params.keys())}")
            else:
                self.log_warning("反馈解析完成，但未识别出具体修改")
            
            self.status = AgentStatus.COMPLETED
            
            return AgentResult(
                success=True,
                data=result,
                status=self.status,
                execution_time=asyncio.get_event_loop().time() - start_time
            )
            
        except Exception as e:
            self.log_error(f"反馈处理失败: {str(e)}")
            self.status = AgentStatus.FAILED
            context.errors.append(f"反馈处理失败: {str(e)}")
            
            return AgentResult(
                success=False,
                data={},
                status=self.status,
                errors=[str(e)],
                execution_time=asyncio.get_event_loop().time() - start_time
            )


class AgentOrchestrator:
    """Agent编排器，负责协调多个Agent的协作"""
    
    def __init__(self):
        self.agents: Dict[AgentType, BaseAgent] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("AgentOrchestrator")
    
    def register_agent(self, agent: BaseAgent):
        """注册Agent"""
        self.agents[agent.agent_type] = agent
        self.logger.info(f"注册Agent: {agent.agent_type.value}")
    
    def get_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """获取Agent"""
        return self.agents.get(agent_type)
    
    async def execute_pipeline(self, pipeline: List[str], context: AgentContext) -> Dict[str, Any]:
        """执行Agent管道"""
        self.logger.info(f"开始执行管道: {' -> '.join(pipeline)}")
        
        results = {
            "success": True,
            "pipeline": pipeline,
            "agent_results": [],
            "final_context": None,
            "total_time": 0.0,
            "errors": [],
            "warnings": []
        }
        
        start_time = asyncio.get_event_loop().time()
        
        for agent_name in pipeline:
            try:
                # 映射Agent名称到类型
                agent_type_map = {
                    "nlp": AgentType.NLP,
                    "data": AgentType.DATA,
                    "kriging": AgentType.KRIGING,
                    "image": AgentType.RENDER,
                    "feedback": AgentType.FEEDBACK
                }
                
                agent_type = agent_type_map.get(agent_name)
                if not agent_type:
                    self.logger.warning(f"未知的Agent名称: {agent_name}")
                    continue
                
                agent = self.get_agent(agent_type)
                if not agent:
                    self.logger.warning(f"Agent未注册: {agent_type.value}")
                    continue
                
                # 执行Agent
                self.logger.info(f"执行Agent: {agent_name}")
                result = await agent.execute(context)
                
                # 记录结果
                agent_result = {
                    "agent": agent_name,
                    "status": result.status.value,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "errors": result.errors,
                    "warnings": result.warnings
                }
                results["agent_results"].append(agent_result)
                
                # 收集错误和警告
                results["errors"].extend(result.errors)
                results["warnings"].extend(result.warnings)
                
                # 如果Agent失败，根据策略决定是否继续
                if not result.success:
                    self.logger.error(f"Agent {agent_name} 执行失败: {result.errors}")
                    results["success"] = False
                    # 可以在这里添加失败处理策略
                    break
                
                self.logger.info(f"Agent {agent_name} 执行成功")
                
            except Exception as e:
                self.logger.error(f"执行Agent {agent_name} 时发生异常: {str(e)}")
                results["success"] = False
                results["errors"].append(f"执行Agent {agent_name} 时发生异常: {str(e)}")
                break
        
        # 计算总时间
        results["total_time"] = asyncio.get_event_loop().time() - start_time
        
        # 保存最终上下文
        results["final_context"] = context.to_dict()
        
        # 记录执行历史
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "pipeline": pipeline,
            "results": results
        })
        
        # 限制历史长度
        if len(self.execution_history) > 50:
            self.execution_history.pop(0)
        
        self.logger.info(f"管道执行完成，总耗时: {results['total_time']:.2f}s")
        
        return results
    
    async def execute_task(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行完整任务"""
        # 创建上下文
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        agent_context = AgentContext(
            task_id=task_id,
            user_input=user_input
        )
        
        # 判断任务类型
        is_feedback = self._is_feedback_task(user_input)
        
        if is_feedback:
            # 反馈任务：先执行NLP获取上下文，然后执行反馈
            if context:
                # 使用提供的上下文
                agent_context.parsed_intent = context
                pipeline = ["feedback", "kriging", "image"]
            else:
                # 需要先解析
                nlp_result = await self.get_agent(AgentType.NLP).execute(agent_context)
                if not nlp_result.success:
                    return {
                        "success": False,
                        "errors": nlp_result.errors,
                        "pipeline": ["nlp"]
                    }
                pipeline = agent_context.parsed_intent.get("plan", {}).get("pipeline", [])
        else:
            # 新任务：执行完整的NLP解析
            nlp_agent = self.get_agent(AgentType.NLP)
            if not nlp_agent:
                return {"success": False, "errors": ["NLP Agent未注册"]}
            
            nlp_result = await nlp_agent.execute(agent_context)
            if not nlp_result.success:
                return {
                    "success": False,
                    "errors": nlp_result.errors,
                    "pipeline": ["nlp"]
                }
            
            pipeline = agent_context.parsed_intent.get("plan", {}).get("pipeline", [])
        
        # 执行管道
        if not pipeline:
            return {
                "success": False,
                "errors": ["无法生成执行管道"],
                "pipeline": []
            }
        
        return await self.execute_pipeline(pipeline, agent_context)
    
    def _is_feedback_task(self, text: str) -> bool:
        """判断是否为反馈任务"""
        feedback_keywords = ["修改", "更改", "换成", "使用", "渲染", "颜色", "方法", "模型", "色带", "调整", "优化"]
        text_lower = text.lower()
        
        # 如果包含反馈关键词且不包含新的实体描述，可能是反馈
        has_feedback_keyword = any(keyword in text_lower for keyword in feedback_keywords)
        
        # 简单的启发式：如果文本较短且包含反馈关键词，认为是反馈
        if has_feedback_keyword and len(text.split()) < 10:
            return True
        
        return False
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history[-limit:]


# 全局实例和便捷函数
orchestrator = AgentOrchestrator()


def setup_agents(
    query_func: Callable,
    kriging_func: Callable,
    render_func: Callable
):
    """设置所有Agent"""
    orchestrator.register_agent(NLPAgent())
    orchestrator.register_agent(DataAgent(query_func))
    orchestrator.register_agent(KrigingAgent(kriging_func))
    orchestrator.register_agent(RenderAgent(render_func))
    orchestrator.register_agent(FeedbackAgent())


async def execute_user_task(
    user_input: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """执行用户任务的便捷函数"""
    return await orchestrator.execute_task(user_input, context)


def get_execution_history(limit: int = 10) -> List[Dict[str, Any]]:
    """获取执行历史的便捷函数"""
    return orchestrator.get_execution_history(limit)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 测试示例
    async def test():
        # 模拟函数
        async def mock_query(query):
            return {"rows": [{"lon": 100, "lat": 30, "value": 100}]}
        
        async def mock_kriging(points, **kwargs):
            return {
                "grid_x": [100, 101],
                "grid_y": [30, 31],
                "z": [[100, 101], [102, 103]],
                "best_model": "spherical",
                "selected_method": "ok"
            }
        
        async def mock_render(**kwargs):
            return {"image_base64": "mock_image", "geojson": {"type": "FeatureCollection"}}
        
        # 设置Agent
        setup_agents(mock_query, mock_kriging, mock_render)
        
        # 测试任务
        result = await execute_user_task("绘制四川盆地龙潭组灰岩等值线图")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # asyncio.run(test())
