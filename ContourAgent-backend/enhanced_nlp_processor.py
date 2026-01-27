"""
增强的自然语言处理处理器
提供更智能的自然语言解析和意图识别功能
"""
from __future__ import annotations
import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """任务类型枚举"""
    DATA_QUERY = "data_query"  # 数据查询
    CONTOUR_MAP = "contour_map"  # 等值线图
    DISTRIBUTION_MAP = "distribution_map"  # 分布图
    STATISTICAL_ANALYSIS = "statistical_analysis"  # 统计分析
    FEEDBACK_MODIFICATION = "feedback_modification"  # 反馈修改


class RegionType(Enum):
    """区域类型枚举"""
    BASIN = "basin"  # 盆地
    AREA = "area"  # 区域
    PROVINCE = "province"  # 省份


@dataclass
class ParsedIntent:
    """解析后的用户意图"""
    task_type: TaskType
    region: str
    stratum: str
    variable: str
    plot_type: str
    method: Optional[str] = None
    model: Optional[str] = None
    confidence: float = 1.0
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class EnhancedNLPProcessor:
    """增强的NLP处理器"""
    
    # 关键词映射表
    REGION_KEYWORDS = {
        "四川盆地": ["四川盆地", "川", "四川"],
        "塔里木盆地": ["塔里木", "塔里木盆地"],
        "鄂尔多斯盆地": ["鄂尔多斯", "鄂尔多斯盆地"],
        "松辽盆地": ["松辽", "松辽盆地"],
        "渤海湾盆地": ["渤海湾", "渤海"],
        "准噶尔盆地": ["准噶尔", "准噶尔盆地"]
    }
    
    STRATUM_KEYWORDS = {
        "龙潭组": ["龙潭", "龙潭组", "P2l", "二叠系龙潭"],
        "长兴组": ["长兴", "长兴组", "P2c"],
        "飞仙关组": ["飞仙关", "飞仙关组", "T1f"],
        "嘉陵江组": ["嘉陵江", "嘉陵江组", "T1j"],
        "雷口坡组": ["雷口坡", "雷口坡组", "T2l"],
        "须家河组": ["须家河", "须家河组", "T3x"]
    }
    
    VARIABLE_KEYWORDS = {
        "灰岩": ["灰岩", "石灰岩", "limestone", "灰"],
        "砂岩": ["砂岩", "sandstone", "砂"],
        "煤岩": ["煤岩", "煤", "coal", "煤层"],
        "泥岩": ["泥岩", "mudstone", "泥"],
        "白云岩": ["白云岩", "dolomite", "白云"],
        "厚度": ["厚度", "厚", "地层厚度", "thickness"],
        "孔隙度": ["孔隙度", "孔隙", "porosity", "孔"],
        "渗透率": ["渗透率", "渗透", "permeability", "渗"]
    }
    
    PLOT_TYPE_KEYWORDS = {
        "分布图": ["分布图", "分布", "平面图", "分布特征"],
        "等值线图": ["等值线图", "等值线", "等值图", "等值"],
        "剖面图": ["剖面图", "剖面", "横截面"],
        "三维图": ["三维图", "3D图", "立体图"]
    }
    
    METHOD_KEYWORDS = {
        "普通克里金": ["普通克里金", "普通", "ok", "ordinary"],
        "泛克里金": ["泛克里金", "泛", "uk", "universal"],
        "协同克里金": ["协同克里金", "协同", "ck", "co-kriging"]
    }
    
    MODEL_KEYWORDS = {
        "球状": ["球状", "spherical", "球"],
        "指数": ["指数", "exponential", "指"],
        "高斯": ["高斯", "gaussian", "高"],
        "线性": ["线性", "linear", "线"]
    }
    
    # 反馈修改关键词
    FEEDBACK_KEYWORDS = ["修改", "更改", "换成", "使用", "渲染", "颜色", "方法", "模型", "色带", "克里金", "高斯", "球状", "指数", "红黄绿", "调整", "优化"]

    def __init__(self):
        self.context_memory = {}  # 上下文记忆
        self.confidence_threshold = 0.6  # 置信度阈值

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取文本中的实体"""
        entities = {
            "regions": [],
            "stratums": [],
            "variables": [],
            "plot_types": [],
            "methods": [],
            "models": []
        }
        
        text_lower = text.lower()
        
        # 匹配区域
        for standard_name, keywords in self.REGION_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if standard_name not in entities["regions"]:
                        entities["regions"].append(standard_name)
                    break
        
        # 匹配地层
        for standard_name, keywords in self.STRATUM_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if standard_name not in entities["stratums"]:
                        entities["stratums"].append(standard_name)
                    break
        
        # 匹配变量
        for standard_name, keywords in self.VARIABLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if standard_name not in entities["variables"]:
                        entities["variables"].append(standard_name)
                    break
        
        # 匹配图件类型
        for standard_name, keywords in self.PLOT_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if standard_name not in entities["plot_types"]:
                        entities["plot_types"].append(standard_name)
                    break
        
        # 匹配方法
        for standard_name, keywords in self.METHOD_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if standard_name not in entities["methods"]:
                        entities["methods"].append(standard_name)
                    break
        
        # 匹配模型
        for standard_name, keywords in self.MODEL_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if standard_name not in entities["models"]:
                        entities["models"].append(standard_name)
                    break
        
        return entities

    def detect_task_type(self, text: str, entities: Dict[str, List[str]]) -> TaskType:
        """检测任务类型"""
        text_lower = text.lower()
        
        # 检测是否为反馈修改
        if any(keyword in text_lower for keyword in self.FEEDBACK_KEYWORDS):
            if len(entities["regions"]) == 0 and len(entities["stratums"]) == 0:
                return TaskType.FEEDBACK_MODIFICATION
        
        # 检测数据查询
        if "查询" in text_lower or "查" in text_lower or "数据" in text_lower:
            if len(entities["variables"]) > 0:
                return TaskType.DATA_QUERY
        
        # 检测统计分析
        if any(word in text_lower for word in ["统计", "分析", "平均", "最大", "最小", "求和"]):
            return TaskType.STATISTICAL_ANALYSIS
        
        # 检测图件生成
        if len(entities["plot_types"]) > 0 or any(word in text_lower for word in ["绘制", "画", "生成", "做图"]):
            if "等值线" in text_lower or "等值" in text_lower:
                return TaskType.CONTOUR_MAP
            elif "分布" in text_lower:
                return TaskType.DISTRIBUTION_MAP
            else:
                # 默认为等值线图
                return TaskType.CONTOUR_MAP
        
        # 默认为等值线图
        return TaskType.CONTOUR_MAP

    def calculate_confidence(self, entities: Dict[str, List[str]], task_type: TaskType) -> float:
        """计算解析置信度"""
        score = 0.0
        
        # 基础分数
        if entities["regions"]:
            score += 0.3
        if entities["stratums"]:
            score += 0.3
        if entities["variables"]:
            score += 0.2
        if entities["plot_types"]:
            score += 0.2
        
        # 任务类型特殊调整
        if task_type == TaskType.FEEDBACK_MODIFICATION:
            score = min(score, 0.5)  # 反馈修改不需要太多实体
        
        return min(score, 1.0)

    def resolve_conflicts(self, entities: Dict[str, List[str]]) -> Dict[str, str]:
        """解决实体冲突，选择最合适的值"""
        resolved = {}
        
        # 如果有多个区域，选择第一个（通常是最具体的）
        if entities["regions"]:
            resolved["region"] = entities["regions"][0]
        
        # 如果有多个地层，选择第一个
        if entities["stratums"]:
            resolved["stratum"] = entities["stratums"][0]
        
        # 如果有多个变量，选择第一个
        if entities["variables"]:
            resolved["variable"] = entities["variables"][0]
        
        # 如果有多个图件类型，选择第一个
        if entities["plot_types"]:
            resolved["plot_type"] = entities["plot_types"][0]
        
        # 方法和模型是可选的
        if entities["methods"]:
            resolved["method"] = entities["methods"][0]
        if entities["models"]:
            resolved["model"] = entities["models"][0]
        
        return resolved

    def generate_pipeline(self, task_type: TaskType, entities: Dict[str, List[str]]) -> List[str]:
        """生成执行管道"""
        pipeline = ["nlp"]
        
        if task_type == TaskType.FEEDBACK_MODIFICATION:
            pipeline.extend(["feedback", "kriging", "image"])
        elif task_type == TaskType.DATA_QUERY:
            pipeline.append("data")
        elif task_type == TaskType.STATISTICAL_ANALYSIS:
            pipeline.append("data")
            # 可以添加统计分析agent
        elif task_type in [TaskType.CONTOUR_MAP, TaskType.DISTRIBUTION_MAP]:
            pipeline.extend(["data", "kriging", "image"])
        
        return pipeline

    def parse_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """主解析函数"""
        warnings = []
        errors = []
        
        # 提取实体
        entities = self.extract_entities(text)
        
        # 检测任务类型
        task_type = self.detect_task_type(text, entities)
        
        # 计算置信度
        confidence = self.calculate_confidence(entities, task_type)
        
        # 解决冲突并提取关键信息
        resolved = self.resolve_conflicts(entities)
        
        # 检查必要字段
        required_fields = ["region", "stratum", "variable"]
        
        # 如果是反馈修改，不需要检查必要字段
        if task_type == TaskType.FEEDBACK_MODIFICATION:
            # 反馈修改应该继承上下文
            if context:
                resolved["region"] = context.get("region", "")
                resolved["stratum"] = context.get("stratum", "")
                resolved["variable"] = context.get("variable", "")
            else:
                errors.append("反馈修改需要上下文支持")
        
        else:
            for field in required_fields:
                if field not in resolved or not resolved[field]:
                    warnings.append(f"缺少必要参数: {field}")
        
        # 设置默认值
        if "plot_type" not in resolved:
            if task_type == TaskType.CONTOUR_MAP:
                resolved["plot_type"] = "等值线图"
            elif task_type == TaskType.DISTRIBUTION_MAP:
                resolved["plot_type"] = "分布图"
            else:
                resolved["plot_type"] = "等值线图"
        
        # 生成执行管道
        pipeline = self.generate_pipeline(task_type, entities)
        
        # 构建返回结果
        result = {
            "task": {
                "region": resolved.get("region", ""),
                "stratum": resolved.get("stratum", ""),
                "variable": resolved.get("variable", ""),
                "plot": resolved.get("plot_type", ""),
                "method": resolved.get("method", None),
                "model": resolved.get("model", None),
                "method_code": self._map_method_code(resolved.get("method", "")),
                "model_code": self._map_model_code(resolved.get("model", "")),
                "variable_code": None,
                "plot_code": None,
                "warnings": warnings,
                "errors": errors,
                "confidence": confidence,
                "task_type": task_type.value
            },
            "plan": {
                "pipeline": pipeline
            }
        }
        
        # 如果置信度太低，添加警告
        if confidence < self.confidence_threshold:
            result["task"]["warnings"].append(f"解析置信度较低 ({confidence:.2f})，结果可能不准确")
        
        return result

    def _map_method_code(self, method: str) -> Optional[str]:
        """映射方法代码"""
        if not method:
            return None
        method_map = {
            "普通克里金": "ok",
            "泛克里金": "uk",
            "协同克里金": "ck"
        }
        return method_map.get(method, None)

    def _map_model_code(self, model: str) -> Optional[str]:
        """映射模型代码"""
        if not model:
            return None
        model_map = {
            "球状": "spherical",
            "指数": "exponential",
            "高斯": "gaussian",
            "线性": "linear"
        }
        return model_map.get(model, None)

    async def parse_text_async(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """异步解析函数"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse_text, text, context)


class FeedbackProcessor:
    """反馈处理器"""
    
    def __init__(self):
        self.modification_keywords = {
            "method": ["方法", "克里金", "普通", "泛", "协同", "ok", "uk", "ck"],
            "model": ["模型", "球状", "指数", "高斯", "线性", "spherical", "exponential", "gaussian"],
            "colormap": ["颜色", "色带", "colormap", "色彩", "配色"],
            "n_classes": ["分级", "类别", "等级", "classes"],
            "smooth_sigma": ["平滑", "smooth", "柔化"],
            "lighten": ["亮度", "明亮", "lighten"]
        }
        
        self.method_map = {
            "普通克里金": "ok", "普通": "ok", "ok": "ok",
            "泛克里金": "uk", "泛": "uk", "uk": "uk",
            "协同克里金": "ck", "协同": "ck", "ck": "ck"
        }
        
        self.model_map = {
            "球状": "spherical", "球": "spherical", "spherical": "spherical",
            "指数": "exponential", "指": "exponential", "exponential": "exponential",
            "高斯": "gaussian", "高": "gaussian", "gaussian": "gaussian",
            "线性": "linear", "线": "linear", "linear": "linear"
        }
        
        self.colormap_map = {
            "红黄绿": "RdYlGn",
            "红黄蓝": "RdYlBu",
            "彩虹": "rainbow",
            "热力": "hot",
            "冷暖": "coolwarm",
            "蓝绿": "GnBu",
            "紫红": "PuRd"
        }

    def parse_feedback(self, feedback_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """解析用户反馈"""
        params = {}
        warnings = []
        
        text_lower = feedback_text.lower()
        
        # 检测修改类型
        for param_type, keywords in self.modification_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                if param_type == "method":
                    # 提取方法
                    for method_key, method_value in self.method_map.items():
                        if method_key.lower() in text_lower:
                            params["method"] = method_value
                            break
                    if "method" not in params:
                        warnings.append("未识别出具体的克里金方法")
                
                elif param_type == "model":
                    # 提取模型
                    for model_key, model_value in self.model_map.items():
                        if model_key.lower() in text_lower:
                            params["variogram_model"] = model_value
                            break
                    if "variogram_model" not in params:
                        warnings.append("未识别出具体的半变异函数模型")
                
                elif param_type == "colormap":
                    # 提取色带
                    for colormap_key, colormap_value in self.colormap_map.items():
                        if colormap_key.lower() in text_lower:
                            params["colormap"] = colormap_value
                            break
                    if "colormap" not in params:
                        # 尝试直接提取颜色名称
                        if "红" in text_lower and "黄" in text_lower and "绿" in text_lower:
                            params["colormap"] = "RdYlGn"
                        elif "红" in text_lower and "黄" in text_lower and "蓝" in text_lower:
                            params["colormap"] = "RdYlBu"
                        else:
                            warnings.append("未识别出具体的色带")
                
                elif param_type == "n_classes":
                    # 提取分级数
                    import re
                    numbers = re.findall(r'\d+', feedback_text)
                    if numbers:
                        params["n_classes"] = int(numbers[0])
                    else:
                        warnings.append("未指定具体的分级数量")
                
                elif param_type == "smooth_sigma":
                    # 提取平滑参数
                    import re
                    numbers = re.findall(r'\d+\.?\d*', feedback_text)
                    if numbers:
                        params["smooth_sigma"] = float(numbers[0])
                    else:
                        # 默认平滑值
                        params["smooth_sigma"] = 1.0
                
                elif param_type == "lighten":
                    # 亮度调整
                    if "增加" in text_lower or "提高" in text_lower or "亮" in text_lower:
                        params["lighten"] = True
                    elif "减少" in text_lower or "降低" in text_lower or "暗" in text_lower:
                        params["lighten"] = False
        
        # 构建返回结果
        result = {
            "mcp_context": {
                "params": params,
                "warnings": warnings
            },
            "parsed_feedback": {
                "original_text": feedback_text,
                "modifications": list(params.keys()),
                "warnings": warnings
            }
        }
        
        return result

    async def parse_feedback_async(self, feedback_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """异步解析反馈"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse_feedback, feedback_text, context)


# 全局实例
enhanced_nlp_processor = EnhancedNLPProcessor()
feedback_processor = FeedbackProcessor()


def parse_text_enhanced(text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """增强的文本解析函数"""
    return enhanced_nlp_processor.parse_text(text, context)


async def parse_text_enhanced_async(text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """增强的异步文本解析函数"""
    return await enhanced_nlp_processor.parse_text_async(text, context)


def parse_feedback_enhanced(feedback_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """增强的反馈解析函数"""
    return feedback_processor.parse_feedback(feedback_text, context)


async def parse_feedback_enhanced_async(feedback_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """增强的异步反馈解析函数"""
    return await feedback_processor.parse_feedback_async(feedback_text, context)


if __name__ == "__main__":
    # 测试示例
    processor = EnhancedNLPProcessor()
    
    test_cases = [
        "绘制四川盆地龙潭组灰岩等值线图",
        "查询塔里木盆地砂岩厚度数据",
        "使用泛克里金方法重新绘图",
        "将颜色改为红黄绿",
        "增加分级到15级",
        "平滑参数调整为2.5"
    ]
    
    for test_text in test_cases:
        print(f"\n输入: {test_text}")
        result = processor.parse_text(test_text)
        print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
