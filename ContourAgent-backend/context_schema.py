from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class MCPContextSchema:
    task: Dict[str, Any] = field(default_factory=dict)       # 任务定义（自然语言输入、解析结果）
    params: Dict[str, Any] = field(default_factory=dict)     # 插值/绘图参数
    data: Dict[str, Any] = field(default_factory=dict)       # 数据输入/预处理结果
    results: Dict[str, Any] = field(default_factory=dict)    # 插值结果、渲染图件路径
    feedbackParsed: Dict[str, Any] = field(default_factory=dict)  # 最新一次反馈解析
    errors: List[str] = field(default_factory=list)          # 错误信息
    history: List[Dict[str, Any]] = field(default_factory=list)   # 历史记录（可选）

    def update_params(self, new_params: Dict[str, Any]):
        """更新 params 部分"""
        self.params.update(new_params)

    def add_error(self, error_msg: str):
        """记录错误"""
        self.errors.append(error_msg)

    def add_history(self, record: Dict[str, Any]):
        """记录操作历史"""
        self.history.append(record)
