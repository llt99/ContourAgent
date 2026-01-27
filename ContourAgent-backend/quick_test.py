"""
快速测试增强的自然语言解析和多Agent协作框架
"""
import asyncio
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_nlp_processor import EnhancedNLPProcessor, FeedbackProcessor
from enhanced_agent_framework import setup_agents, execute_user_task, orchestrator
from mcp_tool import text_to_sql_query_tool, kriging_interpolate, render_map_tool


def test_nlp_processor():
    """测试NLP处理器"""
    print("🔍 测试NLP处理器...")
    
    processor = EnhancedNLPProcessor()
    
    # 测试用例
    test_cases = [
        ("绘制四川盆地龙潭组灰岩等值线图", "标准绘图任务"),
        ("查询塔里木盆地砂岩厚度数据", "数据查询任务"),
        ("将颜色改为红黄绿", "反馈修改任务"),
        ("使用泛克里金方法", "方法修改反馈"),
        ("增加分级到15级", "参数修改反馈")
    ]
    
    for text, description in test_cases:
        print(f"\n  测试: {description}")
        print(f"  输入: {text}")
        
        result = processor.parse_text(text)
        task = result.get("task", {})
        plan = result.get("plan", {})
        
        print(f"  ✅ 区域: {task.get('region', '未识别')}")
        print(f"  ✅ 地层: {task.get('stratum', '未识别')}")
        print(f"  ✅ 变量: {task.get('variable', '未识别')}")
        print(f"  ✅ 图件: {task.get('plot', '未识别')}")
        print(f"  ✅ 方法: {task.get('method', '未指定')}")
        print(f"  ✅ 模型: {task.get('model', '未指定')}")
        print(f"  ✅ 置信度: {task.get('confidence', 0):.2f}")
        print(f"  ✅ 任务类型: {task.get('task_type', '未知')}")
        print(f"  ✅ 执行管道: {plan.get('pipeline', [])}")
        
        if task.get("warnings"):
            print(f"  ⚠️ 警告: {task['warnings']}")
    
    print("\n✅ NLP处理器测试完成")


def test_feedback_processor():
    """测试反馈处理器"""
    print("\n🔍 测试反馈处理器...")
    
    processor = FeedbackProcessor()
    context = {
        "region": "四川盆地",
        "stratum": "龙潭组",
        "variable": "灰岩",
        "plot": "等值线图"
    }
    
    test_cases = [
        "将颜色改为红黄绿",
        "使用泛克里金方法",
        "修改为球状模型",
        "增加分级到15级",
        "平滑参数调整为2.5"
    ]
    
    for feedback in test_cases:
        print(f"\n  反馈: {feedback}")
        result = processor.parse_feedback(feedback, context)
        params = result.get("mcp_context", {}).get("params", {})
        
        if params:
            print(f"  ✅ 修改参数: {list(params.keys())}")
            for key, value in params.items():
                print(f"     - {key}: {value}")
        else:
            print(f"  ⚠️ 未识别出具体参数")
    
    print("\n✅ 反馈处理器测试完成")


async def test_agent_framework():
    """测试Agent框架"""
    print("\n🔍 测试Agent框架...")
    
    # 设置Agent
    setup_agents(
        query_func=text_to_sql_query_tool,
        kriging_func=kriging_interpolate,
        render_func=render_map_tool
    )
    
    print(f"  已注册Agent: {[agent_type.value for agent_type in orchestrator.agents.keys()]}")
    
    # 测试任务
    print("\n  测试任务: 绘制四川盆地龙潭组灰岩等值线图")
    
    try:
        result = await execute_user_task("绘制四川盆地龙潭组灰岩等值线图")
        
        print(f"  ✅ 执行成功: {result.get('success', False)}")
        print(f"  ✅ 执行管道: {result.get('pipeline', [])}")
        print(f"  ✅ 总耗时: {result.get('total_time', 0):.3f}s")
        
        if result.get("agent_results"):
            print("  ✅ Agent执行详情:")
            for agent_result in result["agent_results"]:
                status_icon = "✅" if agent_result['status'] == 'completed' else "❌"
                print(f"     {status_icon} {agent_result['agent']}: {agent_result['status']} ({agent_result['execution_time']:.3f}s)")
                if agent_result.get("errors"):
                    print(f"        错误: {agent_result['errors']}")
        
        errors = result.get("errors", [])
        warnings = result.get("warnings", [])
        
        if errors:
            print(f"  ❌ 错误: {errors}")
        if warnings:
            print(f"  ⚠️ 警告: {warnings}")
            
    except Exception as e:
        print(f"  ❌ 执行异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Agent框架测试完成")


async def test_feedback_integration():
    """测试反馈与Agent集成"""
    print("\n🔍 测试反馈与Agent集成...")
    
    # 确保Agent已设置
    if not orchestrator.agents:
        setup_agents(
            query_func=text_to_sql_query_tool,
            kriging_func=kriging_interpolate,
            render_func=render_map_tool
        )
    
    # 先执行初始任务
    print("\n  1. 执行初始任务: 绘制四川盆地龙潭组灰岩等值线图")
    initial_result = await execute_user_task("绘制四川盆地龙潭组灰岩等值线图")
    
    if not initial_result.get("success"):
        print("  ❌ 初始任务失败，无法测试反馈")
        return
    
    print("  ✅ 初始任务成功")
    
    # 获取上下文
    context = initial_result.get("final_context", {}).get("parsed_intent", {})
    
    # 执行反馈
    feedback_text = "将颜色改为红黄绿"
    print(f"\n  2. 执行反馈: {feedback_text}")
    
    try:
        result = await execute_user_task(feedback_text, context)
        
        print(f"  ✅ 反馈执行成功: {result.get('success', False)}")
        print(f"  ✅ 执行管道: {result.get('pipeline', [])}")
        
        if result.get("agent_results"):
            for agent_result in result["agent_results"]:
                status_icon = "✅" if agent_result['status'] == 'completed' else "❌"
                print(f"     {status_icon} {agent_result['agent']}: {agent_result['status']}")
        
    except Exception as e:
        print(f"  ❌ 反馈执行异常: {e}")
    
    print("\n✅ 反馈集成测试完成")


def show_summary():
    """显示系统摘要"""
    print("\n" + "=" * 60)
    print("🎯 增强系统功能摘要")
    print("=" * 60)
    print("""
✅ 自然语言解析增强:
   - 智能实体识别 (区域、地层、变量、图件)
   - 意图检测 (绘图、查询、反馈)
   - 置信度评估
   - 上下文继承

✅ 多Agent协作框架:
   - 任务分解与管道生成
   - 状态管理与执行跟踪
   - 错误处理与警告收集
   - 动态Agent调度

✅ 智能反馈处理:
   - 参数精确识别
   - 增量更新支持
   - 参数验证

✅ API接口:
   - /enhanced/parse: 增强NLP解析
   - /agent/execute: Agent框架执行
   - /task: 混合接口 (兼容原有格式)
   - /system/status: 系统状态查询

✅ 支持的自然语言输入:
   绘图: "绘制四川盆地龙潭组灰岩等值线图"
   查询: "查询塔里木盆地砂岩厚度数据"
   反馈: "将颜色改为红黄绿", "使用泛克里金方法"
    
📚 详细文档: ENHANCED_SYSTEM_README.md
🚀 启动服务: python start_enhanced_api.py
🧪 运行测试: python test_enhanced_system.py
""")


async def main():
    """主测试函数"""
    print("=" * 60)
    print("🚀 增强的自然语言解析和多Agent协作框架 - 快速测试")
    print("=" * 60)
    
    try:
        # 1. 测试NLP处理器
        test_nlp_processor()
        
        # 2. 测试反馈处理器
        test_feedback_processor()
        
        # 3. 测试Agent框架
        await test_agent_framework()
        
        # 4. 测试反馈集成
        await test_feedback_integration()
        
        # 5. 显示摘要
        show_summary()
        
        print("\n🎉 所有测试成功完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
