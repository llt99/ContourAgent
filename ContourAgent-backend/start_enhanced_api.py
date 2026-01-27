"""
启动增强的自然语言解析和多Agent协作API服务
"""
import asyncio
import sys
import os
import subprocess
import time
import requests
import threading

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_port_in_use(port: int) -> bool:
    """检查端口是否被占用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False
        except socket.error:
            return True

def start_enhanced_api():
    """启动增强API服务"""
    print("=" * 60)
    print("🚀 启动增强的自然语言解析和多Agent协作API服务")
    print("=" * 60)
    
    # 检查端口
    port = 8000
    if check_port_in_use(port):
        print(f"❌ 端口 {port} 已被占用")
        print("请关闭占用该端口的服务或修改端口")
        return
    
    # 检查依赖
    print("\n1. 检查依赖模块...")
    try:
        from enhanced_nlp_processor import EnhancedNLPProcessor
        from enhanced_agent_framework import setup_agents, orchestrator
        from enhanced_api import app
        print("✅ 依赖模块加载成功")
    except ImportError as e:
        print(f"❌ 依赖模块缺失: {e}")
        print("请确保所有增强模块文件都存在")
        return
    
    # 初始化Agent
    print("\n2. 初始化Agent框架...")
    try:
        from mcp_tool import text_to_sql_query_tool, kriging_interpolate, render_map_tool
        setup_agents(
            query_func=text_to_sql_query_tool,
            kriging_func=kriging_interpolate,
            render_func=render_map_tool
        )
        registered_agents = [agent_type.value for agent_type in orchestrator.agents.keys()]
        print(f"✅ Agent初始化完成: {registered_agents}")
    except Exception as e:
        print(f"❌ Agent初始化失败: {e}")
        return
    
    # 启动服务
    print(f"\n3. 启动服务 (http://127.0.0.1:{port})...")
    print("   按 Ctrl+C 停止服务")
    print()
    
    try:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n\n🛑 服务已停止")
    except Exception as e:
        print(f"\n❌ 服务启动失败: {e}")
        import traceback
        traceback.print_exc()

def test_api():
    """测试API服务"""
    print("\n" + "=" * 60)
    print("🧪 测试API服务")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8000"
    
    # 等待服务启动
    print("等待服务启动...")
    for i in range(10):
        try:
            response = requests.get(f"{base_url}/system/status", timeout=1)
            if response.status_code == 200:
                print("✅ 服务已启动")
                break
        except:
            time.sleep(1)
    else:
        print("❌ 服务启动超时")
        return
    
    # 测试系统状态
    print("\n1. 测试系统状态...")
    try:
        response = requests.get(f"{base_url}/system/status")
        if response.status_code == 200:
            status = response.json()
            print("✅ 系统状态正常")
            print(f"   Agent: {status['agent_framework']['registered_agents']}")
            print(f"   NLP类型: {status['nlp_processor']['type']}")
        else:
            print(f"❌ 状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    # 测试NLP解析
    print("\n2. 测试NLP解析...")
    try:
        response = requests.post(
            f"{base_url}/enhanced/parse",
            json={"text": "绘制四川盆地龙潭组灰岩等值线图", "use_enhanced_nlp": True}
        )
        if response.status_code == 200:
            result = response.json()
            task = result['result']['task']
            print("✅ NLP解析成功")
            print(f"   区域: {task.get('region', '未识别')}")
            print(f"   地层: {task.get('stratum', '未识别')}")
            print(f"   变量: {task.get('variable', '未识别')}")
            print(f"   置信度: {task.get('confidence', 0):.2f}")
        else:
            print(f"❌ 状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    # 测试Agent执行
    print("\n3. 测试Agent执行...")
    try:
        response = requests.post(
            f"{base_url}/agent/execute",
            json={"text": "绘制四川盆地龙潭组灰岩等值线图", "use_enhanced_nlp": True}
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Agent执行成功")
            print(f"   执行管道: {result.get('pipeline', [])}")
            print(f"   总耗时: {result.get('total_time', 0):.3f}s")
            print(f"   Agent结果数: {len(result.get('agent_results', []))}")
        else:
            print(f"❌ 状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    # 测试混合接口
    print("\n4. 测试混合接口...")
    try:
        response = requests.post(
            f"{base_url}/task",
            json={"text": "查询塔里木盆地砂岩厚度数据"}
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ 混合接口成功")
            if result.get("agent_results"):
                print(f"   Agent结果数: {len(result['agent_results'])}")
            if result.get("execution_summary"):
                print(f"   执行状态: {result['execution_summary']['success']}")
        else:
            print(f"❌ 状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    print("\n✅ 所有测试完成！")

def show_usage():
    """显示使用说明"""
    print("\n" + "=" * 60)
    print("📚 使用说明")
    print("=" * 60)
    print("""
启动服务:
  python start_enhanced_api.py

API接口:
  - 增强NLP解析: POST /enhanced/parse
  - Agent执行: POST /agent/execute
  - 混合接口: POST /task
  - 系统状态: GET /system/status
  - 历史记录: GET /history

支持的自然语言输入:
  📊 绘图任务:
    - "绘制四川盆地龙潭组灰岩等值线图"
    - "生成塔里木盆地砂岩分布图"
  
  🔍 数据查询:
    - "查询四川盆地龙潭组灰岩数据"
    - "获取塔里木盆地砂岩厚度"
  
  🔄 反馈修改:
    - "将颜色改为红黄绿"
    - "使用泛克里金方法"
    - "增加分级到15级"
    - "平滑参数调整为2.5"

测试API:
  python start_enhanced_api.py --test

查看详细文档:
  请查看 ENHANCED_SYSTEM_README.md
""")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # 启动服务并在后台运行测试
            import subprocess
            import threading
            
            def run_server():
                start_enhanced_api()
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # 等待服务启动
            time.sleep(3)
            
            # 运行测试
            test_api()
            
            # 等待用户按Ctrl+C
            print("\n按 Ctrl+C 停止服务")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n🛑 服务已停止")
                
        elif sys.argv[1] == "--help":
            show_usage()
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("使用 --help 查看帮助")
    else:
        # 直接启动服务
        start_enhanced_api()

if __name__ == "__main__":
    main()
