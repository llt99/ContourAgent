#!/usr/bin/env python3
"""
启动使用测试数据的 ContourAgent 后端服务
"""

import os
import sys
import subprocess
import time

def start_backend():
    """启动后端服务"""
    print("=" * 60)
    print("启动 ContourAgent 测试数据后端服务")
    print("=" * 60)
    
    # 检查当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前目录: {current_dir}")
    
    # 确保在正确的目录
    if not os.path.exists("api.py"):
        print("❌ 错误: 请在 ContourAgent-backend 目录中运行此脚本")
        input("按回车键退出...")
        return
    
    # 检查测试数据文件
    if not os.path.exists("test_data.json"):
        print("⚠️  警告: test_data.json 不存在，将使用内置测试数据")
    else:
        print("✅ 测试数据文件存在")
    
    # 检查修改后的文件
    required_files = [
        "data_query_test.py",
        "mcp_tool.py"  # 已修改导入
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ 错误: 必需文件 {file} 不存在")
            input("按回车键退出...")
            return
    
    print("✅ 所有必需文件都存在")
    
    # 启动服务
    print("\n正在启动 FastAPI 服务...")
    print("服务地址: http://127.0.0.1:8000")
    print("API 文档: http://127.0.0.1:8000/docs")
    print("\n按 Ctrl+C 停止服务")
    print("-" * 60)
    
    try:
        # 使用 uvicorn 启动服务
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api:app", 
            "--host", "127.0.0.1", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        input("按回车键退出...")

if __name__ == "__main__":
    start_backend()
