#!/usr/bin/env python3
"""
简单的HTTP服务器，用于运行ContourAgent前端测试页面
避免file://协议的CORS问题
支持跨目录访问coal.csv文件
"""

import http.server
import socketserver
import webbrowser
import os
import urllib.parse

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # 解析路径
        path = urllib.parse.unquote(path)

        # 处理对coal.csv的请求
        if path.endswith('/coal.csv') or path.endswith('/../ContourAgent-backend/coal.csv'):
            # 返回后端目录中的coal.csv文件
            backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ContourAgent-backend')
            csv_path = os.path.join(backend_dir, 'coal.csv')
            if os.path.exists(csv_path):
                return csv_path
            else:
                # 如果文件不存在，返回404
                return super().translate_path('/404.html')

        # 处理对test-quick.html的请求
        if path.endswith('/test-quick.html'):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            html_path = os.path.join(current_dir, 'test-quick.html')
            if os.path.exists(html_path):
                return html_path

        # 默认行为
        return super().translate_path(path)

    def end_headers(self):
        # 添加CORS头，允许跨域请求
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    # 设置端口
    PORT = 8080

    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 切换到前端目录
    os.chdir(current_dir)

    # 创建HTTP服务器
    Handler = CustomHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🚀 HTTP server started successfully!")
            print(f"📁 Serving directory: {current_dir}")
            print(f"🌐 Access URL: http://127.0.0.1:{PORT}/test-quick.html")
            print(f"")
            print(f"Please open in your browser: http://127.0.0.1:{PORT}/test-quick.html")
            print(f"")
            print(f"📊 Supported features:")
            print(f"  • Directly read coal.csv file for plotting")
            print(f"  • Input 'Plot coal seam distribution of the Longtan Formation in the Sichuan Basin' for testing")
            print(f"  • No database connection required, CSV data is used directly")
            print(f"")
            print(f"Press Ctrl+C to stop the server")

            # Automatically open the browser
            try:
                webbrowser.open(f"http://127.0.0.1:{PORT}/test-quick.html")
            except:
                pass

            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 Server stopped")
                httpd.shutdown()

if __name__ == "__main__":
    main()
