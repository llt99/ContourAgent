@echo off
echo ========================================
echo ContourAgent 测试启动脚本
echo ========================================
echo.

echo 1. 启动后端服务...
cd /d "F:\llt\古地理\ContourAgent\ContourAgent-backend"
start "ContourAgent Backend" python api.py

echo 等待5秒让后端服务启动...
timeout /t 5 /nobreak >nul

echo.
echo 2. 启动前端HTTP服务器...
cd /d "F:\llt\古地理\ContourAgent\ContourAgent-frontend"
start "ContourAgent Frontend" python start_http_server.py

echo 等待3秒让前端服务器启动...
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo ✅ 服务已启动！
echo ========================================
echo.
echo 后端服务: http://127.0.0.1:8000
echo 前端页面: http://127.0.0.1:8080/test-quick.html
echo.
echo 浏览器会自动打开前端测试页面。
echo.
echo 如果浏览器未自动打开，请手动访问：
echo http://127.0.0.1:8080/test-quick.html
echo.
echo ========================================
echo 测试说明：
echo - 在聊天窗口输入任务进行测试
echo - 示例: "绘制四川盆地龙潭组煤岩分布图"
echo - 按 Ctrl+Enter 或点击 Send 按钮发送
echo ========================================
echo.
echo 按任意键关闭此窗口（服务将继续运行）...
pause >nul
