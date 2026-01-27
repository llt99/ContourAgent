@echo off
REM 地质等值线图生成系统 - 测试运行脚本
REM 专门测试论文场景：使用普通克里金和球状模型生成四川盆地龙潭组煤岩分布图

echo ========================================
echo 地质等值线图生成系统测试
echo 论文场景：龙潭组煤岩分布图
echo ========================================
echo.

REM 检查Python环境
echo [1/3] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)
echo [✓] Python环境正常

REM 检查必要模块
echo.
echo [2/3] 检查Python依赖模块...
python -c "import numpy, scipy, sys" >nul 2>&1
if errorlevel 1 (
    echo [警告] 缺少必要模块，正在安装...
    pip install numpy scipy
)
echo [✓] 依赖模块检查完成

REM 运行后端测试
echo.
echo [3/3] 运行后端测试...
echo ========================================
python test-coal-distribution.py

if errorlevel 1 (
    echo.
    echo [错误] 测试执行失败
    echo 请检查：
    echo 1. 是否在项目根目录运行
    echo 2. 是否已安装pykrige模块
    echo 3. 后端文件路径是否正确
    pause
    exit /b 1
)

echo.
echo ========================================
echo 测试完成！
echo.
echo 如需运行前端测试，请：
echo 1. 打开浏览器控制台
echo 2. 复制 test-frontend.js 内容
echo 3. 粘贴执行后输入: runAllTests()
echo.
pause
