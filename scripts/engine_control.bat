@echo off
chcp 65001 >nul
setlocal enableextensions

:: ============================================================================
:: engine_control.bat - 单实例推理引擎总控（RTX 4060 Laptop 8G）
:: 设计要点：
::   1) 启动一律用 start 派生独立窗口，否则服务器会占住本控制台，[4] 停止永远点不到
::   2) 停止后必须复测进程消失，taskkill 返回 0 不等于显存已释放
::   3) 临时 3B 验证走显式路径参数，不做隐式自动降级，避免"挂 7B 名牌跑 3B"
:: ============================================================================

set "BASE_DIR=<REPOSITORY_ROOT>"
set "TEMP_MODEL=%BASE_DIR%\models\qwen2.5-3b-instruct-q4_k_m.gguf"

:MENU
cls
set "RUNNING=未运行"
set "VRAM=GPU 状态不可用"
tasklist /fi "imagename eq llama-server.exe" 2>nul | findstr /i "llama-server.exe" >nul
if not errorlevel 1 set "RUNNING=运行中"
for /f "tokens=1,2 delims=, " %%A in ('nvidia-smi "--query-gpu=memory.used,memory.total" "--format=csv,noheader,nounits" 2^>nul') do set "VRAM=已用 %%A MiB / 总量 %%B MiB"

echo ==============================================================================
echo           Vanitas-Harness 工业级推理引擎总控 (RTX 4060 Laptop 8G)
echo ==============================================================================
echo   实例状态: %RUNNING%            显存: %VRAM%
echo ------------------------------------------------------------------------------
echo [1] 启动 Qwen2.5-7B-Instruct (Q6_K)  - [Port: 8080] 通用主力 / Agent
echo [2] 启动 MiniCPM5-2B (Q6_K)          - [Port: 8081] 轻量高质 / 长文
echo [3] 启动 HY-MT2-7B (Q6_K)            - [Port: 8082] 专业翻译
echo [4] 停止当前所有推理实例 (彻底释放显存)
echo [5] 查看运行状态 (进程 / 端口 / 显存)
echo [6] 临时验证: Qwen2.5-3B Q4_K_M      - [Port: 8080] 仅管道验收，不代表 7B 性能
echo [7] 退出总控
echo ==============================================================================
set /p CHOICE="请输入操作序号 [1-7]: "

if "%CHOICE%"=="1" goto :START_QWEN
if "%CHOICE%"=="2" goto :START_MINICPM
if "%CHOICE%"=="3" goto :START_HYMT
if "%CHOICE%"=="4" goto :STOP_ALL
if "%CHOICE%"=="5" goto :STATUS
if "%CHOICE%"=="6" goto :START_TEMP
if "%CHOICE%"=="7" exit /b 0

echo [WARN] 无效输入，请重新选择。
timeout /t 2 >nul
goto :MENU

:START_QWEN
call :LAUNCH "Vanitas :: Qwen2.5-7B" "run_qwen2.5_7b_q6.bat" ""
goto :MENU

:START_MINICPM
call :LAUNCH "Vanitas :: MiniCPM5-2B" "run_minicpm5_2b_q6.bat" ""
goto :MENU

:START_HYMT
call :LAUNCH "Vanitas :: HY-MT2-7B" "run_hymt2_7b_q6.bat" ""
goto :MENU

:START_TEMP
if not exist "%TEMP_MODEL%" (
    echo [ERROR] 未找到临时验证模型: "%TEMP_MODEL%"
    echo [ERROR] 请先把 qwen2.5-3b-instruct-q4_k_m.gguf 放入 models\ 目录。
    pause
    goto :MENU
)
call :LAUNCH "Vanitas :: Qwen2.5-3B (临时验证)" "run_qwen2.5_7b_q6.bat" "%TEMP_MODEL%"
goto :MENU

:STOP_ALL
echo [ACTION] 正在停止所有 llama-server 实例...
taskkill /f /im llama-server.exe >nul 2>&1
set /a STOP_WAIT=0
:STOP_POLL
tasklist /fi "imagename eq llama-server.exe" 2>nul | findstr /i "llama-server.exe" >nul
if errorlevel 1 goto :STOP_DONE
set /a STOP_WAIT+=1
if %STOP_WAIT% geq 5 goto :STOP_TIMEOUT
ping -n 2 127.0.0.1 >nul
goto :STOP_POLL

:STOP_DONE
echo [OK] 已停止，显存通道已释放。
nvidia-smi "--query-gpu=memory.used,memory.total" "--format=csv,noheader"
pause
goto :MENU

:STOP_TIMEOUT
echo [WARN] 5 秒内仍有 llama-server 存活，请用任务管理器确认后再启动新实例。
pause
goto :MENU

:STATUS
cls
echo ============================== 运行状态 ==============================
echo [进程]
tasklist /fi "imagename eq llama-server.exe" /fo table
echo.
echo [端口 8080-8082]
netstat -ano -p tcp | findstr /r /c:":8080 .*LISTENING" /c:":8081 .*LISTENING" /c:":8082 .*LISTENING"
echo.
echo [显存]
nvidia-smi "--query-gpu=memory.used,memory.total" "--format=csv,noheader"
echo ======================================================================
pause
goto :MENU

:LAUNCH
start "%~1" /D "%BASE_DIR%" cmd /c call "%~dp0%~2" "%~3"
if errorlevel 1 (
    echo [ERROR] 派生启动窗口失败，请检查 %~2 是否存在。
    pause
    exit /b 1
)
echo [OK] 已派生独立窗口：%~1
echo [INFO] 档位选择请在弹出窗口内完成；本控制台继续可用。
timeout /t 2 >nul
exit /b 0
