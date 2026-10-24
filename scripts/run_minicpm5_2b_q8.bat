@echo off
chcp 65001 >nul
setlocal enableextensions

:: ============================================================================
:: run_minicpm5_2b_q8.bat - 轻量高质模型启动器（默认端口 8081）
:: 用法: run_minicpm5_2b_q8.bat [模型路径] [档位 1|2]
::   模型路径省略 -> models\MiniCPM5-2B-Q8_0.gguf
::   档位省略     -> VANITAS_PROFILE -> 交互输入 -> 1
::   VANITAS_PORT -> 覆盖端口，仅用于冲突逃生与自动化测试
:: 实际权重为 Q8_0（非早期文档假设的 Q6_K），约 2.56 GiB，8G 卡余量充足，
:: 因此两档均保持 F16 KV 不做量化；文件名即真实量化级别，禁止改成 q6。
:: ============================================================================

set "BASE_DIR=<REPOSITORY_ROOT>"
set "BIN_DIR=%BASE_DIR%\bin"
set "DEFAULT_MODEL=%BASE_DIR%\models\MiniCPM5-2B-Q8_0.gguf"
set "PORT=8081"
set "HOST=127.0.0.1"

if defined VANITAS_PORT set "PORT=%VANITAS_PORT%"

set "MODEL_PATH=%~1"
if "%MODEL_PATH%"=="" set "MODEL_PATH=%DEFAULT_MODEL%"
if not "%MODEL_PATH:~1,1%"==":" set "MODEL_PATH=%BASE_DIR%\%MODEL_PATH%"
for %%F in ("%MODEL_PATH%") do set "MODEL_ALIAS=%%~nF"

set "PROFILE=%~2"
if "%PROFILE%"=="" set "PROFILE=%VANITAS_PROFILE%"
if not "%PROFILE%"=="" goto :PROFILE_READY

echo ==============================================================================
echo 请选择 MiniCPM5-2B Q8_0 运行档位:
echo [1] 8K 上下文 + F16 KV   极速极稳，应用侧显存约 2.7 GB（默认）
echo [2] 32K 上下文 + F16 KV  长文实验，应用侧显存约 3.8 GB
echo ==============================================================================
set /p PROFILE="请输入选项代号 [默认 1]: "

:PROFILE_READY
if "%PROFILE%"=="" set "PROFILE=1"
if not "%PROFILE%"=="1" if not "%PROFILE%"=="2" (
    echo [WARN] 未知档位 "%PROFILE%"，回退到 1。
    set "PROFILE=1"
)

if "%PROFILE%"=="2" (
    set "CTX_SIZE=32768"
    set "MODE_DESC=32K 上下文 · F16 KV（长文实验）"
) else (
    set "CTX_SIZE=8192"
    set "MODE_DESC=8K 上下文 · F16 KV（极速极稳）"
)

call "%BASE_DIR%\scripts\_env_guard.bat" "%PORT%" "%MODEL_PATH%"
if errorlevel 1 (
    echo [ERROR] 前置校验未通过（exit=%errorlevel%），启动终止。
    pause
    exit /b %errorlevel%
)

echo ==============================================================================
echo [LAUNCH] 模型: %MODEL_ALIAS%
echo [LAUNCH] 档位: %MODE_DESC%
echo [LAUNCH] 上下文: %CTX_SIZE% tokens, KV: f16
echo [LAUNCH] 地址: http://%HOST%:%PORT%  (Web UI 与 OpenAI 兼容 API)
echo ==============================================================================

cd /d "%BIN_DIR%"
llama-server.exe ^
    -m "%MODEL_PATH%" ^
    --host %HOST% ^
    --port %PORT% ^
    -c %CTX_SIZE% ^
    -ngl 999 ^
    -np 1 ^
    -fa on ^
    -ctk f16 ^
    -ctv f16 ^
    -a "%MODEL_ALIAS%" ^
    -fit off ^
    --metrics
set "SERVER_EXIT=%errorlevel%"
echo [EXIT] llama-server 已退出，退出码 %SERVER_EXIT%
if not "%SERVER_EXIT%"=="0" pause
exit /b %SERVER_EXIT%
