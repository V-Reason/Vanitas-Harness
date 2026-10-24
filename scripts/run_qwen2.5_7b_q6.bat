@echo off
chcp 65001 >nul
setlocal enableextensions

:: ============================================================================
:: run_qwen2.5_7b_q6.bat - 通用主力模型启动器（默认端口 8080）
:: 用法: run_qwen2.5_7b_q6.bat [模型路径] [档位 1|2|3]
::   模型路径省略 -> models\qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf
::   档位省略     -> VANITAS_PROFILE -> 交互输入 -> 1
::   VANITAS_PORT -> 覆盖端口，仅用于 8080 被外部进程占用时的逃生与自动化测试
:: 7B 权重是 llama.cpp 分片格式（-00001-of-00002 / -00002-of-00002）：
::   只传第 1 片，llama.cpp 按文件名自动续载其余分片；守卫负责校验全部分片。
::   传第 2 片会被守卫拒绝；不要把分片改名或合并，那会破坏分片元数据。
:: 档位依据：实测空闲底噪约 1.8 GiB，7B Q6_K 权重约 5.96 GiB，F16 KV 会顶到显存上限，
::           因此默认 Q8_0 KV（近无损），F16 只在显存全空时手动选 3。
:: ============================================================================

set "BASE_DIR=<REPOSITORY_ROOT>"
set "BIN_DIR=%BASE_DIR%\bin"
set "DEFAULT_MODEL=%BASE_DIR%\models\qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf"
set "PORT=8080"
set "HOST=127.0.0.1"

if defined VANITAS_PORT set "PORT=%VANITAS_PORT%"

set "MODEL_PATH=%~1"
if "%MODEL_PATH%"=="" set "MODEL_PATH=%DEFAULT_MODEL%"
if not "%MODEL_PATH:~1,1%"==":" set "MODEL_PATH=%BASE_DIR%\%MODEL_PATH%"
for %%F in ("%MODEL_PATH%") do set "MODEL_ALIAS=%%~nF"
:: 分片后缀 -00001-of-00002 共 15 字符；别名必须指回模型本体，否则 /v1/models 与报告会带分片号
if not "%MODEL_ALIAS:-of-=%"=="%MODEL_ALIAS%" set "MODEL_ALIAS=%MODEL_ALIAS:~0,-15%"

set "PROFILE=%~2"
if "%PROFILE%"=="" set "PROFILE=%VANITAS_PROFILE%"
if not "%PROFILE%"=="" goto :PROFILE_READY

echo ==============================================================================
echo 请选择 Qwen2.5-7B-Instruct Q6_K 运行档位:
echo [1] 4K 上下文 + Q8_0 KV  近无损，应用侧显存约 6.0 GB（默认）
echo [2] 8K 上下文 + Q4_0 KV  长文档，应用侧显存约 6.0 GB
echo [3] 4K 上下文 + F16 KV   理论最优精度，需空闲显存 ≥ 7.2 GB
echo ==============================================================================
set /p PROFILE="请输入选项代号 [默认 1]: "

:PROFILE_READY
if "%PROFILE%"=="" set "PROFILE=1"
if not "%PROFILE%"=="1" if not "%PROFILE%"=="2" if not "%PROFILE%"=="3" (
    echo [WARN] 未知档位 "%PROFILE%"，回退到 1。
    set "PROFILE=1"
)

if "%PROFILE%"=="2" (
    set "CTX_SIZE=8192"
    set "KV_K=q4_0"
    set "KV_V=q4_0"
    set "MODE_DESC=8K 上下文 · Q4_0 KV（长文档）"
) else if "%PROFILE%"=="3" (
    set "CTX_SIZE=4096"
    set "KV_K=f16"
    set "KV_V=f16"
    set "MODE_DESC=4K 上下文 · F16 KV（需空闲显存 ≥ 7.2 GB）"
) else (
    set "CTX_SIZE=4096"
    set "KV_K=q8_0"
    set "KV_V=q8_0"
    set "MODE_DESC=4K 上下文 · Q8_0 KV（近无损）"
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
echo [LAUNCH] 上下文: %CTX_SIZE% tokens, KV: %KV_K%
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
    -ctk %KV_K% ^
    -ctv %KV_V% ^
    -a "%MODEL_ALIAS%" ^
    -fit off ^
    --metrics
set "SERVER_EXIT=%errorlevel%"
echo [EXIT] llama-server 已退出，退出码 %SERVER_EXIT%
if not "%SERVER_EXIT%"=="0" pause
exit /b %SERVER_EXIT%
