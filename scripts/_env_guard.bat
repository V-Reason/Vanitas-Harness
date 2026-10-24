@echo off
chcp 65001 >nul
setlocal enableextensions

:: ============================================================================
:: _env_guard.bat - 单实例独占守卫（运行时 / 模型完整性 / 端口）
:: 退出码即调用方契约，禁止在调用方复制本逻辑：
::   0 通过 | 1 端口被占 | 2 模型缺失或非法 | 3 运行时缺失 | 4 旧实例未退出 | 9 参数缺失
:: 顺序刻意保持为"先验模型、再杀旧实例"：目标模型本身不可用时，不应连累正在服务的实例。
:: 杀完旧实例仍占用端口，说明持有者是外部进程，必须报出 PID 让操作者决策。
:: ============================================================================

set "TARGET_PORT=%~1"
set "TARGET_MODEL=%~2"
set "BASE_DIR=%~dp0.."

if "%TARGET_PORT%"=="" goto :E_USAGE
if "%TARGET_MODEL%"=="" goto :E_USAGE
:: 用 delims 剔除数字后仍有残余，说明端口混入了非数字字符（echo 会带尾随空格，不能用正则判尾）
for /f "delims=0123456789" %%P in ("%TARGET_PORT%") do goto :E_USAGE

echo [GUARD] 目标端口 %TARGET_PORT%
echo [GUARD] 目标模型 "%TARGET_MODEL%"

:: 1) 运行时完整性：缺 DLL 时 llama-server 只会抛 0xc0000135，先替操作者挡住
for %%F in (llama-server.exe llama.dll ggml-base.dll ggml-cuda.dll) do (
    if not exist "%BASE_DIR%\bin\%%F" (
        echo [GUARD-FATAL] 运行时缺失: "%BASE_DIR%\bin\%%F"
        echo [GUARD-FATAL] 请将 llama.cpp CUDA 预编译包完整解压到 bin\ 目录。
        exit /b 3
    )
)

:: 2) 模型校验：存在性 -> 体积下限 -> GGUF magic，半截下载与"网页存成 .gguf"都会在这里被拦
if not exist "%TARGET_MODEL%" (
    echo [GUARD-FATAL] 模型文件缺失: "%TARGET_MODEL%"
    echo [GUARD-FATAL] 请核对 models\ 下的文件名，或先完成下载。
    exit /b 2
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='%TARGET_MODEL%'; $fi=Get-Item -LiteralPath $p; if($fi.Length -lt 104857600){exit 2}; $fs=[IO.File]::OpenRead($p); $b=New-Object byte[] 4; [void]$fs.Read($b,0,4); $fs.Dispose(); if([Text.Encoding]::ASCII.GetString($b) -ne 'GGUF'){exit 3}; exit 0"
set "MODEL_CHECK=%errorlevel%"
if "%MODEL_CHECK%"=="2" (
    echo [GUARD-FATAL] 模型文件过小（<100 MB），疑似下载未完成: "%TARGET_MODEL%"
    exit /b 2
)
if "%MODEL_CHECK%"=="3" (
    echo [GUARD-FATAL] 文件头不是 GGUF，可能把网页或压缩包存成了 .gguf: "%TARGET_MODEL%"
    exit /b 2
)
if not "%MODEL_CHECK%"=="0" (
    echo [GUARD-FATAL] 模型校验器异常（exit=%MODEL_CHECK%），请检查 PowerShell 是否可用。
    exit /b 2
)

set "MODEL_MB=?"
for /f "delims=" %%M in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "[math]::Round((Get-Item -LiteralPath '%TARGET_MODEL%').Length/1MB)"') do set "MODEL_MB=%%M"

:: 3) 清理存量实例：8G 显存禁不起双实例，宁可杀错不可放过
set "STALE="
for /f "tokens=1,2 delims=," %%A in ('tasklist /fi "imagename eq llama-server.exe" /fo csv /nh 2^>nul ^| findstr /i "llama-server.exe"') do (
    set "STALE=%%~B"
    echo [GUARD-WARN] 发现存量 llama-server 实例 PID=%%~B，正在强制回收显存...
)
if not defined STALE goto :NO_STALE

taskkill /f /im llama-server.exe >nul 2>&1
set /a KILL_WAIT=0
:KILL_POLL
tasklist /fi "imagename eq llama-server.exe" 2>nul | findstr /i "llama-server.exe" >nul
if errorlevel 1 goto :NO_STALE
set /a KILL_WAIT+=1
if %KILL_WAIT% geq 5 goto :E_KILL
ping -n 2 127.0.0.1 >nul
goto :KILL_POLL

:NO_STALE

:: 4) 端口占用：必须报出 PID 与映像名，否则操作者无从下手
netstat -ano -p tcp | findstr /r /c:":%TARGET_PORT% .*LISTENING" >nul
if errorlevel 1 goto :PORT_FREE

set "OWNER_PID="
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /r /c:":%TARGET_PORT% .*LISTENING"') do if not defined OWNER_PID set "OWNER_PID=%%P"
set "OWNER_NAME=未知进程"
for /f "tokens=1,2 delims=," %%A in ('tasklist /fi "pid eq %OWNER_PID%" /fo csv /nh 2^>nul') do set "OWNER_NAME=%%~A"
echo [GUARD-FATAL] 端口 %TARGET_PORT% 被 PID %OWNER_PID% ^(%OWNER_NAME%^) 占用。
echo [GUARD-FATAL] 处置：结束该进程；或 set "VANITAS_PORT=<空闲端口>" 后重新启动。
exit /b 1

:PORT_FREE

:: 5) 显存快照：只展示不拦截。阈值随档位变化，这里不替操作者拍板
:: 参数必须加引号：for /f 会把未加引号命令里的逗号当作参数分隔符
for /f "tokens=1,2 delims=, " %%A in ('nvidia-smi "--query-gpu=memory.used,memory.total" "--format=csv,noheader,nounits" 2^>nul') do echo [GUARD] GPU 已用 %%A MiB / 总量 %%B MiB

echo [GUARD] 环境核验通过：模型 %MODEL_MB% MB，端口 %TARGET_PORT% 空闲，显存通道已独占。
exit /b 0

:E_USAGE
echo [GUARD-FATAL] 用法: _env_guard.bat ^<端口^> ^<模型绝对路径^>
exit /b 9

:E_KILL
echo [GUARD-FATAL] 存量 llama-server 实例 5 秒内未退出，请手动结束进程后重试。
exit /b 4
