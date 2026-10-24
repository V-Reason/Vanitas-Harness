@echo off
chcp 65001 >nul
setlocal enableextensions

:: ============================================================================
:: _env_guard.bat - 单实例独占守卫（运行时 / 模型完整性 / 端口）
:: 退出码即调用方契约，禁止在调用方复制本逻辑：
::   0 通过 | 1 端口被占 | 2 模型缺失/分片不全/非法 | 3 运行时缺失 | 4 旧实例未退出 | 9 参数缺失
:: 顺序刻意保持为"先验模型、再杀旧实例"：目标模型本身不可用时，不应连累正在服务的实例。
:: 杀完旧实例仍占用端口，说明持有者是外部进程，必须报出 PID 让操作者决策。
:: 分片模型（xxx-00001-of-0000N.gguf）：只接受第 1 片作为入口，逐片校验存在性与总体积；
:: 第 2..N 片由 llama.cpp 依据文件名自动续载，守卫不替操作者拼路径，也不合并权重。
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

:: 2) 模型校验：存在性 -> 分片完整性 -> 体积下限 -> GGUF magic
:: 分片规则来自 llama.cpp 自身命名（%s-%05d-of-%05d.gguf）：只认 -00001- 入口，
:: 缺片、乱序、半截下载、"网页存成 .gguf" 都在这里被拦下。
if not exist "%TARGET_MODEL%" (
    echo [GUARD-FATAL] 模型文件缺失: "%TARGET_MODEL%"
    echo [GUARD-FATAL] 请核对 models\ 下的文件名，或先完成下载。
    exit /b 2
)

set "CHK_STATUS="
set "CHK_A="
set "CHK_B="
set "CHK_C="
for /f "tokens=1-4 delims=|" %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $p='%TARGET_MODEL%'; $n=[IO.Path]::GetFileNameWithoutExtension($p); $d=[IO.Path]::GetDirectoryName($p); if(-not $d){$d='.'}; $m=[regex]::Match($n,'^(.+)-(\d{5})-of-(\d{5})$'); $files=@($p); if($m.Success){ $idx=[int]$m.Groups[2].Value; $tot=[int]$m.Groups[3].Value; if($idx -ne 1){Write-Output ('ERR|2|SHARD_ORDER|' + [IO.Path]::GetFileName($p)); exit 0}; if($tot -lt 1){Write-Output 'ERR|2|BAD_SHARD_COUNT|'; exit 0}; $files=@(); for($i=1;$i -le $tot;$i++){ $c=Join-Path $d ('{0}-{1:D5}-of-{2:D5}.gguf' -f $m.Groups[1].Value,$i,$tot); if(-not (Test-Path -LiteralPath $c)){Write-Output ('ERR|2|SHARD_MISSING|' + [IO.Path]::GetFileName($c)); exit 0}; $files += $c } }; $len=0; foreach($f in $files){ $len += (Get-Item -LiteralPath $f).Length }; if($len -lt 104857600){Write-Output 'ERR|2|TOO_SMALL|'; exit 0}; $fs=[IO.File]::OpenRead($p); $b=New-Object byte[] 4; [void]$fs.Read($b,0,4); $fs.Dispose(); if([Text.Encoding]::ASCII.GetString($b) -ne 'GGUF'){Write-Output 'ERR|2|BAD_MAGIC|'; exit 0}; Write-Output ('OK|{0}|{1}' -f [math]::Round($len/1MB),$files.Count)"') do (
    set "CHK_STATUS=%%A"
    set "CHK_A=%%B"
    set "CHK_B=%%C"
    set "CHK_C=%%D"
)

if not "%CHK_STATUS%"=="OK" goto :MODEL_REJECT

set "MODEL_MB=%CHK_A%"
set "SHARD_COUNT=%CHK_B%"

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

if "%SHARD_COUNT%"=="1" (
    echo [GUARD] 环境核验通过：模型 %MODEL_MB% MB（单文件），端口 %TARGET_PORT% 空闲，显存通道已独占。
) else (
    echo [GUARD] 环境核验通过：模型 %MODEL_MB% MB（%SHARD_COUNT% 个分片，校验完整），端口 %TARGET_PORT% 空闲，显存通道已独占。
)
exit /b 0

:: 模型校验失败分支：退出码与原因码来自 PowerShell 的结构化输出，禁止吞成统一 1
:MODEL_REJECT
set "MODEL_EXIT=%CHK_A%"
if not "%CHK_B%"=="" goto :MODEL_REJECT_DETAIL
echo [GUARD-FATAL] 模型校验器异常（无结构化输出），请检查 PowerShell 是否可用。
exit /b 2

:MODEL_REJECT_DETAIL
if "%CHK_B%"=="SHARD_ORDER" (
    echo [GUARD-FATAL] 分片序号错误：必须传入第 1 个分片，收到 "%CHK_C%"。
    echo [GUARD-FATAL] 正确示例：models\qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf
) else if "%CHK_B%"=="SHARD_MISSING" (
    echo [GUARD-FATAL] 模型分片缺失: "%CHK_C%"
    echo [GUARD-FATAL] 分片必须同目录、同前缀、编号连续；补齐后再启动。
) else if "%CHK_B%"=="BAD_SHARD_COUNT" (
    echo [GUARD-FATAL] 分片总数非法（必须为正整数）: "%TARGET_MODEL%"
) else if "%CHK_B%"=="TOO_SMALL" (
    echo [GUARD-FATAL] 模型体积不足 100 MB，疑似下载未完成: "%TARGET_MODEL%"
) else if "%CHK_B%"=="BAD_MAGIC" (
    echo [GUARD-FATAL] 文件头不是 GGUF，可能把网页或压缩包存成了 .gguf: "%TARGET_MODEL%"
) else (
    echo [GUARD-FATAL] 模型校验失败，原因码 "%CHK_B%"。
)
exit /b %MODEL_EXIT%

:E_USAGE
echo [GUARD-FATAL] 用法: _env_guard.bat ^<端口^> ^<模型绝对路径^>
exit /b 9

:E_KILL
echo [GUARD-FATAL] 存量 llama-server 实例 5 秒内未退出，请手动结束进程后重试。
exit /b 4
