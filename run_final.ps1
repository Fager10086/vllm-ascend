$ErrorActionPreference = "Continue"
$p = Start-Process -FilePath "python" -ArgumentList "D:\vllm-ascend\final_ppt.py" -NoNewWindow -Wait -PassThru -RedirectStandardOutput "D:\vllm-ascend\ppt_stdout.txt" -RedirectStandardError "D:\vllm-ascend\ppt_stderr.txt"
Write-Host "Exit code: $($p.ExitCode)"
