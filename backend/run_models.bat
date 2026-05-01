@echo off
title Llama Server Launcher

:: 启动第一个服务器 (GPU 0)
echo Starting Llama Server on GPU 0 (Port 7101)...
start "Llama Server GPU 0" cmd /k "set CUDA_VISIBLE_DEVICES=0 && F:\llama.cpp\llama-server -m F:\LLMs\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf --reasoning off -c 262144 -np 4 -b 4096 -ub 512 -sm layer -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --port 7101"

:: 稍微等待几秒，防止两个进程同时抢占资源导致冲突
timeout /t 3 /nobreak > nul

:: 启动第二个服务器 (GPU 1)
echo Starting Llama Server on GPU 1 (Port 7102)...
start "Llama Server GPU 1" cmd /k "set CUDA_VISIBLE_DEVICES=1 && F:\llama.cpp\llama-server -m F:\LLMs\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf --reasoning off -c 262144 -np 4 -b 4096 -ub 512 -sm layer -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --port 7102"

echo.
echo Both servers have been launched in separate windows.
pause