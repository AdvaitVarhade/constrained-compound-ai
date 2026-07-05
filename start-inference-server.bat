@echo off
echo ============================================================
echo Starting llama-server for Constrained Compound AI System
echo ============================================================
echo Base Model: qwen2.5-coder-3b-q4_k_m.gguf
echo Adapters: orchestrator_adapter.gguf, coder_adapter.gguf
echo Memory Budget: 4GB VRAM (Fully Offloaded via -ngl 99)
echo.
echo NOTE: Make sure you have downloaded the base GGUF model into the ./models directory!
echo NOTE: Make sure llama-server.exe is downloaded and in your PATH or current directory.
echo.

.\llama-bin\llama-server.exe ^
  -m ./models/qwen2.5-coder-3b-q4_k_m.gguf ^
  -c 4096 ^
  -ngl 99 ^
  --host 127.0.0.1 ^
  --port 8080 ^
  -np 1 ^
  --lora ./coder_adapter.gguf,./orchestrator_adapter.gguf --lora-init-without-apply

pause
