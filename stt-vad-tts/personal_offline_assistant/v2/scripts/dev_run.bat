@echo off
cd /d %~dp0\..
if not exist .venv-v2 python -m venv .venv-v2
call .venv-v2\Scripts\activate
pip install -q fastapi uvicorn pyyaml numpy httpx
python app\serve.py
