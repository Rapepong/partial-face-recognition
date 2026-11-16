"""Entrypoint for Streamlit Community Cloud: runs ../app.py.

It lives in deploy/ so that the cloud installs deploy/requirements.txt (CPU torch, headless OpenCV)
instead of the root requirements.txt, which is for training on a local GPU.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "app.py"), run_name="__main__")
