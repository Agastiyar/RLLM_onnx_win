"""RLLM Python wrapper — sets up CUDA DLL paths before loading native module."""
import os
import sys
import ctypes

# Add CUDA DLL directories before importing the native module
_cuda_paths = [
    r"E:\CUDA\v11.8\bin",
    r"E:\CUDA\v12.6\bin",
    os.path.join(os.path.dirname(__file__), "..", "..", "target", "release"),
]

for _p in _cuda_paths:
    if os.path.isdir(_p):
        os.add_dll_directory(_p)
        if _p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _p + ";" + os.environ.get("PATH", "")

# Now import the native module
from .rllm_native import *
from .rllm_native import version as _version, RllmConfig as _RllmConfig

__version__ = _version()
RllmConfig = _RllmConfig
