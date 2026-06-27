"""conftest.py — 全局测试配置."""

from __future__ import annotations

import os
import sys
import types

# 在 mediapy 被任何模块导入前注入 stub
# mediapy 1.2.6 在 Python 3.12 + numpy 2.x 下无法导入
if "mediapy" not in sys.modules:
    _stub = types.ModuleType("mediapy")
    _stub.__file__ = "(conftest-stub)"
    _stub.set_ffmpeg = lambda _path: None
    sys.modules["mediapy"] = _stub

# 统一设置测试的 src 路径 (session 作用域, 只执行一次)
# 所有测试文件通过 conftest.py 自动获得正确的 PYTHONPATH
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
