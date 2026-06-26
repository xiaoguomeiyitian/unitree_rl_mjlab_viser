"""conftest.py — 全局测试配置."""

from __future__ import annotations

import sys
import types

# 在 mediapy 被任何模块导入前注入 stub
# mediapy 1.2.6 在 Python 3.12 + numpy 2.x 下无法导入
if "mediapy" not in sys.modules:
    _stub = types.ModuleType("mediapy")
    _stub.__file__ = "(conftest-stub)"
    _stub.set_ffmpeg = lambda _path: None
    sys.modules["mediapy"] = _stub
