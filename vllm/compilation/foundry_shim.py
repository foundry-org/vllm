# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-export of ``foundry.integration.vllm`` with ImportError fallback.

Imported by ``vllm/config/compilation.py::CompilationConfig.__post_init__``
when ``graph_extension_config_path`` is set, and by
``vllm/compilation/decorators.py`` for the LOAD-mode ``do_not_compile``
check.

When the foundry package is not installed, the fallbacks below make every
foundry-aware code path short-circuit to upstream behavior, so vLLM still
builds and runs without foundry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import CompilationConfig

try:
    from foundry.integration.vllm import (  # noqa: F401
        CUDAGraphExtensionMode,
        get_graph_extension_mode,
        install_hooks,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

    import enum

    class CUDAGraphExtensionMode(str, enum.Enum):  # type: ignore[no-redef]
        NONE = "none"
        SAVE = "save"
        LOAD = "load"

    def get_graph_extension_mode() -> CUDAGraphExtensionMode:  # type: ignore[no-redef]
        return CUDAGraphExtensionMode.NONE

    def install_hooks(compilation_config: CompilationConfig) -> None:  # type: ignore[no-redef]
        # No-op fallback: foundry is not installed.
        return None


__all__ = [
    "CUDAGraphExtensionMode",
    "get_graph_extension_mode",
    "install_hooks",
]
