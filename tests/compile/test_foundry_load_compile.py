# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Foundry LOAD keeps compilation opt-in for uncaptured forwards."""

import pytest
from torch import Tensor, nn

from vllm.compilation import foundry_shim
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)


@pytest.mark.parametrize(
    ("allow_compile", "expected_disabled"),
    [(None, True), ("0", True), ("1", False)],
)
def test_foundry_load_compile_opt_in(monkeypatch, allow_compile, expected_disabled):
    monkeypatch.setattr(
        foundry_shim,
        "get_graph_extension_mode",
        lambda: foundry_shim.CUDAGraphExtensionMode.LOAD,
    )
    if allow_compile is None:
        monkeypatch.delenv("FOUNDRY_LOAD_ALLOW_COMPILE", raising=False)
    else:
        monkeypatch.setenv("FOUNDRY_LOAD_ALLOW_COMPILE", allow_compile)

    @support_torch_compile
    class Model(nn.Module):
        def __init__(self, *, vllm_config: VllmConfig):
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return x

    config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    with set_current_vllm_config(config):
        model = Model(vllm_config=config)
    assert model.do_not_compile is expected_disabled
