from __future__ import annotations

from types import SimpleNamespace

from backend.reguaz.utils.devices import select_inference_device


def fake_torch(*, mps: bool, cuda: bool) -> object:
    return SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        cuda=SimpleNamespace(is_available=lambda: cuda),
    )


def test_mps_is_preferred_and_cpu_is_safe_fallback() -> None:
    assert (
        select_inference_device(torch_module=fake_torch(mps=True, cuda=True)) == "mps"
    )
    assert (
        select_inference_device(torch_module=fake_torch(mps=False, cuda=False)) == "cpu"
    )
    assert (
        select_inference_device("cpu", torch_module=fake_torch(mps=True, cuda=True))
        == "cpu"
    )
