from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from full_path_pdnn_ffn import FullPathPBitFFN, FullPathPBitFFNConfig, load_full_path_checkpoint


def test_binary_matrix_inputs_and_gradients() -> None:
    config = FullPathPBitFFNConfig(
        input_size=5,
        hidden_sizes=(7,),
        output_size=3,
        input_temperature=0.125,
        hidden_temperature=0.5,
        coding="binary",
        learnable_encoding=True,
    )
    model = FullPathPBitFFN(config)
    model.set_sample_count(2)
    captured: list[torch.Tensor] = []
    hooks = [
        projection.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        for projection in model.projections
    ]
    output = model(torch.randn(4, 5))
    output.square().mean().backward()
    for hook in hooks:
        hook.remove()

    assert output.shape == (4, 3)
    assert captured
    for matrix_input in captured:
        assert torch.all((matrix_input == 0) | (matrix_input == 1))
    assert model.input_threshold.grad is not None
    assert model.input_temperature_raw.grad is not None
    assert model.hidden_thresholds.grad is not None
    assert model.hidden_temperature_raw.grad is not None
    assert float(model.effective_input_temperature()) > 0
    assert float(model.effective_hidden_temperature(0)) > 0


def test_checkpoint_round_trip_and_legacy_default() -> None:
    legacy = FullPathPBitFFN(FullPathPBitFFNConfig(input_size=5, hidden_sizes=(7,), output_size=3))
    assert not legacy.config.learnable_encoding
    assert legacy.config.coding == "bipolar"
    assert not any("threshold" in key or "temperature_raw" in key for key in legacy.state_dict())

    model = FullPathPBitFFN(
        FullPathPBitFFNConfig(
            input_size=5,
            hidden_sizes=(7,),
            output_size=3,
            input_temperature=0.125,
            hidden_temperature=0.5,
            coding="binary",
            learnable_encoding=True,
        )
    )
    payload = {
        "student_type": "full_path_binary_pdnn_v2",
        "student_config": model.checkpoint_config(),
        "student_state_dict": model.state_dict(),
        "training_args": {"layer": 0},
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save(payload, path)
        restored, restored_payload = load_full_path_checkpoint(str(path), device="cpu")
    assert restored_payload["student_type"] == "full_path_binary_pdnn_v2"
    assert restored.config.learnable_encoding
    assert restored.config.coding == "binary"
    assert torch.equal(restored.input_threshold, model.input_threshold)

    old_payload = {
        "student_type": "full_path_bipolar_pdnn_v1",
        "student_config": {
            "input_size": 5,
            "hidden_sizes": [7],
            "output_size": 3,
            "input_temperature": 1.0,
            "hidden_temperature": 1.0,
        },
        "student_state_dict": legacy.state_dict(),
        "training_args": {"layer": 0},
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "legacy_checkpoint.pt"
        torch.save(old_payload, path)
        restored_legacy, _ = load_full_path_checkpoint(str(path), device="cpu")
    assert restored_legacy.config.coding == "bipolar"
    assert not restored_legacy.config.learnable_encoding


if __name__ == "__main__":
    test_binary_matrix_inputs_and_gradients()
    test_checkpoint_round_trip_and_legacy_default()
    print("learnable encoding tests passed")
