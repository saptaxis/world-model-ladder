import pytest
import yaml
from pathlib import Path
from utils.config import RunConfig, load_config, generate_run_name


def test_run_config_defaults():
    cfg = RunConfig(arch="mlp", arch_params={"hidden_dims": [256]},
                    prediction="delta", training_mode="single_step",
                    data_mix="policy", data_path="/tmp/data")
    assert cfg.lr == 1e-3
    assert cfg.batch_size == 64
    assert cfg.rollout_k == 1


def test_run_config_empty_data_path_raises():
    with pytest.raises(ValueError, match="data_path is required"):
        RunConfig(arch="mlp", data_path="")


def test_generate_run_name():
    cfg = RunConfig(arch="mlp", arch_params={}, prediction="delta",
                    training_mode="single_step", data_mix="policy",
                    data_path="/tmp/data")
    name = generate_run_name(cfg)
    assert name == "mlp-delta-single_step_k1-policy"


def test_generate_run_name_with_suffix():
    cfg = RunConfig(arch="gru", arch_params={}, prediction="delta",
                    training_mode="multi_step", rollout_k=10,
                    data_mix="policy_primitives", data_path="/tmp/data",
                    suffix="seed42")
    name = generate_run_name(cfg)
    assert name == "gru-delta-multi_step_k10-policy_primitives--seed42"


def test_yaml_round_trip(tmp_path):
    cfg = RunConfig(arch="mlp", arch_params={"hidden_dims": [256, 256]},
                    prediction="delta", training_mode="multi_step",
                    rollout_k=5, data_mix="policy", data_path="/data/episodes")
    path = tmp_path / "test.yaml"
    cfg.save(path)
    loaded = RunConfig.load(path)
    assert loaded.arch == cfg.arch
    assert loaded.arch_params == cfg.arch_params
    assert loaded.rollout_k == cfg.rollout_k
    assert loaded.data_path == cfg.data_path


def test_load_config_with_overrides(tmp_path):
    cfg = RunConfig(arch="mlp", arch_params={"hidden_dims": [128]},
                    prediction="delta", training_mode="single_step",
                    data_mix="policy", data_path="/data")
    path = tmp_path / "base.yaml"
    cfg.save(path)
    loaded = load_config(str(path), overrides={"rollout_k": 20, "suffix": "test"})
    assert loaded.rollout_k == 20
    assert loaded.suffix == "test"
    assert loaded.arch == "mlp"  # unchanged
