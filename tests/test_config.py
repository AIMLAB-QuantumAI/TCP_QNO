import textwrap
from pathlib import Path

import pytest

from qtro.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_match_the_shipped_grid_and_recipe():
    cfg = load_config(None, None)
    assert (cfg.grid.n_w, cfg.grid.n_q, cfg.grid.n_phase) == (16, 16, 2)
    assert cfg.grid.beta == 0.5
    assert cfg.model.d_env == 4
    assert cfg.model.n_layers == 8
    assert cfg.model.d_cond == 8
    assert cfg.model.intensity_hidden == 64
    assert cfg.model.reset_max_probability == 1.0
    assert cfg.model.qubit_init_scale == 0.01
    assert cfg.model.qubit_controls == -1
    assert cfg.train.lr == 1e-3
    assert cfg.train.n_steps == 20000
    assert cfg.train.batch_configs == 16
    assert cfg.train.h0 == 5
    assert cfg.train.seeds == [0, 1, 2]
    assert cfg.sim.device == "auto"


def test_yaml_overrides_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        grid:
          n_w: 8
        net:
          capacity_pps: 3000.0
    """), encoding="utf-8")
    cfg = load_config(str(p), None)
    assert cfg.grid.n_w == 8
    assert cfg.net.capacity_pps == 3000.0
    assert cfg.grid.n_q == 16  # untouched default


def test_set_override_coerces_type():
    cfg = load_config(None, ["grid.n_w=32", "aqm.w_q=0.0002", "sim.device=cpu"])
    assert cfg.grid.n_w == 32 and isinstance(cfg.grid.n_w, int)
    assert cfg.aqm.w_q == 0.0002
    assert cfg.sim.device == "cpu"


def test_unknown_override_key_raises():
    with pytest.raises(KeyError):
        load_config(None, ["grid.nonexistent=1"])
    with pytest.raises(KeyError):
        load_config(None, ["nosuchsection.x=1"])


def test_unknown_yaml_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("grid:\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(str(p), None)


def test_removed_keys_are_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  base_channel_mode: none\n", encoding="utf-8")
    with pytest.raises(KeyError, match="base_channel_mode"):
        load_config(str(path), None)


@pytest.mark.parametrize("name", ["base.yaml", "ns3_h20.yaml", "ns3_h5.yaml"])
def test_shipped_configs_list_every_field(name):
    """Each shipped config claims to be fully resolved, so prove it is one."""
    from dataclasses import fields, is_dataclass

    import yaml

    with open(ROOT / "configs" / name, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    def check(obj, mapping, path=""):
        for f in fields(obj):
            assert f.name in mapping, f"configs/{name} misses {path}{f.name}"
            value = getattr(obj, f.name)
            if is_dataclass(value):
                check(value, mapping[f.name], f"{path}{f.name}.")

    check(load_config(None, None), data)


def test_list_of_int_round_trips():
    cfg = load_config(None, ["data.horizons_train=1,2,3"])
    assert cfg.data.horizons_train == [1, 2, 3]
    assert all(isinstance(x, int) for x in cfg.data.horizons_train)


def test_single_element_list_stays_a_list():
    cfg = load_config(None, ["train.seeds=7"])
    assert cfg.train.seeds == [7]


def test_list_tolerates_spaces_and_trailing_separators():
    cfg = load_config(None, ["data.horizons_train= 1, 5 ,10, "])
    assert cfg.data.horizons_train == [1, 5, 10]


def test_empty_list_override_is_rejected():
    with pytest.raises(ValueError, match="comma-separated"):
        load_config(None, ["train.seeds=,"])


def test_scalar_fields_still_coerce():
    cfg = load_config(None, ["train.lr=0.05", "train.batch_configs=4", "sim.device=cpu"])
    assert cfg.train.lr == 0.05
    assert cfg.train.batch_configs == 4
    assert cfg.sim.device == "cpu"


def test_sections_are_reachable_by_yaml_too(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  h0: 3\ndata:\n  c_min: 8\n", encoding="utf-8")
    cfg = load_config(str(path), [])
    assert cfg.train.h0 == 3 and cfg.data.c_min == 8


def test_unknown_key_in_a_section_still_raises():
    with pytest.raises(KeyError):
        load_config(None, ["train.nonexistent=1"])
