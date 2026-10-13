import pytest

from qtro.config import load_config
from qtro.variants import MODEL_NAMES, parse_variant


def test_depth_variant_sets_layers_and_environment_and_labels_the_checkpoint():
    cfg = load_config(None, None)
    assert parse_variant("qtro", "4x8", cfg) == ("L4_dE8", {})
    assert (cfg.model.n_layers, cfg.model.d_env) == (4, 8)
    assert parse_variant("qubit_ucr", "16x8", cfg) == ("L16_dE8", {})


def test_rank_variant_reaches_the_registry_as_a_kwarg():
    cfg = load_config(None, None)
    assert parse_variant("cnto", "rank27", cfg) == ("rank27", {"rank": 27})
    assert parse_variant("deeponet", "RANK19", cfg) == ("rank19", {"rank": 19})


@pytest.mark.parametrize("model,text", [("qtro", "mix2x4x4"), ("qtro", "4"), ("qtro", "0x8"),
                                        ("cnto", "4x8"), ("cnto", "rank0"), ("nope", "4x8")])
def test_malformed_variants_are_rejected(model, text):
    with pytest.raises(ValueError):
        parse_variant(model, text, load_config(None, None))


def test_every_registry_name_has_a_variant_family():
    assert set(MODEL_NAMES) == {"qtro", "cnto", "seqpred", "deeponet",
                                "qubit_hea", "qubit_dense", "qubit_ucr"}
