import json
import subprocess
import sys
from pathlib import Path

from helpers import write_shards

from qtro.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train" / "train.py"


def test_train_cli_writes_a_loadable_checkpoint_and_a_validation_trace(tmp_path):
    data = tmp_path / "data"
    write_shards(data, n_conditions=4, horizons=(1,), M=512, seed=0)
    out = tmp_path / "run"
    cmd = [sys.executable, str(TRAIN), "--config", str(ROOT / "configs" / "base.yaml"),
           "--data", str(data), "--out", str(out), "--model", "qtro", "--variant", "1x2",
           "--seeds", "0", "--steps", "2", "--batch-configs", "1", "--h0", "1",
           "--val-fraction", "0.25", "--val-horizons", "1", "--eval-every", "1",
           "--device", "cpu"]
    subprocess.run(cmd, check=True, cwd=tmp_path, capture_output=True, text=True)
    checkpoint = out / "checkpoints" / "qtro_L1_dE2_seed0.pt"
    model, _, _, payload = load_checkpoint(checkpoint, "cpu")
    assert payload["model_name"] == "qtro"
    records = json.loads((out / "train_qtro_L1_dE2.json").read_text(encoding="utf-8"))
    assert records[0]["training_data"] == str(data)
    assert [t["step"] for t in records[0]["validation_trace"]] == [0, 1]


def test_train_cli_refuses_a_dataset_without_a_train_split(tmp_path):
    data = tmp_path / "data"
    write_shards(data, n_conditions=2, horizons=(1,), M=512, seed=0, splits=("ns3_heldout",))
    result = subprocess.run(
        [sys.executable, str(TRAIN), "--data", str(data), "--out", str(tmp_path / "r"),
         "--steps", "1", "--seeds", "0", "--device", "cpu"],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and "split" in result.stderr


def test_depth_screen_dry_run_prints_the_four_variants(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train" / "screen_depth.py"),
         "--data", "d", "--out", str(tmp_path / "s"), "--dry-run"],
        cwd=tmp_path, capture_output=True, text=True, check=True)
    for variant in ("8x4", "4x8", "3x10", "2x16"):
        assert f"--variant {variant}" in result.stdout
    assert str(ROOT / "scripts" / "train" / "train.py") in result.stdout
