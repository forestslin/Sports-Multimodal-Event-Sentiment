"""Inspect the Echoes schema and game identifiers before matching modalities."""

from pathlib import Path
import pyarrow.ipc as ipc

path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "SN-echoes"
    / "whisper_v1_en"
    / "1.0.0"
    / "soccer_net_echoes_hf_dataset-train.arrow"
)
with path.open("rb") as handle:
    reader = ipc.open_stream(handle)
    table = reader.read_all()
print(table.schema)
print("rows", len(table))
print(table.slice(0, min(3, len(table))).to_pylist())
