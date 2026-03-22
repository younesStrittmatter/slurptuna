from pathlib import Path

from slurptuna.slurm_backend import find_missing_chunks


def test_find_missing_chunks(tmp_path: Path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()

    (chunks / "chunk_00000.json").write_text("{}", encoding="utf-8")
    (chunks / "chunk_00002.json").write_text("{}", encoding="utf-8")

    missing = find_missing_chunks(chunks, 4)
    assert missing == [1, 3]
