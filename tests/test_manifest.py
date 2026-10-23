from pathlib import Path

from product_retrieval.data import load_manifest, validate_independent_splits


HEADER = (
    "image_path,product_code,model_family,physical_part_id,capture_session,"
    "split,orientation,source\n"
)


def test_manifest_detects_physical_sample_leakage(tmp_path, monkeypatch):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.jpg").touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        HEADER
        + "a.jpg,A,M,P1,S1,gallery,0,local\n"
        + "b.jpg,A,M,P1,S1,query,90,local\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    entries = load_manifest(manifest)
    warnings = validate_independent_splits(entries)

    assert len(entries) == 2
    assert "Possible leakage" in warnings[0]

