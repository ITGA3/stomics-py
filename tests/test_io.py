from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stomics.io import attach_spatial, ensure_cache_dir, load_spatial_csv

# ---------------------------------------------------------------------------
# load_spatial_csv
# ---------------------------------------------------------------------------


def test_load_spatial_csv_valid(tmp_path):
    csv_file = tmp_path / "coords.csv"
    csv_file.write_text("barcode,x,y\nAAA,10,20\nBBB,30,40\n")
    df = load_spatial_csv(str(csv_file))
    assert list(df.columns) == ["barcode", "x", "y"]
    assert len(df) == 2
    assert df["x"].tolist() == [10, 30]


def test_load_spatial_csv_missing_columns(tmp_path):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="must contain columns"):
        load_spatial_csv(str(csv_file))


def test_load_spatial_csv_only_x_missing_y(tmp_path):
    csv_file = tmp_path / "bad2.csv"
    csv_file.write_text("x,z\n1,2\n")
    with pytest.raises(ValueError):
        load_spatial_csv(str(csv_file))


# ---------------------------------------------------------------------------
# attach_spatial
# ---------------------------------------------------------------------------


def test_attach_spatial():
    import pandas as pd

    mock_adata = MagicMock()
    mock_adata.obsm = {}

    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    attach_spatial(mock_adata, df)

    arr = mock_adata.obsm["spatial"]
    assert arr.shape == (2, 2)
    assert arr[0, 0] == pytest.approx(1.0)
    assert arr[0, 1] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ensure_cache_dir
# ---------------------------------------------------------------------------


def test_ensure_cache_dir_creates_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache = ensure_cache_dir("STD0001")
    assert cache.exists()
    assert cache.is_dir()
    assert cache == tmp_path / ".stomics" / "cache" / "STD0001"


def test_ensure_cache_dir_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache1 = ensure_cache_dir("STD0002")
    cache2 = ensure_cache_dir("STD0002")
    assert cache1 == cache2
    assert cache1.exists()


# ---------------------------------------------------------------------------
# load_h5ad — ImportError path
# ---------------------------------------------------------------------------


def test_load_h5ad_import_error():
    import builtins
    import sys

    from stomics.io import load_h5ad

    real_anndata = sys.modules.pop("anndata", None)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anndata":
            raise ImportError("no anndata")
        return real_import(name, *args, **kwargs)

    try:
        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError, match="anndata is required"):
                load_h5ad("/some/path.h5ad")
    finally:
        if real_anndata is not None:
            sys.modules["anndata"] = real_anndata
