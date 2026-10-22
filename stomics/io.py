from pathlib import Path

import pandas as pd


def load_h5ad(path: str):
    """Load an .h5ad file and return an AnnData object."""
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover
        raise ImportError(  # pragma: no cover
            "anndata is required to load .h5ad files. Install with: "
            'pip install "stomics-py[anndata] @ git+https://github.com/ITGA3/stomics-py"'
        ) from exc
    return ad.read_h5ad(path)


def load_spatial_csv(path: str):
    """Load a spatial coordinates CSV and return a DataFrame with x/y columns."""
    df = pd.read_csv(path)
    expected = {"x", "y"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(
            f"Spatial coordinates CSV must contain columns 'x' and 'y'. Found: {list(df.columns)}"
        )
    return df


def attach_spatial(adata, coords_df) -> None:
    """Populate adata.obsm['spatial'] from a coordinates DataFrame in-place."""
    adata.obsm["spatial"] = coords_df[["x", "y"]].to_numpy(dtype=float)


def ensure_cache_dir(accession: str) -> Path:
    """Return (and create) the per-dataset cache directory under ~/.stomics/cache/."""
    cache = Path.home() / ".stomics" / "cache" / accession
    cache.mkdir(parents=True, exist_ok=True)
    return cache
