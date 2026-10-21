import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
import responses as resp_lib

from stomics import StomicsClient
from stomics.models import Dataset, DatasetPage, File, VocabularyTerm

BASE = "https://api.stomics.io"


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def client():
    return StomicsClient(api_key="sk-test", base_url=BASE)


@pytest.fixture
def anon_client():
    return StomicsClient(base_url=BASE)


# ===========================================================================
# Initialisation
# ===========================================================================


def test_client_explicit_key():
    c = StomicsClient(api_key="sk-explicit")
    assert c.api_key == "sk-explicit"
    assert c._session.headers["Authorization"] == "Bearer sk-explicit"


def test_client_env_key(monkeypatch):
    monkeypatch.setenv("STOMICS_API_KEY", "sk-from-env")
    c = StomicsClient()
    assert c.api_key == "sk-from-env"
    assert c._session.headers["Authorization"] == "Bearer sk-from-env"


def test_client_no_key(monkeypatch):
    monkeypatch.delenv("STOMICS_API_KEY", raising=False)
    c = StomicsClient()
    assert c.api_key is None
    assert "Authorization" not in c._session.headers


def test_client_base_url_trailing_slash():
    c = StomicsClient(api_key="k", base_url="https://example.com/")
    assert c.base_url == "https://example.com"


# ===========================================================================
# search
# ===========================================================================


@resp_lib.activate
def test_search_no_filters(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets",
        json={
            "items": [
                {
                    "accession": "STD0001",
                    "title": "Brain Atlas",
                    "status": "published",
                    "download_count": 10,
                }
            ],
            "total": 1,
            "page": 1,
            "per_page": 20,
        },
    )
    results = client.search()
    assert len(results) == 1
    assert results[0].accession == "STD0001"
    assert results[0].title == "Brain Atlas"
    assert isinstance(results[0], Dataset)


@resp_lib.activate
def test_search_with_all_filters(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets",
        json={"items": [], "total": 0, "page": 2, "per_page": 5},
    )
    results = client.search(
        query="brain",
        organism="1",
        tissue="2",
        disease="3",
        technology="4",
        page=2,
        per_page=5,
    )
    assert results == []
    assert isinstance(results, DatasetPage)
    assert (results.total, results.page, results.per_page) == (0, 2, 5)
    req = resp_lib.calls[0].request
    assert "q=brain" in req.url
    assert "organism_ids=1" in req.url
    assert "tissue_ids=2" in req.url
    assert "disease_ids=3" in req.url
    assert "technology_ids=4" in req.url
    assert "page=2" in req.url
    assert "per_page=5" in req.url


@resp_lib.activate
def test_search_sends_several_ids_as_repeated_parameters(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets",
        json={"items": [], "total": 0, "page": 1, "per_page": 20},
    )
    client.search(organism="1, 2", tissue=[3, "4"], technology=5)
    url = resp_lib.calls[0].request.url
    assert "organism_ids=1&organism_ids=2" in url
    assert "tissue_ids=3&tissue_ids=4" in url
    assert "technology_ids=5" in url
    assert "%2C" not in url


@resp_lib.activate
def test_search_resolves_names_through_the_vocabulary(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/vocabulary/terms",
        json=[
            {"id": 7, "ontology": "NCBITaxon", "term_id": "9606", "label": "Homo sapiens"},
            {
                "id": 8,
                "ontology": "NCBITaxon",
                "term_id": "0",
                "label": "Homo sapiens neanderthalensis",
            },
        ],
    )
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/vocabulary/technologies",
        json=[{"id": 3, "name": "10x Visium", "vendor": "10x Genomics"}],
    )
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets",
        json={"items": [], "total": 0, "page": 1, "per_page": 20},
    )
    client.search(organism="homo sapiens", technology="10X Visium")
    assert "kind=organism" in resp_lib.calls[0].request.url
    assert "q=homo+sapiens" in resp_lib.calls[0].request.url
    url = resp_lib.calls[2].request.url
    assert "organism_ids=7" in url
    assert "organism_ids=8" not in url
    assert "technology_ids=3" in url


@resp_lib.activate
def test_search_unknown_name_raises_before_searching(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/api/v1/vocabulary/terms", json=[])
    with pytest.raises(ValueError, match="unknown tissue 'spleen'"):
        client.search(tissue="spleen")
    assert len(resp_lib.calls) == 1


@resp_lib.activate
def test_search_returns_multiple_datasets(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets",
        json={
            "items": [
                {"accession": "STD0001", "title": "A", "status": "published", "download_count": 0},
                {"accession": "STD0002", "title": "B", "status": "published", "download_count": 5},
            ],
            "total": 2,
            "page": 1,
            "per_page": 20,
        },
    )
    results = client.search()
    assert len(results) == 2
    assert results[1].accession == "STD0002"


@resp_lib.activate
def test_search_http_error(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/api/v1/datasets", status=500)
    with pytest.raises(requests.HTTPError):
        client.search()


# ===========================================================================
# get_dataset
# ===========================================================================


@resp_lib.activate
def test_get_dataset(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001",
        json={
            "accession": "STD0001",
            "title": "Brain Atlas",
            "abstract": "A spatial brain atlas.",
            "status": "published",
            "download_count": 42,
            "published_at": "2025-01-01T00:00:00Z",
            "created_at": "2024-12-01T00:00:00Z",
        },
    )
    ds = client.get_dataset("STD0001")
    assert ds.accession == "STD0001"
    assert ds.abstract == "A spatial brain atlas."
    assert ds.download_count == 42


@resp_lib.activate
def test_get_dataset_not_found(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/api/v1/datasets/STD9999", status=404)
    with pytest.raises(requests.HTTPError):
        client.get_dataset("STD9999")


# ===========================================================================
# list_files
# ===========================================================================


@resp_lib.activate
def test_list_files(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files",
        json=[
            {
                "id": 1,
                "dataset_id": 1,
                "filename": "matrix.h5ad",
                "file_type": "expression_matrix",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "size_bytes": 2048,
                "download_url": "https://s3.example.com/presigned/matrix.h5ad",
            },
            {
                "id": 2,
                "dataset_id": 1,
                "filename": "coords.csv",
                "file_type": "spatial_coordinates",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "download_url": "https://s3.example.com/presigned/coords.csv",
            },
        ],
    )
    files = client.list_files("STD0001")
    assert len(files) == 2
    assert files[0].filename == "matrix.h5ad"
    assert files[0].download_url == "https://s3.example.com/presigned/matrix.h5ad"
    assert isinstance(files[0], File)


@resp_lib.activate
def test_list_files_empty(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/api/v1/datasets/STD0001/files", json=[])
    assert client.list_files("STD0001") == []


# ===========================================================================
# download_file
# ===========================================================================


@resp_lib.activate
def test_download_file_no_progress(client, tmp_path):
    dest = str(tmp_path / "file.h5ad")
    presigned = "https://s3.example.com/file.h5ad?sig=abc"

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files/1/download",
        json={"url": presigned},
    )
    resp_lib.add(resp_lib.GET, presigned, body=b"fake-content")

    client.download_file("STD0001", 1, dest, show_progress=False)
    assert Path(dest).read_bytes() == b"fake-content"


@resp_lib.activate
def test_download_file_creates_parent_dirs(client, tmp_path):
    dest = str(tmp_path / "deep" / "nested" / "file.h5ad")
    presigned = "https://s3.example.com/file.h5ad"

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files/1/download",
        json={"url": presigned},
    )
    resp_lib.add(resp_lib.GET, presigned, body=b"data")

    client.download_file("STD0001", 1, dest, show_progress=False)
    assert Path(dest).exists()


@resp_lib.activate
def test_download_file_with_tqdm(client, tmp_path):
    """show_progress=True falls back gracefully when tqdm is available."""
    dest = str(tmp_path / "prog.h5ad")
    presigned = "https://s3.example.com/prog.h5ad"

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files/2/download",
        json={"url": presigned},
    )
    resp_lib.add(
        resp_lib.GET,
        presigned,
        body=b"progress-data",
        headers={"content-length": "13"},
    )

    mock_tqdm = MagicMock()
    mock_tqdm.return_value.__enter__ = MagicMock(return_value=mock_tqdm.return_value)
    mock_tqdm.return_value.__exit__ = MagicMock(return_value=False)

    with patch("stomics.client.tqdm", mock_tqdm, create=True):
        # Import tqdm so it is importable during the test
        import tqdm as tqdm_module

        with patch.dict("sys.modules", {"tqdm": tqdm_module}):
            client.download_file("STD0001", 2, dest, show_progress=True)

    assert Path(dest).exists()


@resp_lib.activate
def test_download_file_progress_tqdm_import_error(client, tmp_path):
    """Falls back to no-progress download when tqdm is not installed."""
    dest = str(tmp_path / "notqdm.h5ad")
    presigned = "https://s3.example.com/notqdm.h5ad"

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files/3/download",
        json={"url": presigned},
    )
    resp_lib.add(resp_lib.GET, presigned, body=b"fallback-data")

    import builtins

    real_import = builtins.__import__

    def fail_tqdm(name, *args, **kwargs):
        if name == "tqdm":
            raise ImportError("no tqdm")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fail_tqdm):
        client.download_file("STD0001", 3, dest, show_progress=True)

    assert Path(dest).read_bytes() == b"fallback-data"


@resp_lib.activate
def test_download_file_http_error_on_presigned(client, tmp_path):
    dest = str(tmp_path / "fail.h5ad")
    presigned = "https://s3.example.com/fail.h5ad"

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files/9/download",
        json={"url": presigned},
    )
    resp_lib.add(resp_lib.GET, presigned, status=403)

    with pytest.raises(requests.HTTPError):
        client.download_file("STD0001", 9, dest, show_progress=False)


# ===========================================================================
# get_file / trigger_metadata_extraction (Feature 24)
# ===========================================================================


@resp_lib.activate
def test_get_file_returns_file_with_metadata(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files/7",
        json={
            "id": 7,
            "dataset_id": 1,
            "filename": "matrix.h5ad",
            "file_type": "expression_matrix",
            "upload_status": "complete",
            "created_at": "2025-01-01T00:00:00Z",
            "file_metadata": {
                "format": "h5ad",
                "n_obs": 3522,
                "n_vars": 18000,
                "has_spatial_obsm": True,
                "has_umap_obsm": False,
                "x_min": 204.5,
                "x_max": 6812.0,
                "y_min": 117.3,
                "y_max": 7984.5,
            },
        },
    )
    file = client.get_file("STD0001", 7)
    assert isinstance(file, File)
    assert file.id == 7
    assert file.file_metadata is not None
    assert file.file_metadata.n_obs == 3522
    assert file.file_metadata.n_vars == 18000
    assert file.file_metadata.has_spatial_obsm is True
    assert file.file_metadata.x_min == 204.5


@resp_lib.activate
def test_get_file_no_metadata_returns_none(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files/8",
        json={
            "id": 8,
            "dataset_id": 1,
            "filename": "coords.csv",
            "file_type": "spatial_coordinates",
            "upload_status": "complete",
            "created_at": "2025-01-01T00:00:00Z",
            "file_metadata": None,
        },
    )
    file = client.get_file("STD0001", 8)
    assert file.file_metadata is None


@resp_lib.activate
def test_get_file_parse_error_metadata(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files/9",
        json={
            "id": 9,
            "dataset_id": 1,
            "filename": "bad.h5ad",
            "file_type": "expression_matrix",
            "upload_status": "complete",
            "created_at": "2025-01-01T00:00:00Z",
            "file_metadata": {"parse_error": True, "error": "Could not open file"},
        },
    )
    file = client.get_file("STD0001", 9)
    assert file.file_metadata is not None
    assert file.file_metadata.parse_error is True
    assert file.file_metadata.error == "Could not open file"


@resp_lib.activate
def test_trigger_metadata_extraction(client):
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0001/files/7/extract",
        json={
            "id": 7,
            "dataset_id": 1,
            "filename": "matrix.h5ad",
            "file_type": "expression_matrix",
            "upload_status": "complete",
            "created_at": "2025-01-01T00:00:00Z",
            "file_metadata": None,
        },
        status=202,
    )
    file = client.trigger_metadata_extraction("STD0001", 7)
    assert isinstance(file, File)
    assert file.id == 7


@resp_lib.activate
def test_trigger_metadata_extraction_http_error(client):
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0001/files/99/extract",
        status=404,
    )
    with pytest.raises(requests.HTTPError):
        client.trigger_metadata_extraction("STD0001", 99)


# ===========================================================================
# get_expression
# ===========================================================================


@resp_lib.activate
def test_get_expression(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/expression",
        json={
            "spots": [
                {"x": 1, "y": 2, "expression": 3.5},
                {"x": 3, "y": 4, "expression": 0.0},
            ]
        },
    )
    df = client.get_expression("STD0001", gene="BRCA1")
    assert len(df) == 2
    assert list(df.columns) == ["x", "y", "expression"]


@resp_lib.activate
def test_get_expression_with_sample_id(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/expression",
        json={"spots": []},
    )
    df = client.get_expression("STD0001", gene="TP53", sample_id="sample_1")
    req = resp_lib.calls[0].request
    assert "sample_id=sample_1" in req.url
    assert len(df) == 0


@resp_lib.activate
def test_get_expression_empty_spots(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/expression",
        json={"spots": []},
    )
    import pandas as pd

    df = client.get_expression("STD0001", gene="NOTEXIST")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ===========================================================================
# load_anndata
# ===========================================================================


def test_load_anndata_missing_anndata_import(client, tmp_path, monkeypatch):
    """Raises ImportError when anndata is not installed."""
    import builtins

    real_import = builtins.__import__

    def fail_anndata(name, *args, **kwargs):
        if name == "anndata":
            raise ImportError("no anndata")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fail_anndata):
        with pytest.raises(ImportError, match="anndata is required"):
            client.load_anndata("STD0001")


@resp_lib.activate
def test_load_anndata_no_matrix_raises(client, tmp_path, monkeypatch):
    """Raises ValueError when no expression matrix file exists."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files",
        json=[],
    )

    mock_ad = MagicMock()
    with patch.dict("sys.modules", {"anndata": mock_ad}):
        with pytest.raises(ValueError, match="No expression matrix"):
            client.load_anndata("STD0001")


@resp_lib.activate
def test_load_anndata_no_matrix_for_sample(client, tmp_path, monkeypatch):
    """Raises ValueError when no expression matrix for the requested sample_id."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files",
        json=[
            {
                "id": 1,
                "dataset_id": 1,
                "filename": "matrix.h5ad",
                "file_type": "expression_matrix",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "sample_id": 99,
            }
        ],
    )

    mock_ad = MagicMock()
    with patch.dict("sys.modules", {"anndata": mock_ad}):
        with pytest.raises(ValueError, match="No expression matrix"):
            client.load_anndata("STD0001", sample_id="1")


@resp_lib.activate
def test_load_anndata_uses_cached_file(client, tmp_path, monkeypatch):
    """Skips download if the file is already in cache."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cache_dir = tmp_path / ".stomics" / "cache" / "STD0001"
    cache_dir.mkdir(parents=True)
    (cache_dir / "matrix.h5ad").write_bytes(b"cached")

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0001/files",
        json=[
            {
                "id": 1,
                "dataset_id": 1,
                "filename": "matrix.h5ad",
                "file_type": "expression_matrix",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "download_url": "https://s3.example.com/matrix.h5ad",
            }
        ],
    )

    mock_adata = MagicMock()
    mock_ad = MagicMock()
    mock_ad.read_h5ad.return_value = mock_adata

    with patch.dict("sys.modules", {"anndata": mock_ad}):
        result = client.load_anndata("STD0001")

    assert result is mock_adata
    assert len(resp_lib.calls) == 1  # only list_files, no download


@resp_lib.activate
def test_load_anndata_downloads_via_download_url(client, tmp_path, monkeypatch):
    """Downloads matrix using download_url when not cached."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    presigned = "https://s3.example.com/matrix.h5ad"
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0002/files",
        json=[
            {
                "id": 10,
                "dataset_id": 2,
                "filename": "matrix.h5ad",
                "file_type": "expression_matrix",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "download_url": presigned,
            }
        ],
    )
    resp_lib.add(resp_lib.GET, presigned, body=b"h5ad-data")

    mock_adata = MagicMock()
    mock_ad = MagicMock()
    mock_ad.read_h5ad.return_value = mock_adata

    with patch.dict("sys.modules", {"anndata": mock_ad}):
        result = client.load_anndata("STD0002")

    assert result is mock_adata


@resp_lib.activate
def test_load_anndata_downloads_via_download_endpoint(client, tmp_path, monkeypatch):
    """Falls back to download endpoint when download_url is not present."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    presigned = "https://s3.example.com/fallback.h5ad"
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0003/files",
        json=[
            {
                "id": 20,
                "dataset_id": 3,
                "filename": "matrix.h5ad",
                "file_type": "expression_matrix",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
    )
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0003/files/20/download",
        json={"url": presigned},
    )
    resp_lib.add(resp_lib.GET, presigned, body=b"fallback")

    mock_adata = MagicMock()
    mock_ad = MagicMock()
    mock_ad.read_h5ad.return_value = mock_adata

    with patch.dict("sys.modules", {"anndata": mock_ad}):
        result = client.load_anndata("STD0003")

    assert result is mock_adata


@resp_lib.activate
def test_load_anndata_with_spatial_coords_via_download_url(client, tmp_path, monkeypatch):
    """Attaches spatial coordinates when coords file has download_url."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    matrix_url = "https://s3.example.com/matrix.h5ad"
    coords_url = "https://s3.example.com/coords.csv"

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0004/files",
        json=[
            {
                "id": 30,
                "dataset_id": 4,
                "filename": "matrix.h5ad",
                "file_type": "expression_matrix",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "download_url": matrix_url,
            },
            {
                "id": 31,
                "dataset_id": 4,
                "filename": "coords.csv",
                "file_type": "spatial_coordinates",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "download_url": coords_url,
            },
        ],
    )
    resp_lib.add(resp_lib.GET, matrix_url, body=b"h5ad")
    resp_lib.add(resp_lib.GET, coords_url, body=b"x,y\n1,2\n3,4\n")

    mock_adata = MagicMock()
    mock_adata.obsm = {}
    mock_ad = MagicMock()
    mock_ad.read_h5ad.return_value = mock_adata

    import pandas as pd

    with patch.dict("sys.modules", {"anndata": mock_ad}):
        with patch("stomics.io.pd.read_csv", return_value=pd.DataFrame({"x": [1, 3], "y": [2, 4]})):
            result = client.load_anndata("STD0004")

    assert result is mock_adata
    assert "spatial" in mock_adata.obsm


@resp_lib.activate
def test_load_anndata_coords_download_fallback(client, tmp_path, monkeypatch):
    """Downloads coords via endpoint when download_url absent."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    matrix_url = "https://s3.example.com/m.h5ad"
    coords_presigned = "https://s3.example.com/c.csv"

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0005/files",
        json=[
            {
                "id": 40,
                "dataset_id": 5,
                "filename": "matrix.h5ad",
                "file_type": "expression_matrix",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "download_url": matrix_url,
            },
            {
                "id": 41,
                "dataset_id": 5,
                "filename": "coords.csv",
                "file_type": "spatial_coordinates",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
            },
        ],
    )
    resp_lib.add(resp_lib.GET, matrix_url, body=b"h5ad")
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0005/files/41/download",
        json={"url": coords_presigned},
    )
    resp_lib.add(resp_lib.GET, coords_presigned, body=b"x,y\n5,6\n")

    mock_adata = MagicMock()
    mock_adata.obsm = {}
    mock_ad = MagicMock()
    mock_ad.read_h5ad.return_value = mock_adata

    import pandas as pd

    with patch.dict("sys.modules", {"anndata": mock_ad}):
        with patch("stomics.io.pd.read_csv", return_value=pd.DataFrame({"x": [5], "y": [6]})):
            result = client.load_anndata("STD0005")

    assert result is mock_adata


@resp_lib.activate
def test_load_anndata_bad_coords_silently_ignored(client, tmp_path, monkeypatch):
    """Malformed coords file is silently skipped (adata returned without obsm['spatial'])."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    matrix_url = "https://s3.example.com/mx.h5ad"
    coords_url = "https://s3.example.com/bad.csv"

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0006/files",
        json=[
            {
                "id": 50,
                "dataset_id": 6,
                "filename": "matrix.h5ad",
                "file_type": "expression_matrix",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "download_url": matrix_url,
            },
            {
                "id": 51,
                "dataset_id": 6,
                "filename": "bad.csv",
                "file_type": "spatial_coordinates",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "download_url": coords_url,
            },
        ],
    )
    resp_lib.add(resp_lib.GET, matrix_url, body=b"h5ad")
    resp_lib.add(resp_lib.GET, coords_url, body=b"a,b\n1,2\n")

    mock_adata = MagicMock()
    mock_adata.obsm = {}
    mock_ad = MagicMock()
    mock_ad.read_h5ad.return_value = mock_adata

    with patch.dict("sys.modules", {"anndata": mock_ad}):
        result = client.load_anndata("STD0006")

    assert result is mock_adata
    assert "spatial" not in mock_adata.obsm


@resp_lib.activate
def test_load_anndata_uses_cached_coords(client, tmp_path, monkeypatch):
    """Skips coords download if already cached."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cache_dir = tmp_path / ".stomics" / "cache" / "STD0007"
    cache_dir.mkdir(parents=True)
    (cache_dir / "matrix.h5ad").write_bytes(b"cached")
    (cache_dir / "coords.csv").write_bytes(b"x,y\n1,2\n")

    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/datasets/STD0007/files",
        json=[
            {
                "id": 60,
                "dataset_id": 7,
                "filename": "matrix.h5ad",
                "file_type": "expression_matrix",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "download_url": "https://s3.example.com/m7.h5ad",
            },
            {
                "id": 61,
                "dataset_id": 7,
                "filename": "coords.csv",
                "file_type": "spatial_coordinates",
                "upload_status": "complete",
                "created_at": "2025-01-01T00:00:00Z",
                "download_url": "https://s3.example.com/c7.csv",
            },
        ],
    )

    mock_adata = MagicMock()
    mock_adata.obsm = {}
    mock_ad = MagicMock()
    mock_ad.read_h5ad.return_value = mock_adata

    import pandas as pd

    with patch.dict("sys.modules", {"anndata": mock_ad}):
        with patch("stomics.io.pd.read_csv", return_value=pd.DataFrame({"x": [1], "y": [2]})):
            client.load_anndata("STD0007")

    # Only list_files was called; no S3 downloads
    assert len(resp_lib.calls) == 1


# ===========================================================================
# _build_* static methods
# ===========================================================================


def test_build_dataset_ignores_unknown_fields():
    data = {"accession": "STD0001", "title": "T", "unknown_field": "ignored"}
    ds = StomicsClient._build_dataset(data)
    assert ds.accession == "STD0001"
    assert not hasattr(ds, "unknown_field")


def test_build_file_ignores_unknown_fields():
    data = {
        "id": 1,
        "dataset_id": 1,
        "filename": "f.h5ad",
        "file_type": "expression_matrix",
        "upload_status": "complete",
        "created_at": "2025-01-01T00:00:00Z",
        "extra": "ignored",
    }
    f = StomicsClient._build_file(data)
    assert f.id == 1
    assert not hasattr(f, "extra")


def test_build_sample_ignores_unknown_fields():
    data = {"id": 1, "dataset_id": 2, "name": "S1", "other": "x"}
    s = StomicsClient._build_sample(data)
    assert s.name == "S1"
    assert not hasattr(s, "other")


# ===========================================================================
# _get error propagation
# ===========================================================================


@resp_lib.activate
def test_get_raises_http_error(client):
    resp_lib.add(resp_lib.GET, f"{BASE}/api/v1/datasets/STD0001", status=401)
    with pytest.raises(requests.HTTPError):
        client._get("/api/v1/datasets/STD0001")


# ===========================================================================
# QC Report methods
# ===========================================================================


@resp_lib.activate
def test_get_qc_reports(client):
    payload = {
        "reports": [
            {
                "id": 1,
                "dataset_accession": "STD0001",
                "status": "passed",
                "created_at": "2024-01-01T00:00:00Z",
                "sample_id": None,
                "report_json": None,
                "completed_at": "2024-01-01T00:01:00Z",
                "curator_override": None,
            }
        ],
        "dataset_qc_status": "passed",
    }
    resp_lib.add(resp_lib.GET, f"{BASE}/api/v1/datasets/STD0001/qc", json=payload)
    result = client.get_qc_reports("STD0001")
    assert result.dataset_qc_status == "passed"
    assert len(result.reports) == 1
    assert result.reports[0].id == 1
    assert result.reports[0].status == "passed"


@resp_lib.activate
def test_get_qc_reports_empty(client):
    payload = {"reports": [], "dataset_qc_status": "not_run"}
    resp_lib.add(resp_lib.GET, f"{BASE}/api/v1/datasets/STD0001/qc", json=payload)
    result = client.get_qc_reports("STD0001")
    assert result.dataset_qc_status == "not_run"
    assert result.reports == []


@resp_lib.activate
def test_trigger_qc(client):
    payload = {
        "id": 2,
        "dataset_accession": "STD0001",
        "status": "pending",
        "created_at": "2024-01-01T00:00:00Z",
        "sample_id": None,
        "report_json": None,
        "completed_at": None,
        "curator_override": None,
    }
    resp_lib.add(resp_lib.POST, f"{BASE}/api/v1/datasets/STD0001/qc/run", json=payload, status=202)
    report = client.trigger_qc("STD0001")
    assert report.id == 2
    assert report.status == "pending"


# ===========================================================================
# run_differential_expression
# ===========================================================================

_DE_RESPONSE = {
    "genes": [
        {
            "gene": "MYH7",
            "mean_a": 60.0,
            "mean_b": 5.0,
            "log2fc": 3.58,
            "pvalue": 0.001,
            "padj": 0.01,
        },
        {
            "gene": "GAPDH",
            "mean_a": 35.0,
            "mean_b": 30.0,
            "log2fc": 0.22,
            "pvalue": 0.5,
            "padj": 0.8,
        },
    ],
    "n_spots_a": 10,
    "n_spots_b": 50,
    "method": "wilcoxon",
    "subsampled": False,
}

_DE_URL = f"{BASE}/api/v1/datasets/STD0001/samples/1/differential-expression"


@resp_lib.activate
def test_run_de_rest_region_b(client):
    from stomics.models import DEGeneResult, DifferentialExpressionResult

    resp_lib.add(resp_lib.POST, _DE_URL, json=_DE_RESPONSE, status=200)
    region_a = ["BARCODE_r00_c01", "BARCODE_r00_c02", "BARCODE_r01_c01"]
    result = client.run_differential_expression("STD0001", 1, region_a, region_b="rest")
    assert isinstance(result, DifferentialExpressionResult)
    assert result.n_spots_a == 10
    assert result.n_spots_b == 50
    assert result.method == "wilcoxon"
    assert result.subsampled is False
    assert len(result.genes) == 2
    assert isinstance(result.genes[0], DEGeneResult)
    assert result.genes[0].gene == "MYH7"
    assert result.genes[0].log2fc == 3.58
    assert result.genes[0].padj == 0.01


@resp_lib.activate
def test_run_de_explicit_region_b(client):
    resp_lib.add(resp_lib.POST, _DE_URL, json=_DE_RESPONSE, status=200)
    region_a = ["BARCODE_r00_c01"]
    region_b = ["BARCODE_r10_c10", "BARCODE_r10_c11"]
    client.run_differential_expression("STD0001", 1, region_a, region_b=region_b)
    import json as _json

    body = _json.loads(resp_lib.calls[0].request.body)
    assert body["region_b"] == region_b


@resp_lib.activate
def test_run_de_ttest_method(client):
    payload = dict(_DE_RESPONSE)
    payload["method"] = "t-test"
    resp_lib.add(resp_lib.POST, _DE_URL, json=payload, status=200)
    result = client.run_differential_expression("STD0001", 1, ["BARCODE_r00_c01"], method="t-test")
    assert result.method == "t-test"


@resp_lib.activate
def test_run_de_n_top_genes_passed_in_payload(client):
    resp_lib.add(resp_lib.POST, _DE_URL, json=_DE_RESPONSE, status=200)
    client.run_differential_expression("STD0001", 1, ["BARCODE_r00_c01"], n_top_genes=10)
    import json as _json

    body = _json.loads(resp_lib.calls[0].request.body)
    assert body["n_top_genes"] == 10


@resp_lib.activate
def test_run_de_empty_genes_response(client):
    payload = {
        "genes": [],
        "n_spots_a": 5,
        "n_spots_b": 100,
        "method": "wilcoxon",
        "subsampled": False,
    }
    resp_lib.add(resp_lib.POST, _DE_URL, json=payload, status=200)
    result = client.run_differential_expression("STD0001", 1, ["BARCODE_r00_c01"])
    assert result.genes == []
    assert result.n_spots_a == 5


@resp_lib.activate
def test_run_de_subsampled_flag(client):
    payload = dict(_DE_RESPONSE)
    payload["subsampled"] = True
    resp_lib.add(resp_lib.POST, _DE_URL, json=payload, status=200)
    result = client.run_differential_expression("STD0001", 1, ["BARCODE_r00_c01"])
    assert result.subsampled is True


@resp_lib.activate
def test_run_de_401_raises(client):
    resp_lib.add(resp_lib.POST, _DE_URL, status=401)
    with pytest.raises(requests.HTTPError):
        client.run_differential_expression("STD0001", 1, ["BARCODE_r00_c01"])


@resp_lib.activate
def test_run_de_429_raises(client):
    resp_lib.add(resp_lib.POST, _DE_URL, status=429)
    with pytest.raises(requests.HTTPError):
        client.run_differential_expression("STD0001", 1, ["BARCODE_r00_c01"])


# ===========================================================================
# get_gene_sets (Feature 16)
# ===========================================================================

_GENE_SETS_URL = f"{BASE}/api/v1/gene-sets"

_GENE_SETS_RESPONSE = {
    "gene_sets": [
        {
            "id": 1,
            "name": "HALLMARK_INFLAMMATORY_RESPONSE",
            "source": "MSigDB_Hallmark",
            "gene_symbols": ["PTPRC", "CD68", "CD4"],
            "description": "Inflammatory response",
        },
        {
            "id": 2,
            "name": "GO_MYELINATION",
            "source": "GO_BP",
            "gene_symbols": ["MBP", "PLP1"],
            "description": None,
        },
    ],
    "total": 2,
}


@resp_lib.activate
def test_get_gene_sets_basic(client):
    from stomics.models import GeneSet, GeneSetList

    resp_lib.add(resp_lib.GET, _GENE_SETS_URL, json=_GENE_SETS_RESPONSE, status=200)
    result = client.get_gene_sets()
    assert isinstance(result, GeneSetList)
    assert result.total == 2
    assert len(result.gene_sets) == 2
    assert isinstance(result.gene_sets[0], GeneSet)
    gs = result.gene_sets[0]
    assert gs.id == 1
    assert gs.name == "HALLMARK_INFLAMMATORY_RESPONSE"
    assert gs.source == "MSigDB_Hallmark"
    assert isinstance(gs.gene_symbols, list)
    assert "PTPRC" in gs.gene_symbols


@resp_lib.activate
def test_get_gene_sets_sends_q_param(client):
    resp_lib.add(resp_lib.GET, _GENE_SETS_URL, json=_GENE_SETS_RESPONSE, status=200)
    client.get_gene_sets(q="hallmark")
    assert "q=hallmark" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_gene_sets_sends_source_param(client):
    resp_lib.add(resp_lib.GET, _GENE_SETS_URL, json=_GENE_SETS_RESPONSE, status=200)
    client.get_gene_sets(source="GO_BP")
    assert "source=GO_BP" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_gene_sets_sends_limit_offset(client):
    resp_lib.add(resp_lib.GET, _GENE_SETS_URL, json=_GENE_SETS_RESPONSE, status=200)
    client.get_gene_sets(limit=25, offset=50)
    url = resp_lib.calls[0].request.url
    assert "limit=25" in url
    assert "offset=50" in url


@resp_lib.activate
def test_get_gene_sets_empty_response(client):
    resp_lib.add(resp_lib.GET, _GENE_SETS_URL, json={"gene_sets": [], "total": 0}, status=200)
    result = client.get_gene_sets()
    assert result.total == 0
    assert result.gene_sets == []


@resp_lib.activate
def test_get_gene_sets_500_raises(client):
    resp_lib.add(resp_lib.GET, _GENE_SETS_URL, status=500)
    with pytest.raises(requests.HTTPError):
        client.get_gene_sets()


# ===========================================================================
# compute_module_score (Feature 16)
# ===========================================================================

_MODULE_SCORE_URL = f"{BASE}/api/v1/datasets/STD0001/samples/1/module-score"

_MODULE_SCORE_RESPONSE = {
    "spots": [
        {"barcode": "BARCODE_r00_c01", "x": 1.5, "y": 2.3, "score": 1.2},
        {"barcode": "BARCODE_r00_c02", "x": 3.0, "y": 4.6, "score": -0.5},
    ],
    "method": "mean_normalized",
    "gene_set_name": "HALLMARK_INFLAMMATORY_RESPONSE",
    "n_genes_total": 3,
    "n_genes_matched": 2,
    "matched_genes": ["PTPRC", "CD68"],
}


@resp_lib.activate
def test_compute_module_score_with_gene_set_id(client):
    from stomics.models import ModuleScoreResult, ModuleScoreSpot

    resp_lib.add(resp_lib.POST, _MODULE_SCORE_URL, json=_MODULE_SCORE_RESPONSE, status=200)
    result = client.compute_module_score("STD0001", 1, gene_set_id=1)
    assert isinstance(result, ModuleScoreResult)
    assert result.method == "mean_normalized"
    assert result.gene_set_name == "HALLMARK_INFLAMMATORY_RESPONSE"
    assert result.n_genes_total == 3
    assert result.n_genes_matched == 2
    assert result.matched_genes == ["PTPRC", "CD68"]
    assert len(result.spots) == 2
    assert isinstance(result.spots[0], ModuleScoreSpot)
    spot = result.spots[0]
    assert spot.barcode == "BARCODE_r00_c01"
    assert spot.score == 1.2
    import json as _json

    body = _json.loads(resp_lib.calls[0].request.body)
    assert body["gene_set_id"] == 1
    assert "custom_genes" not in body


@resp_lib.activate
def test_compute_module_score_with_custom_genes(client):
    resp_lib.add(resp_lib.POST, _MODULE_SCORE_URL, json=_MODULE_SCORE_RESPONSE, status=200)
    result = client.compute_module_score("STD0001", 1, custom_genes=["MKI67", "TOP2A"])
    import json as _json

    body = _json.loads(resp_lib.calls[0].request.body)
    assert body["custom_genes"] == ["MKI67", "TOP2A"]
    assert "gene_set_id" not in body
    assert len(result.spots) == 2


def test_compute_module_score_no_params_raises_valueerror(client):
    with pytest.raises(ValueError, match="Exactly one"):
        client.compute_module_score("STD0001", 1)


def test_compute_module_score_both_params_raises_valueerror(client):
    with pytest.raises(ValueError, match="not both"):
        client.compute_module_score("STD0001", 1, gene_set_id=1, custom_genes=["GAPDH"])


@resp_lib.activate
def test_compute_module_score_404_raises(client):
    resp_lib.add(resp_lib.POST, _MODULE_SCORE_URL, status=404)
    with pytest.raises(requests.HTTPError):
        client.compute_module_score("STD0001", 1, gene_set_id=99999)


@resp_lib.activate
def test_compute_module_score_422_raises(client):
    resp_lib.add(resp_lib.POST, _MODULE_SCORE_URL, status=422)
    with pytest.raises(requests.HTTPError):
        client.compute_module_score("STD0001", 1, gene_set_id=1)


@resp_lib.activate
def test_compute_module_score_spots_have_correct_types(client):
    resp_lib.add(resp_lib.POST, _MODULE_SCORE_URL, json=_MODULE_SCORE_RESPONSE, status=200)
    result = client.compute_module_score("STD0001", 1, gene_set_id=1)
    for spot in result.spots:
        assert isinstance(spot.barcode, str)
        assert isinstance(spot.x, float)
        assert isinstance(spot.y, float)
        assert isinstance(spot.score, float)


# ===========================================================================
# launch_to_platform (Feature 20)
# ===========================================================================

_LAUNCH_URL = f"{BASE}/api/v1/datasets/STD0001/launch-bundle"
_ARIA2C_URL = f"{BASE}/api/v1/datasets/STD0001/aria2c-manifest"

_TERRA_LAUNCH_RESPONSE = {
    "launch_url": "https://app.terra.bio/#import-data?url=https%3A%2F%2Fs3.example.com%2Fmanifest.json",
    "platform": "terra",
}

_GALAXY_LAUNCH_RESPONSE = {
    "launch_url": "https://usegalaxy.eu/?tool_id=upload1&URL=https%3A%2F%2Fs3.example.com%2Ff.csv",
    "platform": "galaxy",
}

_SEQERA_LAUNCH_RESPONSE = {
    "launch_url": "https://cloud.seqera.io/runs/run-abc123",
    "platform": "seqera",
}


@resp_lib.activate
def test_launch_to_platform_terra_returns_launch_result(client):
    resp_lib.add(resp_lib.POST, _LAUNCH_URL, json=_TERRA_LAUNCH_RESPONSE, status=200)
    result = client.launch_to_platform("STD0001", platform="terra")
    from stomics.models import LaunchResult

    assert isinstance(result, LaunchResult)
    assert result.platform == "terra"
    assert "terra.bio" in result.launch_url


@resp_lib.activate
def test_launch_to_platform_terra_posts_correct_body(client):
    resp_lib.add(resp_lib.POST, _LAUNCH_URL, json=_TERRA_LAUNCH_RESPONSE, status=200)
    client.launch_to_platform(
        "STD0001", platform="terra", file_types=["expression_matrix", "spatial_coordinates"]
    )
    body = json.loads(resp_lib.calls[0].request.body)
    assert body["platform"] == "terra"
    assert body["file_types"] == ["expression_matrix", "spatial_coordinates"]


@resp_lib.activate
def test_launch_to_platform_omits_file_types_when_none(client):
    resp_lib.add(resp_lib.POST, _LAUNCH_URL, json=_TERRA_LAUNCH_RESPONSE, status=200)
    client.launch_to_platform("STD0001", platform="terra")
    body = json.loads(resp_lib.calls[0].request.body)
    assert "file_types" not in body


@resp_lib.activate
def test_launch_to_platform_sends_workflow_id_for_seqera(client):
    resp_lib.add(resp_lib.POST, _LAUNCH_URL, json=_SEQERA_LAUNCH_RESPONSE, status=200)
    client.launch_to_platform("STD0001", platform="seqera", workflow_id="wf-123")
    body = json.loads(resp_lib.calls[0].request.body)
    assert body["workflow_id"] == "wf-123"


@resp_lib.activate
def test_launch_to_platform_galaxy(client):
    resp_lib.add(resp_lib.POST, _LAUNCH_URL, json=_GALAXY_LAUNCH_RESPONSE, status=200)
    result = client.launch_to_platform("STD0001", platform="galaxy")
    assert result.platform == "galaxy"
    assert "usegalaxy.eu" in result.launch_url


@resp_lib.activate
def test_launch_to_platform_422_raises(client):
    resp_lib.add(resp_lib.POST, _LAUNCH_URL, json={"detail": "Seqera not configured"}, status=422)
    with pytest.raises(requests.HTTPError):
        client.launch_to_platform("STD0001", platform="seqera")


@resp_lib.activate
def test_launch_to_platform_404_raises(client):
    resp_lib.add(resp_lib.POST, _LAUNCH_URL, json={"detail": "Dataset not found"}, status=404)
    with pytest.raises(requests.HTTPError):
        client.launch_to_platform("STD0001", platform="terra")


# ===========================================================================
# download_aria2c_manifest (Feature 20)
# ===========================================================================

_ARIA2C_CONTENT = (
    "https://s3.example.com/matrix.csv?sig=abc\n"
    "  out=matrix.csv\n"
    "https://s3.example.com/coords.csv?sig=def\n"
    "  out=coords.csv"
)


@resp_lib.activate
def test_download_aria2c_manifest_writes_file(client, tmp_path):
    resp_lib.add(
        resp_lib.GET, _ARIA2C_URL, body=_ARIA2C_CONTENT, content_type="text/plain", status=200
    )
    dest = str(tmp_path / "out.txt")
    client.download_aria2c_manifest("STD0001", dest)
    assert Path(dest).exists()
    assert "matrix.csv" in Path(dest).read_text()


@resp_lib.activate
def test_download_aria2c_manifest_passes_file_types(client, tmp_path):
    resp_lib.add(
        resp_lib.GET, _ARIA2C_URL, body=_ARIA2C_CONTENT, content_type="text/plain", status=200
    )
    dest = str(tmp_path / "out.txt")
    client.download_aria2c_manifest("STD0001", dest, file_types=["expression_matrix"])
    # Verify query params were sent
    called_url = resp_lib.calls[0].request.url
    assert "file_types" in called_url


@resp_lib.activate
def test_download_aria2c_manifest_creates_parent_dirs(client, tmp_path):
    resp_lib.add(
        resp_lib.GET, _ARIA2C_URL, body=_ARIA2C_CONTENT, content_type="text/plain", status=200
    )
    dest = str(tmp_path / "deep" / "nested" / "out.txt")
    client.download_aria2c_manifest("STD0001", dest)
    assert Path(dest).exists()


@resp_lib.activate
def test_download_aria2c_manifest_404_raises(client, tmp_path):
    resp_lib.add(resp_lib.GET, _ARIA2C_URL, status=404)
    with pytest.raises(requests.HTTPError):
        client.download_aria2c_manifest("STD0001", str(tmp_path / "out.txt"))


# ===========================================================================
# Pre-existing coverage gaps: list_versions, get_dataset with version,
# list_files with version, compare_datasets, get_comments, post_comment
# ===========================================================================

_VERSIONS_URL = f"{BASE}/api/v1/datasets/STD0001/versions"
_COMMENTS_URL = f"{BASE}/api/v1/datasets/STD0001/comments"
_COMPARE_URL = f"{BASE}/api/v1/datasets/compare"

_VERSION_RESPONSE = {
    "versions": [
        {
            "id": 1,
            "dataset_accession": "STD0001",
            "version_number": 1,
            "changelog": "Initial release",
            "published_at": "2026-01-01T00:00:00Z",
            "created_by": 1,
            "file_snapshot": None,
            "versioned_doi": None,
        }
    ],
    "total": 1,
}

_COMPARE_RESPONSE = {
    "datasets": [
        {
            "accession": "STD0001",
            "title": "Brain Atlas",
            "abstract": "A brain atlas.",
            "status": "published",
        },
        {
            "accession": "STD0002",
            "title": "Liver Dataset",
            "abstract": "Liver data.",
            "status": "published",
        },
    ]
}

_COMMENT_DATA = {
    "id": 1,
    "dataset_accession": "STD0001",
    "user_id": 42,
    "body": "Interesting batch effect in sample 3.",
    "is_deleted": False,
    "is_pinned": False,
    "created_at": "2026-06-01T10:00:00Z",
    "updated_at": "2026-06-01T10:00:00Z",
    "parent_id": None,
    "author": {"id": 42, "full_name": "Alice", "institution": "MIT"},
    "reactions": {"helpful": 2, "flag": 0},
    "reply_count": 0,
    "replies": [],
    "user_reaction": None,
}

_COMMENT_NO_AUTHOR = {
    **_COMMENT_DATA,
    "id": 2,
    "author": None,
}

_COMMENT_LIST_RESPONSE = {
    "items": [_COMMENT_DATA, _COMMENT_NO_AUTHOR],
    "total": 2,
}


@resp_lib.activate
def test_list_versions_returns_versions(client):
    resp_lib.add(resp_lib.GET, _VERSIONS_URL, json=_VERSION_RESPONSE, status=200)
    versions = client.list_versions("STD0001")
    assert len(versions) == 1
    assert versions[0].version_number == 1
    assert versions[0].changelog == "Initial release"


@resp_lib.activate
def test_get_dataset_with_version_param(client):
    dataset_resp = {
        "accession": "STD0001",
        "title": "Versioned Dataset",
        "abstract": "An abstract.",
        "status": "published",
    }
    resp_lib.add(resp_lib.GET, f"{BASE}/api/v1/datasets/STD0001", json=dataset_resp, status=200)
    ds = client.get_dataset("STD0001", version=2)
    assert ds.accession == "STD0001"
    called_url = resp_lib.calls[0].request.url
    assert "version=2" in called_url


@resp_lib.activate
def test_list_files_with_version_param(client):
    files_resp = [
        {
            "id": 1,
            "dataset_id": 1,
            "filename": "matrix.h5ad",
            "file_type": "expression_matrix",
            "upload_status": "complete",
            "created_at": "2026-01-01T00:00:00Z",
        }
    ]
    resp_lib.add(resp_lib.GET, f"{BASE}/api/v1/datasets/STD0001/files", json=files_resp, status=200)
    files = client.list_files("STD0001", version=1)
    assert len(files) == 1
    called_url = resp_lib.calls[0].request.url
    assert "version=1" in called_url


@resp_lib.activate
def test_compare_datasets_returns_datasets(client):
    resp_lib.add(resp_lib.GET, _COMPARE_URL, json=_COMPARE_RESPONSE, status=200)
    datasets = client.compare_datasets(["STD0001", "STD0002"])
    assert len(datasets) == 2
    assert datasets[0].accession == "STD0001"
    assert datasets[1].accession == "STD0002"


def test_compare_datasets_empty_raises_valueerror(client):
    with pytest.raises(ValueError, match="must not be empty"):
        client.compare_datasets([])


def test_compare_datasets_too_many_raises_valueerror(client):
    with pytest.raises(ValueError, match="Maximum 6"):
        client.compare_datasets(["A", "B", "C", "D", "E", "F", "G"])


@resp_lib.activate
def test_compare_datasets_404_raises(client):
    resp_lib.add(resp_lib.GET, _COMPARE_URL, status=404)
    with pytest.raises(requests.HTTPError):
        client.compare_datasets(["STD0001"])


@resp_lib.activate
def test_get_comments_returns_comment_list(client):
    resp_lib.add(resp_lib.GET, _COMMENTS_URL, json=_COMMENT_LIST_RESPONSE, status=200)
    result = client.get_comments("STD0001")
    assert result.total == 2
    assert len(result.items) == 2
    # Comment with author
    c = result.items[0]
    assert c.id == 1
    assert c.body == "Interesting batch effect in sample 3."
    assert c.author is not None
    assert c.author.full_name == "Alice"
    assert c.author.institution == "MIT"
    assert c.reactions.helpful == 2
    # Comment without author (None branch of _build_comment)
    c2 = result.items[1]
    assert c2.author is None


@resp_lib.activate
def test_get_comments_empty(client):
    resp_lib.add(resp_lib.GET, _COMMENTS_URL, json={"items": [], "total": 0}, status=200)
    result = client.get_comments("STD0001")
    assert result.total == 0
    assert result.items == []


@resp_lib.activate
def test_post_comment_returns_comment(client):
    resp_lib.add(resp_lib.POST, _COMMENTS_URL, json=_COMMENT_DATA, status=201)
    comment = client.post_comment("STD0001", "Interesting batch effect in sample 3.")
    assert comment.id == 1
    assert comment.body == "Interesting batch effect in sample 3."
    body = json.loads(resp_lib.calls[0].request.body)
    assert body["body"] == "Interesting batch effect in sample 3."
    assert "parent_id" not in body


@resp_lib.activate
def test_post_comment_with_parent_id(client):
    resp_lib.add(resp_lib.POST, _COMMENTS_URL, json={**_COMMENT_DATA, "parent_id": 5}, status=201)
    comment = client.post_comment("STD0001", "Reply text", parent_id=5)
    body = json.loads(resp_lib.calls[0].request.body)
    assert body["parent_id"] == 5
    assert comment.parent_id == 5


@resp_lib.activate
def test_post_comment_401_raises(client):
    resp_lib.add(resp_lib.POST, _COMMENTS_URL, status=401)
    with pytest.raises(requests.HTTPError):
        client.post_comment("STD0001", "A comment")


# ===========================================================================
# Reviewer Anonymous Access Links (Feature 21)
# ===========================================================================

_REVIEWER_TOKENS_URL = f"{BASE}/api/v1/datasets/STD0001/reviewer-tokens"
_REVIEWER_TOKEN_ID_URL = f"{BASE}/api/v1/datasets/STD0001/reviewer-tokens/tok-abc-123"

_MOCK_REVIEWER_TOKEN = {
    "id": "tok-abc-123",
    "dataset_accession": "STD0001",
    "label": "Nature Reviewer #1",
    "token_prefix": "abcd1234",
    "expires_at": "2026-12-31T00:00:00Z",
    "access_count": 0,
    "last_used_at": None,
    "created_at": "2026-06-01T00:00:00Z",
    "revoked_at": None,
}

_MOCK_CREATE_RESPONSE = {**_MOCK_REVIEWER_TOKEN, "token": "abcd1234_raw_secret_value"}


@resp_lib.activate
def test_create_reviewer_token_returns_token(client):
    resp_lib.add(resp_lib.POST, _REVIEWER_TOKENS_URL, json=_MOCK_CREATE_RESPONSE, status=201)
    result = client.create_reviewer_token("STD0001", label="Nature Reviewer #1", expires_in_days=90)
    assert result.id == "tok-abc-123"
    assert result.label == "Nature Reviewer #1"
    assert result.token == "abcd1234_raw_secret_value"
    assert result.access_count == 0


@resp_lib.activate
def test_create_reviewer_token_posts_correct_body(client):
    resp_lib.add(resp_lib.POST, _REVIEWER_TOKENS_URL, json=_MOCK_CREATE_RESPONSE, status=201)
    client.create_reviewer_token("STD0001", label="My Reviewer", expires_in_days=30)
    body = json.loads(resp_lib.calls[0].request.body)
    assert body["label"] == "My Reviewer"
    assert body["expires_in_days"] == 30


@resp_lib.activate
def test_create_reviewer_token_403_raises(client):
    resp_lib.add(resp_lib.POST, _REVIEWER_TOKENS_URL, status=403)
    with pytest.raises(requests.HTTPError):
        client.create_reviewer_token("STD0001", label="Rev", expires_in_days=30)


@resp_lib.activate
def test_list_reviewer_tokens_returns_list(client):
    resp_lib.add(
        resp_lib.GET,
        _REVIEWER_TOKENS_URL,
        json={"items": [_MOCK_REVIEWER_TOKEN], "total": 1},
        status=200,
    )
    result = client.list_reviewer_tokens("STD0001")
    assert len(result) == 1
    assert result[0].id == "tok-abc-123"
    assert result[0].label == "Nature Reviewer #1"


@resp_lib.activate
def test_list_reviewer_tokens_empty(client):
    resp_lib.add(resp_lib.GET, _REVIEWER_TOKENS_URL, json={"items": [], "total": 0}, status=200)
    result = client.list_reviewer_tokens("STD0001")
    assert result == []


@resp_lib.activate
def test_list_reviewer_tokens_404_raises(client):
    resp_lib.add(resp_lib.GET, _REVIEWER_TOKENS_URL, status=404)
    with pytest.raises(requests.HTTPError):
        client.list_reviewer_tokens("STD0001")


@resp_lib.activate
def test_revoke_reviewer_token_returns_revoked(client):
    revoked = {**_MOCK_REVIEWER_TOKEN, "revoked_at": "2026-06-15T00:00:00Z"}
    resp_lib.add(resp_lib.DELETE, _REVIEWER_TOKEN_ID_URL, json=revoked, status=200)
    result = client.revoke_reviewer_token("STD0001", "tok-abc-123")
    assert result.revoked_at == "2026-06-15T00:00:00Z"
    assert result.id == "tok-abc-123"


@resp_lib.activate
def test_revoke_reviewer_token_404_raises(client):
    resp_lib.add(resp_lib.DELETE, _REVIEWER_TOKEN_ID_URL, status=404)
    with pytest.raises(requests.HTTPError):
        client.revoke_reviewer_token("STD0001", "tok-abc-123")


# ===========================================================================
# UMAP / Embedding Visualization (Feature 22)
# ===========================================================================

_EMB_BASE = f"{BASE}/api/v1/datasets/STD0001/samples/1/embeddings"
_EMB_LIST_URL = _EMB_BASE
_EMB_GET_URL = f"{_EMB_BASE}/1"
_EMB_UPLOAD_URL = f"{_EMB_BASE}/upload"
_EMB_COMPUTE_URL = f"{_EMB_BASE}/compute"
_EMB_STATUS_URL = f"{_EMB_BASE}/compute/status"

_EMB_INFO = {
    "id": 1,
    "method": "UMAP",
    "n_components": 2,
    "parameters": None,
    "compute_status": None,
    "created_at": "2026-06-01T00:00:00Z",
}

_EMB_COORDS = {
    "embedding_id": 1,
    "method": "UMAP",
    "n_components": 2,
    "spots": [
        {"barcode": "BARCODE_r1c1", "dim_1": 0.1, "dim_2": -0.2, "dim_3": None},
        {"barcode": "BARCODE_r2c2", "dim_1": 1.5, "dim_2": 0.8, "dim_3": None},
    ],
}


@resp_lib.activate
def test_list_embeddings_returns_embedding_infos(client):
    resp_lib.add(resp_lib.GET, _EMB_LIST_URL, json={"embeddings": [_EMB_INFO]}, status=200)
    result = client.list_embeddings("STD0001", sample_id=1)
    assert len(result) == 1
    emb = result[0]
    assert emb.id == 1
    assert emb.method == "UMAP"
    assert emb.n_components == 2
    assert emb.compute_status is None


@resp_lib.activate
def test_list_embeddings_empty(client):
    resp_lib.add(resp_lib.GET, _EMB_LIST_URL, json={"embeddings": []}, status=200)
    result = client.list_embeddings("STD0001", sample_id=1)
    assert result == []


@resp_lib.activate
def test_list_embeddings_404_raises(client):
    resp_lib.add(resp_lib.GET, _EMB_LIST_URL, status=404)
    with pytest.raises(requests.HTTPError):
        client.list_embeddings("STD0001", sample_id=1)


@resp_lib.activate
def test_list_embeddings_calls_correct_url(client):
    resp_lib.add(resp_lib.GET, _EMB_LIST_URL, json={"embeddings": []}, status=200)
    client.list_embeddings("STD0001", sample_id=1)
    assert resp_lib.calls[0].request.url == _EMB_LIST_URL


@resp_lib.activate
def test_get_embedding_returns_coords(client):
    resp_lib.add(resp_lib.GET, _EMB_GET_URL, json=_EMB_COORDS, status=200)
    result = client.get_embedding("STD0001", sample_id=1, embedding_id=1)
    assert result.embedding_id == 1
    assert result.method == "UMAP"
    assert result.n_components == 2
    assert len(result.spots) == 2


@resp_lib.activate
def test_get_embedding_spots_have_correct_types(client):
    resp_lib.add(resp_lib.GET, _EMB_GET_URL, json=_EMB_COORDS, status=200)
    result = client.get_embedding("STD0001", sample_id=1, embedding_id=1)
    spot = result.spots[0]
    assert spot.barcode == "BARCODE_r1c1"
    assert isinstance(spot.dim_1, float)
    assert isinstance(spot.dim_2, float)
    assert spot.dim_3 is None


@resp_lib.activate
def test_get_embedding_3d_spots(client):
    coords3d = {
        **_EMB_COORDS,
        "n_components": 3,
        "spots": [{"barcode": "BARCODE_r1c1", "dim_1": 0.1, "dim_2": 0.2, "dim_3": 0.3}],
    }
    resp_lib.add(resp_lib.GET, _EMB_GET_URL, json=coords3d, status=200)
    result = client.get_embedding("STD0001", sample_id=1, embedding_id=1)
    assert result.spots[0].dim_3 == 0.3


@resp_lib.activate
def test_get_embedding_404_raises(client):
    resp_lib.add(resp_lib.GET, _EMB_GET_URL, status=404)
    with pytest.raises(requests.HTTPError):
        client.get_embedding("STD0001", sample_id=1, embedding_id=1)


@resp_lib.activate
def test_get_embedding_500_raises(client):
    resp_lib.add(resp_lib.GET, _EMB_GET_URL, status=500)
    with pytest.raises(requests.HTTPError):
        client.get_embedding("STD0001", sample_id=1, embedding_id=1)


@resp_lib.activate
def test_upload_embedding_returns_result(client, tmp_path):
    csv_path = tmp_path / "umap.csv"
    csv_path.write_text("barcode,UMAP_1,UMAP_2\nBARCODE_r1c1,0.1,0.2\n")
    resp_lib.add(
        resp_lib.POST,
        _EMB_UPLOAD_URL,
        json={"embedding_id": 3, "inserted": 1, "sample_id": 1},
        status=201,
    )

    result = client.upload_embedding("STD0001", sample_id=1, csv_path=str(csv_path))
    assert result.embedding_id == 3
    assert result.inserted == 1
    assert result.sample_id == 1


@resp_lib.activate
def test_upload_embedding_posts_multipart(client, tmp_path):
    csv_path = tmp_path / "umap.csv"
    csv_path.write_text("barcode,UMAP_1,UMAP_2\nBARCODE_r1c1,0.1,0.2\n")
    resp_lib.add(
        resp_lib.POST,
        _EMB_UPLOAD_URL,
        json={"embedding_id": 1, "inserted": 1, "sample_id": 1},
        status=201,
    )

    client.upload_embedding("STD0001", sample_id=1, csv_path=str(csv_path))
    req = resp_lib.calls[0].request
    assert req.url == _EMB_UPLOAD_URL
    assert "multipart" in req.headers.get("Content-Type", "").lower()


@resp_lib.activate
def test_upload_embedding_422_raises(client, tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("barcode,x,y\nINVALID,0,0\n")
    resp_lib.add(resp_lib.POST, _EMB_UPLOAD_URL, status=422)
    with pytest.raises(requests.HTTPError):
        client.upload_embedding("STD0001", sample_id=1, csv_path=str(csv_path))


@resp_lib.activate
def test_upload_embedding_403_raises(client, tmp_path):
    csv_path = tmp_path / "emb.csv"
    csv_path.write_text("barcode,UMAP_1,UMAP_2\nBARCODE_r1c1,0.1,0.2\n")
    resp_lib.add(resp_lib.POST, _EMB_UPLOAD_URL, status=403)
    with pytest.raises(requests.HTTPError):
        client.upload_embedding("STD0001", sample_id=1, csv_path=str(csv_path))


@resp_lib.activate
def test_compute_embedding_returns_pending(client):
    resp_lib.add(
        resp_lib.POST, _EMB_COMPUTE_URL, json={"embedding_id": 7, "status": "pending"}, status=202
    )
    result = client.compute_embedding("STD0001", sample_id=1)
    assert result.embedding_id == 7
    assert result.status == "pending"


@resp_lib.activate
def test_compute_embedding_posts_to_correct_url(client):
    resp_lib.add(
        resp_lib.POST, _EMB_COMPUTE_URL, json={"embedding_id": 7, "status": "pending"}, status=202
    )
    client.compute_embedding("STD0001", sample_id=1)
    assert resp_lib.calls[0].request.url == _EMB_COMPUTE_URL


@resp_lib.activate
def test_compute_embedding_409_raises(client):
    resp_lib.add(resp_lib.POST, _EMB_COMPUTE_URL, status=409)
    with pytest.raises(requests.HTTPError):
        client.compute_embedding("STD0001", sample_id=1)


@resp_lib.activate
def test_compute_embedding_403_raises(client):
    resp_lib.add(resp_lib.POST, _EMB_COMPUTE_URL, status=403)
    with pytest.raises(requests.HTTPError):
        client.compute_embedding("STD0001", sample_id=1)


@resp_lib.activate
def test_get_embedding_compute_status_not_started(client):
    resp_lib.add(
        resp_lib.GET,
        _EMB_STATUS_URL,
        json={"embedding_id": None, "status": "not_started"},
        status=200,
    )
    result = client.get_embedding_compute_status("STD0001", sample_id=1)
    assert result.status == "not_started"
    assert result.embedding_id is None


@resp_lib.activate
def test_get_embedding_compute_status_computing(client):
    resp_lib.add(
        resp_lib.GET, _EMB_STATUS_URL, json={"embedding_id": 7, "status": "computing"}, status=200
    )
    result = client.get_embedding_compute_status("STD0001", sample_id=1)
    assert result.status == "computing"
    assert result.embedding_id == 7


@resp_lib.activate
def test_get_embedding_compute_status_ready(client):
    resp_lib.add(
        resp_lib.GET, _EMB_STATUS_URL, json={"embedding_id": 7, "status": "ready"}, status=200
    )
    result = client.get_embedding_compute_status("STD0001", sample_id=1)
    assert result.status == "ready"


@resp_lib.activate
def test_get_embedding_compute_status_failed(client):
    resp_lib.add(
        resp_lib.GET, _EMB_STATUS_URL, json={"embedding_id": 7, "status": "failed"}, status=200
    )
    result = client.get_embedding_compute_status("STD0001", sample_id=1)
    assert result.status == "failed"


@resp_lib.activate
def test_get_embedding_compute_status_calls_correct_url(client):
    resp_lib.add(
        resp_lib.GET,
        _EMB_STATUS_URL,
        json={"embedding_id": None, "status": "not_started"},
        status=200,
    )
    client.get_embedding_compute_status("STD0001", sample_id=1)
    assert resp_lib.calls[0].request.url == _EMB_STATUS_URL


# ---------------------------------------------------------------------------
# spatially_variable_genes (Feature 23)
# ---------------------------------------------------------------------------

_SVG_URL = f"{BASE}/api/v1/datasets/STD0001/samples/1/spatially-variable-genes"

_SVG_RESPONSE = {
    "genes": [
        {
            "gene": "ACTB",
            "morans_i": 0.42,
            "pvalue": 0.001,
            "padj": 0.003,
            "mean_expr": 1.25,
            "n_spots_expressed": 1800,
        },
        {
            "gene": "GAPDH",
            "morans_i": 0.31,
            "pvalue": 0.01,
            "padj": 0.02,
            "mean_expr": 0.9,
            "n_spots_expressed": 1500,
        },
    ],
    "method": "morans_i",
    "n_genes_tested": 18000,
    "sample_id": 1,
    "subsampled": False,
    "n_spots": 3522,
    "cached": False,
}


@resp_lib.activate
def test_spatially_variable_genes_returns_svg_result(client):
    resp_lib.add(resp_lib.POST, _SVG_URL, json=_SVG_RESPONSE, status=200)
    result = client.spatially_variable_genes("STD0001", sample_id=1)
    assert result.method == "morans_i"
    assert result.n_genes_tested == 18000
    assert result.sample_id == 1
    assert result.subsampled is False
    assert result.n_spots == 3522
    assert result.cached is False
    assert len(result.genes) == 2


@resp_lib.activate
def test_spatially_variable_genes_gene_fields(client):
    resp_lib.add(resp_lib.POST, _SVG_URL, json=_SVG_RESPONSE, status=200)
    result = client.spatially_variable_genes("STD0001", sample_id=1)
    g = result.genes[0]
    assert g.gene == "ACTB"
    assert g.morans_i == 0.42
    assert g.pvalue == 0.001
    assert g.padj == 0.003
    assert g.mean_expr == 1.25
    assert g.n_spots_expressed == 1800


@resp_lib.activate
def test_spatially_variable_genes_posts_to_correct_url(client):
    resp_lib.add(resp_lib.POST, _SVG_URL, json=_SVG_RESPONSE, status=200)
    client.spatially_variable_genes("STD0001", sample_id=1)
    assert resp_lib.calls[0].request.url == _SVG_URL


@resp_lib.activate
def test_spatially_variable_genes_sends_params(client):
    resp_lib.add(resp_lib.POST, _SVG_URL, json=_SVG_RESPONSE, status=200)
    client.spatially_variable_genes(
        "STD0001", sample_id=1, method="morans_i", n_top_genes=50, min_expr_fraction=0.1
    )
    import json as json_mod

    body = json_mod.loads(resp_lib.calls[0].request.body)
    assert body["method"] == "morans_i"
    assert body["n_top_genes"] == 50
    assert body["min_expr_fraction"] == 0.1


@resp_lib.activate
def test_spatially_variable_genes_cached_result(client):
    cached_response = {**_SVG_RESPONSE, "cached": True}
    resp_lib.add(resp_lib.POST, _SVG_URL, json=cached_response, status=200)
    result = client.spatially_variable_genes("STD0001", sample_id=1)
    assert result.cached is True


@resp_lib.activate
def test_spatially_variable_genes_429_raises(client):
    resp_lib.add(resp_lib.POST, _SVG_URL, status=429)
    with pytest.raises(requests.HTTPError):
        client.spatially_variable_genes("STD0001", sample_id=1)


@resp_lib.activate
def test_spatially_variable_genes_empty_genes(client):
    empty_response = {**_SVG_RESPONSE, "genes": [], "n_genes_tested": 0}
    resp_lib.add(resp_lib.POST, _SVG_URL, json=empty_response, status=200)
    result = client.spatially_variable_genes("STD0001", sample_id=1)
    assert result.genes == []
    assert result.n_genes_tested == 0


# ===========================================================================
# Format Conversion (Feature 25)
# ===========================================================================

_CONVERT_URL = f"{BASE}/api/v1/datasets/STD0001/files/1/convert"
_CONVERT_JOB_URL = f"{BASE}/api/v1/datasets/STD0001/files/1/convert/job-abc"

_PENDING_JOB = {
    "job_id": "job-abc",
    "status": "pending",
    "target_format": "h5ad",
    "size_bytes": None,
    "url": None,
}
_READY_JOB = {
    "job_id": "job-abc",
    "status": "ready",
    "target_format": "h5ad",
    "size_bytes": 1024,
    "url": "https://storage.example.com/output.h5ad?token=xxx",
}


@resp_lib.activate
def test_start_format_conversion_pending(client):
    resp_lib.add(resp_lib.POST, _CONVERT_URL, json=_PENDING_JOB, status=202)
    job = client.start_format_conversion("STD0001", file_id=1, target_format="h5ad")
    assert job.job_id == "job-abc"
    assert job.status == "pending"
    assert job.target_format == "h5ad"
    assert job.size_bytes is None
    assert job.url is None


@resp_lib.activate
def test_start_format_conversion_already_ready(client):
    resp_lib.add(resp_lib.POST, _CONVERT_URL, json=_READY_JOB, status=202)
    job = client.start_format_conversion("STD0001", file_id=1, target_format="h5ad")
    assert job.status == "ready"
    assert job.url == "https://storage.example.com/output.h5ad?token=xxx"
    assert job.size_bytes == 1024


@resp_lib.activate
def test_start_format_conversion_404_raises(client):
    resp_lib.add(resp_lib.POST, _CONVERT_URL, status=404)
    with pytest.raises(requests.HTTPError):
        client.start_format_conversion("STD0001", file_id=1, target_format="h5ad")


@resp_lib.activate
def test_get_format_conversion_status(client):
    resp_lib.add(resp_lib.GET, _CONVERT_JOB_URL, json=_READY_JOB, status=200)
    job = client.get_format_conversion_status("STD0001", file_id=1, job_id="job-abc")
    assert job.status == "ready"
    assert job.size_bytes == 1024


@resp_lib.activate
def test_get_format_conversion_status_404_raises(client):
    resp_lib.add(resp_lib.GET, _CONVERT_JOB_URL, status=404)
    with pytest.raises(requests.HTTPError):
        client.get_format_conversion_status("STD0001", file_id=1, job_id="job-abc")


@resp_lib.activate
def test_download_converted_file_ready_immediately(client, tmp_path):
    dest = tmp_path / "output.h5ad"
    resp_lib.add(resp_lib.GET, _CONVERT_JOB_URL, json=_READY_JOB, status=200)
    resp_lib.add(resp_lib.GET, _READY_JOB["url"], body=b"mock-h5ad-content", status=200)
    result = client.download_converted_file("STD0001", file_id=1, job_id="job-abc", dest=dest)
    assert result == dest
    assert dest.read_bytes() == b"mock-h5ad-content"
    assert "Authorization" not in resp_lib.calls[-1].request.headers


@resp_lib.activate
def test_download_converted_file_polls_until_ready(client, tmp_path):
    dest = tmp_path / "output.h5ad"
    converting_job = {**_PENDING_JOB, "status": "converting"}
    resp_lib.add(resp_lib.GET, _CONVERT_JOB_URL, json=converting_job, status=200)
    resp_lib.add(resp_lib.GET, _CONVERT_JOB_URL, json=_READY_JOB, status=200)
    resp_lib.add(resp_lib.GET, _READY_JOB["url"], body=b"content", status=200)
    with patch("time.sleep"):
        result = client.download_converted_file("STD0001", file_id=1, job_id="job-abc", dest=dest)
    assert result == dest
    assert dest.read_bytes() == b"content"


@resp_lib.activate
def test_download_converted_file_failed_raises(client, tmp_path):
    dest = tmp_path / "output.h5ad"
    failed_job = {**_PENDING_JOB, "status": "failed"}
    resp_lib.add(resp_lib.GET, _CONVERT_JOB_URL, json=failed_job, status=200)
    with pytest.raises(requests.HTTPError, match="failed"):
        client.download_converted_file("STD0001", file_id=1, job_id="job-abc", dest=dest)


@resp_lib.activate
def test_download_converted_file_no_url_raises(client, tmp_path):
    dest = tmp_path / "output.h5ad"
    ready_no_url = {**_READY_JOB, "url": None}
    resp_lib.add(resp_lib.GET, _CONVERT_JOB_URL, json=ready_no_url, status=200)
    with pytest.raises(RuntimeError, match="no download URL"):
        client.download_converted_file("STD0001", file_id=1, job_id="job-abc", dest=dest)


# ===========================================================================
# Gene-Centric Cross-Dataset Expression Search (Feature 26)
# ===========================================================================

_GENE_SEARCH_URL = f"{BASE}/api/v1/search/genes"
_GENE_AUTOCOMPLETE_URL = f"{BASE}/api/v1/search/genes/autocomplete"

_GENE_SEARCH_ITEM = {
    "dataset_accession": "STD0001",
    "dataset_title": "Brain Atlas",
    "sample_id": 1,
    "organism": "Homo sapiens",
    "tissue": "brain cortex",
    "technology": "10x Visium",
    "mean_expr": 4.82,
    "pct_spots_expressed": 0.91,
    "max_expr": 12.3,
    "thumbnail_url": None,
}

_GENE_SEARCH_RESPONSE = {
    "items": [_GENE_SEARCH_ITEM],
    "total": 1,
    "page": 1,
    "per_page": 20,
}

_AUTOCOMPLETE_RESPONSE = [
    {"gene_symbol": "GAPDH", "dataset_count": 5},
    {"gene_symbol": "GATA3", "dataset_count": 2},
]


@resp_lib.activate
def test_search_by_gene_returns_response(client):
    resp_lib.add(resp_lib.GET, _GENE_SEARCH_URL, json=_GENE_SEARCH_RESPONSE, status=200)
    result = client.search_by_gene("GAPDH")
    assert result.total == 1
    assert result.page == 1
    assert result.per_page == 20
    assert len(result.items) == 1


@resp_lib.activate
def test_search_by_gene_item_fields(client):
    resp_lib.add(resp_lib.GET, _GENE_SEARCH_URL, json=_GENE_SEARCH_RESPONSE, status=200)
    result = client.search_by_gene("GAPDH")
    item = result.items[0]
    assert item.dataset_accession == "STD0001"
    assert item.dataset_title == "Brain Atlas"
    assert item.sample_id == 1
    assert item.organism == "Homo sapiens"
    assert item.tissue == "brain cortex"
    assert item.technology == "10x Visium"
    assert item.mean_expr == 4.82
    assert item.pct_spots_expressed == 0.91
    assert item.max_expr == 12.3
    assert item.thumbnail_url is None


@resp_lib.activate
def test_search_by_gene_calls_correct_url(client):
    resp_lib.add(resp_lib.GET, _GENE_SEARCH_URL, json=_GENE_SEARCH_RESPONSE, status=200)
    client.search_by_gene("GAPDH")
    assert resp_lib.calls[0].request.url.startswith(_GENE_SEARCH_URL)
    assert "q=GAPDH" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_search_by_gene_sends_default_params(client):
    resp_lib.add(resp_lib.GET, _GENE_SEARCH_URL, json=_GENE_SEARCH_RESPONSE, status=200)
    client.search_by_gene("GAPDH")
    url = resp_lib.calls[0].request.url
    assert "min_pct_expressed=0.1" in url
    assert "sort_by=mean_expr" in url
    assert "page=1" in url
    assert "per_page=20" in url


@resp_lib.activate
def test_search_by_gene_sends_optional_filters(client):
    resp_lib.add(resp_lib.GET, _GENE_SEARCH_URL, json=_GENE_SEARCH_RESPONSE, status=200)
    client.search_by_gene("GAPDH", organism_id=2, tissue_id=3, technology_id=1)
    url = resp_lib.calls[0].request.url
    assert "organism_id=2" in url
    assert "tissue_id=3" in url
    assert "technology_id=1" in url


@resp_lib.activate
def test_search_by_gene_no_optional_filters_when_none(client):
    resp_lib.add(resp_lib.GET, _GENE_SEARCH_URL, json=_GENE_SEARCH_RESPONSE, status=200)
    client.search_by_gene("GAPDH")
    url = resp_lib.calls[0].request.url
    assert "organism_id" not in url
    assert "tissue_id" not in url
    assert "technology_id" not in url


@resp_lib.activate
def test_search_by_gene_custom_min_pct(client):
    resp_lib.add(resp_lib.GET, _GENE_SEARCH_URL, json=_GENE_SEARCH_RESPONSE, status=200)
    client.search_by_gene("GAPDH", min_pct_expressed=0.5)
    assert "min_pct_expressed=0.5" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_search_by_gene_custom_sort(client):
    resp_lib.add(resp_lib.GET, _GENE_SEARCH_URL, json=_GENE_SEARCH_RESPONSE, status=200)
    client.search_by_gene("GAPDH", sort_by="pct_spots_expressed")
    assert "sort_by=pct_spots_expressed" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_search_by_gene_custom_pagination(client):
    resp_lib.add(resp_lib.GET, _GENE_SEARCH_URL, json=_GENE_SEARCH_RESPONSE, status=200)
    client.search_by_gene("GAPDH", page=2, per_page=10)
    url = resp_lib.calls[0].request.url
    assert "page=2" in url
    assert "per_page=10" in url


@resp_lib.activate
def test_search_by_gene_empty_results(client):
    resp_lib.add(
        resp_lib.GET,
        _GENE_SEARCH_URL,
        json={"items": [], "total": 0, "page": 1, "per_page": 20},
        status=200,
    )
    result = client.search_by_gene("NONEXISTENT")
    assert result.total == 0
    assert result.items == []


@resp_lib.activate
def test_search_by_gene_null_optional_fields(client):
    item_no_optional = {
        "dataset_accession": "STD0002",
        "dataset_title": "Test",
        "sample_id": 1,
        "organism": None,
        "tissue": None,
        "technology": None,
        "mean_expr": 1.0,
        "pct_spots_expressed": 0.2,
        "max_expr": 3.0,
        "thumbnail_url": None,
    }
    resp_lib.add(
        resp_lib.GET,
        _GENE_SEARCH_URL,
        json={**_GENE_SEARCH_RESPONSE, "items": [item_no_optional]},
        status=200,
    )
    result = client.search_by_gene("GAPDH")
    assert result.items[0].organism is None
    assert result.items[0].tissue is None
    assert result.items[0].technology is None


@resp_lib.activate
def test_search_by_gene_404_raises(client):
    resp_lib.add(resp_lib.GET, _GENE_SEARCH_URL, json={"detail": "Gene not found"}, status=404)
    with pytest.raises(requests.HTTPError):
        client.search_by_gene("UNKNOWN")


@resp_lib.activate
def test_gene_autocomplete_returns_list(client):
    resp_lib.add(resp_lib.GET, _GENE_AUTOCOMPLETE_URL, json=_AUTOCOMPLETE_RESPONSE, status=200)
    result = client.gene_autocomplete("GAP")
    assert len(result) == 2
    assert result[0].gene_symbol == "GAPDH"
    assert result[0].dataset_count == 5


@resp_lib.activate
def test_gene_autocomplete_calls_correct_url(client):
    resp_lib.add(resp_lib.GET, _GENE_AUTOCOMPLETE_URL, json=_AUTOCOMPLETE_RESPONSE, status=200)
    client.gene_autocomplete("GAP")
    url = resp_lib.calls[0].request.url
    assert _GENE_AUTOCOMPLETE_URL in url
    assert "q=GAP" in url


@resp_lib.activate
def test_gene_autocomplete_default_per_page(client):
    resp_lib.add(resp_lib.GET, _GENE_AUTOCOMPLETE_URL, json=_AUTOCOMPLETE_RESPONSE, status=200)
    client.gene_autocomplete("GAP")
    assert "per_page=10" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_gene_autocomplete_custom_per_page(client):
    resp_lib.add(resp_lib.GET, _GENE_AUTOCOMPLETE_URL, json=_AUTOCOMPLETE_RESPONSE, status=200)
    client.gene_autocomplete("GAP", per_page=5)
    assert "per_page=5" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_gene_autocomplete_empty_list(client):
    resp_lib.add(resp_lib.GET, _GENE_AUTOCOMPLETE_URL, json=[], status=200)
    result = client.gene_autocomplete("XYZ")
    assert result == []


@resp_lib.activate
def test_gene_autocomplete_suggestion_fields(client):
    resp_lib.add(
        resp_lib.GET,
        _GENE_AUTOCOMPLETE_URL,
        json=[{"gene_symbol": "BRCA1", "dataset_count": 7}],
        status=200,
    )
    result = client.gene_autocomplete("BRC")
    assert result[0].gene_symbol == "BRCA1"
    assert result[0].dataset_count == 7


@resp_lib.activate
def test_gene_autocomplete_404_raises(client):
    resp_lib.add(resp_lib.GET, _GENE_AUTOCOMPLETE_URL, json={"detail": "Bad request"}, status=422)
    with pytest.raises(requests.HTTPError):
        client.gene_autocomplete("")


# ===========================================================================
# Teams (Feature 27)
# ===========================================================================

_TEAMS_URL = f"{BASE}/api/v1/teams"
_TEAM_SLUG_URL = f"{BASE}/api/v1/teams/smith-lab"
_MY_TEAMS_URL = f"{BASE}/api/v1/teams/my"
_MEMBERS_URL = f"{BASE}/api/v1/teams/smith-lab/members"
_INVITATIONS_URL = f"{BASE}/api/v1/teams/smith-lab/invitations"
_ACCEPT_URL = f"{BASE}/api/v1/teams/smith-lab/invitations/accept"
_MEMBER_URL = f"{BASE}/api/v1/teams/smith-lab/members/42"
_ASSIGN_TEAM_URL = f"{BASE}/api/v1/datasets/STD0001/team"

_TEAM_JSON = {
    "id": 1,
    "name": "Smith Lab",
    "slug": "smith-lab",
    "owner_id": 10,
    "created_at": "2026-06-01T00:00:00Z",
    "member_count": 4,
    "dataset_count": 7,
    "description": "Harvard Medical School",
}

_MEMBER_JSON = {
    "id": 5,
    "team_id": 1,
    "user_id": 42,
    "role": "member",
    "invited_at": "2026-06-01T00:00:00Z",
    "accepted_at": "2026-06-02T00:00:00Z",
    "full_name": "Alice Researcher",
    "email": "alice@university.edu",
    "institution": "MIT",
}


@resp_lib.activate
def test_list_teams_returns_team_list(client):
    resp_lib.add(
        resp_lib.GET,
        _TEAMS_URL,
        json={"items": [_TEAM_JSON], "total": 1, "page": 1, "per_page": 20},
        status=200,
    )
    result = client.list_teams()
    assert result.total == 1
    assert result.page == 1
    assert result.per_page == 20
    assert len(result.items) == 1


@resp_lib.activate
def test_list_teams_team_fields(client):
    resp_lib.add(
        resp_lib.GET,
        _TEAMS_URL,
        json={"items": [_TEAM_JSON], "total": 1, "page": 1, "per_page": 20},
        status=200,
    )
    result = client.list_teams()
    team = result.items[0]
    assert team.id == 1
    assert team.name == "Smith Lab"
    assert team.slug == "smith-lab"
    assert team.owner_id == 10
    assert team.member_count == 4
    assert team.dataset_count == 7
    assert team.description == "Harvard Medical School"


@resp_lib.activate
def test_list_teams_sends_pagination_params(client):
    resp_lib.add(
        resp_lib.GET,
        _TEAMS_URL,
        json={"items": [], "total": 0, "page": 2, "per_page": 10},
        status=200,
    )
    client.list_teams(page=2, per_page=10)
    req = resp_lib.calls[0].request
    assert "page=2" in req.url
    assert "per_page=10" in req.url


@resp_lib.activate
def test_get_team_returns_team(client):
    resp_lib.add(resp_lib.GET, _TEAM_SLUG_URL, json=_TEAM_JSON, status=200)
    team = client.get_team("smith-lab")
    assert team.slug == "smith-lab"
    assert team.name == "Smith Lab"


@resp_lib.activate
def test_get_team_404_raises(client):
    resp_lib.add(resp_lib.GET, _TEAM_SLUG_URL, json={"detail": "Not found"}, status=404)
    with pytest.raises(requests.HTTPError):
        client.get_team("smith-lab")


@resp_lib.activate
def test_get_my_teams_returns_list(client):
    resp_lib.add(resp_lib.GET, _MY_TEAMS_URL, json=[_TEAM_JSON], status=200)
    result = client.get_my_teams()
    assert len(result) == 1
    assert result[0].slug == "smith-lab"


@resp_lib.activate
def test_get_my_teams_empty(client):
    resp_lib.add(resp_lib.GET, _MY_TEAMS_URL, json=[], status=200)
    result = client.get_my_teams()
    assert result == []


@resp_lib.activate
def test_create_team_returns_team(client):
    resp_lib.add(resp_lib.POST, _TEAMS_URL, json=_TEAM_JSON, status=201)
    team = client.create_team("Smith Lab", description="Harvard Medical School")
    assert team.name == "Smith Lab"
    assert team.description == "Harvard Medical School"


@resp_lib.activate
def test_create_team_no_description(client):
    resp_lib.add(resp_lib.POST, _TEAMS_URL, json={**_TEAM_JSON, "description": None}, status=201)
    team = client.create_team("Smith Lab")
    assert team.description is None


@resp_lib.activate
def test_create_team_sends_payload(client):
    resp_lib.add(resp_lib.POST, _TEAMS_URL, json=_TEAM_JSON, status=201)
    client.create_team("Smith Lab", description="Harvard")
    req = resp_lib.calls[0].request
    body = json.loads(req.body)
    assert body["name"] == "Smith Lab"
    assert body["description"] == "Harvard"


@resp_lib.activate
def test_create_team_401_raises(client):
    resp_lib.add(resp_lib.POST, _TEAMS_URL, json={"detail": "Unauthorized"}, status=401)
    with pytest.raises(requests.HTTPError):
        client.create_team("Smith Lab")


@resp_lib.activate
def test_update_team_returns_updated_team(client):
    updated = {**_TEAM_JSON, "name": "Smith Lab v2"}
    resp_lib.add(resp_lib.PATCH, _TEAM_SLUG_URL, json=updated, status=200)
    team = client.update_team("smith-lab", name="Smith Lab v2")
    assert team.name == "Smith Lab v2"


@resp_lib.activate
def test_update_team_sends_only_provided_fields(client):
    resp_lib.add(resp_lib.PATCH, _TEAM_SLUG_URL, json=_TEAM_JSON, status=200)
    client.update_team("smith-lab", description="New desc")
    body = json.loads(resp_lib.calls[0].request.body)
    assert "name" not in body
    assert body["description"] == "New desc"


@resp_lib.activate
def test_update_team_403_raises(client):
    resp_lib.add(resp_lib.PATCH, _TEAM_SLUG_URL, json={"detail": "Forbidden"}, status=403)
    with pytest.raises(requests.HTTPError):
        client.update_team("smith-lab", name="X")


@resp_lib.activate
def test_delete_team_success(client):
    resp_lib.add(resp_lib.DELETE, _TEAM_SLUG_URL, status=204)
    client.delete_team("smith-lab")
    assert resp_lib.calls[0].request.method == "DELETE"


@resp_lib.activate
def test_delete_team_403_raises(client):
    resp_lib.add(resp_lib.DELETE, _TEAM_SLUG_URL, json={"detail": "Forbidden"}, status=403)
    with pytest.raises(requests.HTTPError):
        client.delete_team("smith-lab")


@resp_lib.activate
def test_list_team_members_returns_list(client):
    resp_lib.add(resp_lib.GET, _MEMBERS_URL, json=[_MEMBER_JSON], status=200)
    members = client.list_team_members("smith-lab")
    assert len(members) == 1
    m = members[0]
    assert m.id == 5
    assert m.user_id == 42
    assert m.role == "member"
    assert m.full_name == "Alice Researcher"
    assert m.email == "alice@university.edu"
    assert m.institution == "MIT"


@resp_lib.activate
def test_list_team_members_empty(client):
    resp_lib.add(resp_lib.GET, _MEMBERS_URL, json=[], status=200)
    assert client.list_team_members("smith-lab") == []


@resp_lib.activate
def test_list_team_members_pending_member(client):
    pending = {**_MEMBER_JSON, "accepted_at": None}
    resp_lib.add(resp_lib.GET, _MEMBERS_URL, json=[pending], status=200)
    members = client.list_team_members("smith-lab")
    assert members[0].accepted_at is None


@resp_lib.activate
def test_invite_team_member_returns_member(client):
    resp_lib.add(resp_lib.POST, _INVITATIONS_URL, json=_MEMBER_JSON, status=201)
    member = client.invite_team_member("smith-lab", "alice@university.edu", role="member")
    assert member.email == "alice@university.edu"


@resp_lib.activate
def test_invite_team_member_sends_payload(client):
    resp_lib.add(resp_lib.POST, _INVITATIONS_URL, json=_MEMBER_JSON, status=201)
    client.invite_team_member("smith-lab", "alice@university.edu", role="admin")
    body = json.loads(resp_lib.calls[0].request.body)
    assert body["email"] == "alice@university.edu"
    assert body["role"] == "admin"


@resp_lib.activate
def test_invite_team_member_default_role(client):
    resp_lib.add(resp_lib.POST, _INVITATIONS_URL, json=_MEMBER_JSON, status=201)
    client.invite_team_member("smith-lab", "alice@university.edu")
    body = json.loads(resp_lib.calls[0].request.body)
    assert body["role"] == "member"


@resp_lib.activate
def test_invite_team_member_404_raises(client):
    resp_lib.add(resp_lib.POST, _INVITATIONS_URL, json={"detail": "User not found"}, status=404)
    with pytest.raises(requests.HTTPError):
        client.invite_team_member("smith-lab", "nobody@example.com")


@resp_lib.activate
def test_accept_team_invitation_returns_member(client):
    resp_lib.add(
        resp_lib.POST,
        _ACCEPT_URL,
        json={**_MEMBER_JSON, "accepted_at": "2026-06-16T00:00:00Z"},
        status=200,
    )
    member = client.accept_team_invitation("smith-lab")
    assert member.accepted_at == "2026-06-16T00:00:00Z"


@resp_lib.activate
def test_accept_team_invitation_404_raises(client):
    resp_lib.add(resp_lib.POST, _ACCEPT_URL, json={"detail": "No pending invitation"}, status=404)
    with pytest.raises(requests.HTTPError):
        client.accept_team_invitation("smith-lab")


@resp_lib.activate
def test_remove_team_member_success(client):
    resp_lib.add(resp_lib.DELETE, _MEMBER_URL, status=204)
    client.remove_team_member("smith-lab", 42)
    assert resp_lib.calls[0].request.method == "DELETE"


@resp_lib.activate
def test_remove_team_member_403_raises(client):
    resp_lib.add(resp_lib.DELETE, _MEMBER_URL, json={"detail": "Forbidden"}, status=403)
    with pytest.raises(requests.HTTPError):
        client.remove_team_member("smith-lab", 42)


@resp_lib.activate
def test_change_team_member_role_returns_member(client):
    updated = {**_MEMBER_JSON, "role": "admin"}
    resp_lib.add(resp_lib.PATCH, _MEMBER_URL, json=updated, status=200)
    member = client.change_team_member_role("smith-lab", 42, "admin")
    assert member.role == "admin"


@resp_lib.activate
def test_change_team_member_role_sends_payload(client):
    resp_lib.add(resp_lib.PATCH, _MEMBER_URL, json=_MEMBER_JSON, status=200)
    client.change_team_member_role("smith-lab", 42, "owner")
    body = json.loads(resp_lib.calls[0].request.body)
    assert body["role"] == "owner"


@resp_lib.activate
def test_change_team_member_role_403_raises(client):
    resp_lib.add(resp_lib.PATCH, _MEMBER_URL, json={"detail": "Forbidden"}, status=403)
    with pytest.raises(requests.HTTPError):
        client.change_team_member_role("smith-lab", 42, "admin")


@resp_lib.activate
def test_assign_dataset_team_sets_team(client):
    ds_json = {
        "id": 1,
        "accession": "STD0001",
        "title": "Test",
        "abstract": "...",
        "status": "draft",
        "organism": None,
        "tissue": None,
        "technology": None,
        "disease": None,
        "submitter_id": 1,
        "access_type": "open",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "team_id": 1,
    }
    resp_lib.add(resp_lib.PATCH, _ASSIGN_TEAM_URL, json=ds_json, status=200)
    ds = client.assign_dataset_team("STD0001", team_id=1)
    assert ds.accession == "STD0001"
    body = json.loads(resp_lib.calls[0].request.body)
    assert body["team_id"] == 1


@resp_lib.activate
def test_assign_dataset_team_removes_team(client):
    ds_json = {
        "id": 1,
        "accession": "STD0001",
        "title": "Test",
        "abstract": "...",
        "status": "draft",
        "organism": None,
        "tissue": None,
        "technology": None,
        "disease": None,
        "submitter_id": 1,
        "access_type": "open",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "team_id": None,
    }
    resp_lib.add(resp_lib.PATCH, _ASSIGN_TEAM_URL, json=ds_json, status=200)
    client.assign_dataset_team("STD0001", team_id=None)
    body = json.loads(resp_lib.calls[0].request.body)
    assert body["team_id"] is None


@resp_lib.activate
def test_assign_dataset_team_403_raises(client):
    resp_lib.add(resp_lib.PATCH, _ASSIGN_TEAM_URL, json={"detail": "Forbidden"}, status=403)
    with pytest.raises(requests.HTTPError):
        client.assign_dataset_team("STD0001", team_id=1)


# ===========================================================================
# Programmatic submission (Feature 28)
# ===========================================================================

_DS_JSON = {
    "id": 1,
    "accession": "STD0042",
    "title": "My Atlas",
    "abstract": "...",
    "status": "draft",
    "organism": None,
    "tissue": None,
    "technology": None,
    "disease": None,
    "submitter_id": 1,
    "access_type": "open",
    "created_at": "2026-06-16T00:00:00Z",
    "updated_at": "2026-06-16T00:00:00Z",
    "team_id": None,
}
_SAMPLE_JSON = {
    "id": 7,
    "dataset_id": 1,
    "name": "Section 1",
    "title": "Section 1",
    "description": None,
    "section_id": "A1",
}
_FILE_JSON = {
    "id": 99,
    "dataset_id": 1,
    "sample_id": None,
    "filename": "matrix.h5ad",
    "file_type": "expression_matrix",
    "size_bytes": 1024,
    "md5_checksum": None,
    "storage_path": "STD0042/matrix.h5ad",
    "upload_status": "complete",
    "created_at": "2026-06-16T00:00:00Z",
    "download_url": None,
    "file_metadata": None,
}
_SUBMIT_DS_JSON = {**_DS_JSON, "status": "under_review"}


# --- create_dataset ---


def _vocab(resp_lib_module):
    resp_lib_module.add(
        resp_lib_module.GET,
        f"{BASE}/api/v1/vocabulary/terms",
        json=[
            {"id": 1, "ontology": "NCBITaxon", "term_id": "10090", "label": "Mus musculus"},
            {"id": 2, "ontology": "UBERON", "term_id": "0000956", "label": "brain cortex"},
            {"id": 4, "ontology": "MONDO", "term_id": "0004992", "label": "cancer"},
        ],
    )
    resp_lib_module.add(
        resp_lib_module.GET,
        f"{BASE}/api/v1/vocabulary/technologies",
        json=[{"id": 3, "name": "10x Visium", "vendor": "10x Genomics"}],
    )


def _posted_body():
    posts = [c for c in resp_lib.calls if c.request.method == "POST"]
    return json.loads(posts[0].request.body)


@resp_lib.activate
def test_create_dataset_success(client):
    _vocab(resp_lib)
    resp_lib.add(resp_lib.POST, f"{BASE}/api/v1/datasets", json=_DS_JSON, status=201)
    ds = client.create_dataset(
        title="My Atlas",
        abstract="...",
        organism="Mus musculus",
        tissue="brain cortex",
        technology="10x Visium",
    )
    assert ds.accession == "STD0042"
    assert ds.status == "draft"


@resp_lib.activate
def test_create_dataset_with_disease_and_embargo(client):
    _vocab(resp_lib)
    resp_lib.add(resp_lib.POST, f"{BASE}/api/v1/datasets", json=_DS_JSON, status=201)
    client.create_dataset(
        title="My Atlas",
        abstract="...",
        organism="Mus musculus",
        tissue="brain cortex",
        technology="10x Visium",
        disease="cancer",
        embargo_days=180,
    )
    body = _posted_body()
    assert body["disease_id"] == 4
    assert body["embargo_days"] == 180


@resp_lib.activate
def test_create_dataset_sends_ids_the_api_reads(client):
    _vocab(resp_lib)
    resp_lib.add(resp_lib.POST, f"{BASE}/api/v1/datasets", json=_DS_JSON, status=201)
    client.create_dataset(
        title="T",
        abstract="A",
        organism="Mus musculus",
        tissue="brain cortex",
        technology="10x Visium",
    )
    body = _posted_body()
    assert body["title"] == "T"
    assert body["abstract"] == "A"
    assert (body["organism_id"], body["tissue_id"], body["technology_id"]) == (1, 2, 3)
    assert "organism" not in body
    assert "disease_id" not in body


@resp_lib.activate
def test_create_dataset_accepts_ids_without_lookups(client):
    resp_lib.add(resp_lib.POST, f"{BASE}/api/v1/datasets", json=_DS_JSON, status=201)
    client.create_dataset(title="T", abstract="A", organism=1, tissue="2", technology=3)
    assert len(resp_lib.calls) == 1
    body = _posted_body()
    assert (body["organism_id"], body["tissue_id"], body["technology_id"]) == (1, 2, 3)


@resp_lib.activate
def test_create_dataset_unknown_name_creates_nothing(client):
    _vocab(resp_lib)
    with pytest.raises(ValueError, match="unknown organism 'Danio rerio'"):
        client.create_dataset(
            title="T", abstract="A", organism="Danio rerio", tissue="brain cortex", technology=3
        )
    assert all(c.request.method == "GET" for c in resp_lib.calls)


@resp_lib.activate
def test_create_dataset_401_raises(client):
    resp_lib.add(
        resp_lib.POST, f"{BASE}/api/v1/datasets", json={"detail": "Unauthorized"}, status=401
    )
    with pytest.raises(requests.HTTPError):
        client.create_dataset(title="T", abstract="A", organism=1, tissue=2, technology=3)


# --- add_sample ---


@resp_lib.activate
def test_add_sample_success(client):
    bulk_json = {"inserted": 1, "samples": [_SAMPLE_JSON]}
    resp_lib.add(
        resp_lib.POST, f"{BASE}/api/v1/datasets/STD0042/samples", json=bulk_json, status=201
    )
    s = client.add_sample("STD0042", title="Section 1", section_id="A1")
    assert s.name == "Section 1"


@resp_lib.activate
def test_add_sample_minimal(client):
    bulk_json = {
        "inserted": 1,
        "samples": [
            {
                "id": 1,
                "dataset_id": 1,
                "name": "S1",
                "title": "S1",
                "description": None,
                "section_id": None,
            }
        ],
    }
    resp_lib.add(
        resp_lib.POST, f"{BASE}/api/v1/datasets/STD0042/samples", json=bulk_json, status=201
    )
    s = client.add_sample("STD0042", title="S1")
    assert s.name == "S1"


@resp_lib.activate
def test_add_sample_sends_array(client):
    bulk_json = {"inserted": 1, "samples": [_SAMPLE_JSON]}
    resp_lib.add(
        resp_lib.POST, f"{BASE}/api/v1/datasets/STD0042/samples", json=bulk_json, status=201
    )
    client.add_sample("STD0042", title="Section 1", description="desc", section_id="A1")
    body = json.loads(resp_lib.calls[0].request.body)
    assert isinstance(body, list)
    assert body[0]["title"] == "Section 1"
    assert body[0]["description"] == "desc"
    assert body[0]["section_id"] == "A1"


@resp_lib.activate
def test_add_sample_empty_response_raises(client):
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/samples",
        json={"inserted": 0, "samples": []},
        status=201,
    )
    with pytest.raises(RuntimeError):
        client.add_sample("STD0042", title="S1")


@resp_lib.activate
def test_add_sample_403_raises(client):
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/samples",
        json={"detail": "Forbidden"},
        status=403,
    )
    with pytest.raises(requests.HTTPError):
        client.add_sample("STD0042", title="S1")


# --- add_samples ---


@resp_lib.activate
def test_add_samples_bulk(client):
    from stomics.models import BulkSampleResponse

    bulk_json = {
        "inserted": 2,
        "samples": [
            {
                "id": 1,
                "dataset_id": 1,
                "name": "S1",
                "title": "S1",
                "description": None,
                "section_id": None,
            },
            {
                "id": 2,
                "dataset_id": 1,
                "name": "S2",
                "title": "S2",
                "description": None,
                "section_id": None,
            },
        ],
    }
    resp_lib.add(
        resp_lib.POST, f"{BASE}/api/v1/datasets/STD0042/samples", json=bulk_json, status=201
    )
    result = client.add_samples("STD0042", [{"title": "S1"}, {"title": "S2"}])
    assert isinstance(result, BulkSampleResponse)
    assert result.inserted == 2
    assert len(result.samples) == 2


@resp_lib.activate
def test_add_samples_sends_payload(client):
    bulk_json = {"inserted": 1, "samples": [_SAMPLE_JSON]}
    resp_lib.add(
        resp_lib.POST, f"{BASE}/api/v1/datasets/STD0042/samples", json=bulk_json, status=201
    )
    client.add_samples("STD0042", [{"title": "S1", "section_id": "A1"}])
    body = json.loads(resp_lib.calls[0].request.body)
    assert body[0]["title"] == "S1"


@resp_lib.activate
def test_add_samples_422_raises(client):
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/samples",
        json={"detail": "error"},
        status=422,
    )
    with pytest.raises(requests.HTTPError):
        client.add_samples("STD0042", [])


# --- upload_file ---


@resp_lib.activate
def test_upload_file_success(client, tmp_path):
    local = tmp_path / "matrix.h5ad"
    local.write_bytes(b"fake matrix data")

    # Mock the upload-url endpoint
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/files/upload-url",
        json={"upload_url": "https://s3.example.com/upload", "file_id": 99},
        status=201,
    )
    # Mock the S3 PUT
    resp_lib.add(resp_lib.PUT, "https://s3.example.com/upload", body=b"", status=200)
    # Mock the complete endpoint
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/files/99/complete",
        json=_FILE_JSON,
        status=200,
    )

    f = client.upload_file(
        "STD0042", str(local), file_type="expression_matrix", show_progress=False
    )
    assert f.id == 99
    assert f.filename == "matrix.h5ad"
    assert f.upload_status == "complete"


@resp_lib.activate
def test_upload_file_with_progress(client, tmp_path):
    local = tmp_path / "matrix.h5ad"
    local.write_bytes(b"fake data")

    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/files/upload-url",
        json={"upload_url": "https://s3.example.com/upload", "file_id": 99},
        status=201,
    )
    resp_lib.add(resp_lib.PUT, "https://s3.example.com/upload", body=b"", status=200)
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/files/99/complete",
        json=_FILE_JSON,
        status=200,
    )

    with patch("tqdm.tqdm") as mock_tqdm:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_tqdm.return_value = mock_ctx
        f = client.upload_file(
            "STD0042", str(local), file_type="expression_matrix", show_progress=True
        )
    assert f.id == 99


@resp_lib.activate
def test_upload_file_no_tqdm_fallback(client, tmp_path):
    """When tqdm is unavailable, falls back to plain read."""
    local = tmp_path / "matrix.h5ad"
    local.write_bytes(b"data")

    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/files/upload-url",
        json={"upload_url": "https://s3.example.com/upload", "file_id": 99},
        status=201,
    )
    resp_lib.add(resp_lib.PUT, "https://s3.example.com/upload", body=b"", status=200)
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/files/99/complete",
        json=_FILE_JSON,
        status=200,
    )

    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "tqdm":
            raise ImportError("no tqdm")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        f = client.upload_file(
            "STD0042", str(local), file_type="expression_matrix", show_progress=True
        )
    assert f.id == 99


@resp_lib.activate
def test_upload_file_no_progress_false(client, tmp_path):
    """show_progress=False skips tqdm entirely."""
    local = tmp_path / "matrix.h5ad"
    local.write_bytes(b"data")

    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/files/upload-url",
        json={"upload_url": "https://s3.example.com/upload", "file_id": 42},
        status=201,
    )
    resp_lib.add(resp_lib.PUT, "https://s3.example.com/upload", body=b"", status=200)
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/files/42/complete",
        json={**_FILE_JSON, "id": 42},
        status=200,
    )
    f = client.upload_file("STD0042", str(local), file_type="tissue_image", show_progress=False)
    assert f.id == 42


@resp_lib.activate
def test_upload_file_url_error_raises(client, tmp_path):
    local = tmp_path / "f.h5ad"
    local.write_bytes(b"data")
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/files/upload-url",
        json={"detail": "Forbidden"},
        status=403,
    )
    with pytest.raises(requests.HTTPError):
        client.upload_file(
            "STD0042", str(local), file_type="expression_matrix", show_progress=False
        )


@resp_lib.activate
def test_upload_file_storage_put_failure_raises(client, tmp_path):
    local = tmp_path / "f.h5ad"
    local.write_bytes(b"data")
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/STD0042/files/upload-url",
        json={"upload_url": "https://s3.example.com/upload", "file_id": 99},
        status=201,
    )
    resp_lib.add(resp_lib.PUT, "https://s3.example.com/upload", body=b"", status=403)
    with pytest.raises(requests.HTTPError):
        client.upload_file(
            "STD0042", str(local), file_type="expression_matrix", show_progress=False
        )
    # The upload is never registered as complete.
    assert not any(c.request.url.endswith("/complete") for c in resp_lib.calls)


# --- submit ---


@resp_lib.activate
def test_submit_success(client):
    resp_lib.add(
        resp_lib.POST, f"{BASE}/api/v1/datasets/STD0042/submit", json=_SUBMIT_DS_JSON, status=200
    )
    ds = client.submit("STD0042")
    assert ds.status == "under_review"
    assert ds.accession == "STD0042"


@resp_lib.activate
def test_submit_404_raises(client):
    resp_lib.add(
        resp_lib.POST,
        f"{BASE}/api/v1/datasets/INVALID/submit",
        json={"detail": "Not Found"},
        status=404,
    )
    with pytest.raises(requests.HTTPError):
        client.submit("INVALID")


# ===========================================================================
# list_vocabulary
# ===========================================================================


@resp_lib.activate
def test_list_vocabulary_technologies(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/vocabulary/technologies",
        json=[{"id": 3, "name": "10x Visium", "vendor": "10x Genomics"}],
    )
    terms = client.list_vocabulary("technology")
    assert terms == [
        VocabularyTerm(id=3, label="10x Visium", kind="technology", vendor="10x Genomics")
    ]
    assert "?" not in resp_lib.calls[0].request.url


@resp_lib.activate
def test_list_vocabulary_terms_with_query(client):
    resp_lib.add(
        resp_lib.GET,
        f"{BASE}/api/v1/vocabulary/terms",
        json=[{"id": 9, "ontology": "UBERON", "term_id": "0002107", "label": "liver"}],
    )
    terms = client.list_vocabulary("tissue", query="liv")
    assert terms[0].label == "liver"
    assert terms[0].ontology == "UBERON"
    assert terms[0].term_id == "0002107"
    url = resp_lib.calls[0].request.url
    assert "kind=tissue" in url and "q=liv" in url


def test_list_vocabulary_rejects_unknown_kind(client):
    with pytest.raises(ValueError, match="kind must be one of"):
        client.list_vocabulary("species")
