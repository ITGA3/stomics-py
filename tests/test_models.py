from stomics.models import (
    Dataset,
    EmbeddingComputeResult,
    EmbeddingComputeStatus,
    EmbeddingCoordinate,
    EmbeddingCoords,
    EmbeddingInfo,
    EmbeddingUploadResult,
    File,
    FileMetadata,
    GeneAutocompleteSuggestion,
    GeneSearchResponse,
    GeneSearchResult,
    Sample,
)


def test_dataset_defaults():
    d = Dataset(accession="STD0001", title="My Dataset")
    assert d.accession == "STD0001"
    assert d.title == "My Dataset"
    assert d.abstract is None
    assert d.status == "draft"
    assert d.download_count == 0
    assert d.published_at is None
    assert d.created_at is None
    assert d.updated_at is None
    assert d.submitter_id is None
    assert d.organism_id is None
    assert d.tissue_id is None
    assert d.disease_id is None
    assert d.technology_id is None
    assert d.collection_id is None


def test_dataset_full():
    d = Dataset(
        accession="STD0002",
        title="Brain Atlas",
        abstract="A spatial brain atlas.",
        status="published",
        submitter_id=5,
        organism_id=1,
        tissue_id=2,
        disease_id=3,
        technology_id=4,
        collection_id=10,
        download_count=42,
        published_at="2025-01-01T00:00:00Z",
        created_at="2024-12-01T00:00:00Z",
        updated_at="2025-01-02T00:00:00Z",
    )
    assert d.status == "published"
    assert d.download_count == 42
    assert d.collection_id == 10


def test_file_defaults():
    f = File(
        id=1,
        dataset_id=10,
        filename="matrix.h5ad",
        file_type="expression_matrix",
        upload_status="complete",
        created_at="2025-01-01T00:00:00Z",
    )
    assert f.id == 1
    assert f.sample_id is None
    assert f.size_bytes is None
    assert f.md5_checksum is None
    assert f.storage_path is None
    assert f.download_url is None


def test_file_with_download_url():
    f = File(
        id=2,
        dataset_id=10,
        filename="coords.csv",
        file_type="spatial_coordinates",
        upload_status="complete",
        created_at="2025-01-01T00:00:00Z",
        size_bytes=1024,
        download_url="https://example.com/presigned/coords.csv",
    )
    assert f.download_url == "https://example.com/presigned/coords.csv"
    assert f.size_bytes == 1024


def test_sample_defaults():
    s = Sample(id=1, dataset_id=5, name="Sample A")
    assert s.id == 1
    assert s.dataset_id == 5
    assert s.name == "Sample A"
    assert s.description is None


def test_sample_with_description():
    s = Sample(id=2, dataset_id=5, name="Sample B", description="Brain section")
    assert s.description == "Brain section"


# ---------------------------------------------------------------------------
# Embedding models (Feature 22)
# ---------------------------------------------------------------------------


def test_embedding_info_required_fields():
    e = EmbeddingInfo(id=1, method="UMAP", n_components=2, created_at="2026-06-01T00:00:00Z")
    assert e.id == 1
    assert e.method == "UMAP"
    assert e.n_components == 2
    assert e.parameters is None
    assert e.compute_status is None


def test_embedding_info_with_all_fields():
    e = EmbeddingInfo(
        id=3,
        method="tSNE",
        n_components=2,
        created_at="2026-06-01T00:00:00Z",
        parameters={"perplexity": 30},
        compute_status="ready",
    )
    assert e.parameters == {"perplexity": 30}
    assert e.compute_status == "ready"


def test_embedding_coordinate_defaults():
    c = EmbeddingCoordinate(barcode="BARCODE_r1c1", dim_1=0.1, dim_2=-0.2)
    assert c.barcode == "BARCODE_r1c1"
    assert c.dim_1 == 0.1
    assert c.dim_2 == -0.2
    assert c.dim_3 is None


def test_embedding_coordinate_3d():
    c = EmbeddingCoordinate(barcode="BARCODE_r2c2", dim_1=0.5, dim_2=0.3, dim_3=0.7)
    assert c.dim_3 == 0.7


def test_embedding_coords_defaults():
    coords = EmbeddingCoords(embedding_id=1, method="UMAP", n_components=2)
    assert coords.embedding_id == 1
    assert coords.method == "UMAP"
    assert coords.spots == []


def test_embedding_coords_with_spots():
    spots = [
        EmbeddingCoordinate(barcode="BARCODE_r1c1", dim_1=0.1, dim_2=0.2),
        EmbeddingCoordinate(barcode="BARCODE_r2c2", dim_1=1.5, dim_2=0.8),
    ]
    coords = EmbeddingCoords(embedding_id=2, method="tSNE", n_components=2, spots=spots)
    assert len(coords.spots) == 2
    assert coords.spots[0].barcode == "BARCODE_r1c1"


def test_embedding_upload_result():
    r = EmbeddingUploadResult(embedding_id=5, inserted=100, sample_id=1)
    assert r.embedding_id == 5
    assert r.inserted == 100
    assert r.sample_id == 1


def test_embedding_compute_result():
    r = EmbeddingComputeResult(embedding_id=7, status="pending")
    assert r.embedding_id == 7
    assert r.status == "pending"


def test_embedding_compute_status_not_started():
    s = EmbeddingComputeStatus(status="not_started")
    assert s.status == "not_started"
    assert s.embedding_id is None


def test_embedding_compute_status_with_id():
    s = EmbeddingComputeStatus(status="computing", embedding_id=7)
    assert s.embedding_id == 7
    assert s.status == "computing"


def test_file_metadata_defaults():
    fm = FileMetadata()
    assert fm.format is None
    assert fm.n_obs is None
    assert fm.n_vars is None
    assert fm.has_spatial_obsm is None
    assert fm.parse_error is None


def test_file_metadata_full():
    fm = FileMetadata(
        format="h5ad",
        n_obs=3522,
        n_vars=18000,
        obsm_keys=["spatial", "X_umap"],
        has_spatial_obsm=True,
        has_umap_obsm=True,
        has_pca_obsm=False,
        x_min=204.5,
        x_max=6812.0,
        y_min=117.3,
        y_max=7984.5,
    )
    assert fm.n_obs == 3522
    assert fm.n_vars == 18000
    assert fm.has_spatial_obsm is True
    assert fm.has_umap_obsm is True
    assert fm.has_pca_obsm is False
    assert fm.x_min == 204.5
    assert fm.obsm_keys == ["spatial", "X_umap"]


def test_file_with_metadata():
    fm = FileMetadata(n_obs=100, parse_error=False)
    f = File(
        id=1,
        dataset_id=2,
        filename="matrix.h5ad",
        file_type="expression_matrix",
        upload_status="complete",
        created_at="2025-01-01T00:00:00Z",
        file_metadata=fm,
    )
    assert f.file_metadata is fm
    assert f.file_metadata.n_obs == 100


def test_file_metadata_parse_error():
    fm = FileMetadata(parse_error=True, error="Could not read file")
    assert fm.parse_error is True
    assert fm.error == "Could not read file"


# ---------------------------------------------------------------------------
# Gene-Centric Cross-Dataset Expression Search models (Feature 26)
# ---------------------------------------------------------------------------


def test_gene_search_result_required_fields():
    r = GeneSearchResult(
        dataset_accession="STD0001",
        dataset_title="Brain Atlas",
        sample_id=1,
        mean_expr=4.82,
        pct_spots_expressed=0.91,
        max_expr=12.3,
    )
    assert r.dataset_accession == "STD0001"
    assert r.dataset_title == "Brain Atlas"
    assert r.sample_id == 1
    assert r.mean_expr == 4.82
    assert r.pct_spots_expressed == 0.91
    assert r.max_expr == 12.3


def test_gene_search_result_optional_defaults_to_none():
    r = GeneSearchResult(
        dataset_accession="STD0001",
        dataset_title="T",
        sample_id=1,
        mean_expr=1.0,
        pct_spots_expressed=0.5,
        max_expr=5.0,
    )
    assert r.organism is None
    assert r.tissue is None
    assert r.technology is None
    assert r.thumbnail_url is None


def test_gene_search_result_optional_fields():
    r = GeneSearchResult(
        dataset_accession="STD0001",
        dataset_title="T",
        sample_id=1,
        mean_expr=1.0,
        pct_spots_expressed=0.5,
        max_expr=5.0,
        organism="Homo sapiens",
        tissue="brain cortex",
        technology="10x Visium",
        thumbnail_url="https://example.com/thumb.jpg",
    )
    assert r.organism == "Homo sapiens"
    assert r.tissue == "brain cortex"
    assert r.technology == "10x Visium"
    assert r.thumbnail_url == "https://example.com/thumb.jpg"


def test_gene_search_response_fields():
    item = GeneSearchResult(
        dataset_accession="STD0001",
        dataset_title="T",
        sample_id=1,
        mean_expr=1.0,
        pct_spots_expressed=0.5,
        max_expr=5.0,
    )
    resp = GeneSearchResponse(items=[item], total=1, page=1, per_page=20)
    assert resp.total == 1
    assert resp.page == 1
    assert resp.per_page == 20
    assert len(resp.items) == 1
    assert resp.items[0] is item


def test_gene_search_response_empty_items():
    resp = GeneSearchResponse(items=[], total=0, page=1, per_page=20)
    assert resp.items == []
    assert resp.total == 0


def test_gene_autocomplete_suggestion_fields():
    s = GeneAutocompleteSuggestion(gene_symbol="GAPDH", dataset_count=5)
    assert s.gene_symbol == "GAPDH"
    assert s.dataset_count == 5


def test_gene_autocomplete_suggestion_singular():
    s = GeneAutocompleteSuggestion(gene_symbol="TP53", dataset_count=1)
    assert s.dataset_count == 1
