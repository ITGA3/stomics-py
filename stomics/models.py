from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FileMetadata:
    """Metadata automatically extracted from an uploaded file (Feature 24).

    Populated by the background extraction service after a file is uploaded.
    ``parse_error`` is set to ``True`` when the file itself could not be read;
    ``extraction_unavailable`` when the server lacks the parser for the format,
    in which case the file is not implicated and may be re-extracted later.
    """

    format: Optional[str] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None
    obsm_keys: Optional[list] = None
    has_spatial_obsm: Optional[bool] = None
    has_umap_obsm: Optional[bool] = None
    has_pca_obsm: Optional[bool] = None
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    barcode_column: Optional[str] = None
    scalefactors: Optional[dict] = None
    has_tissue_image: Optional[bool] = None
    parse_error: Optional[bool] = None
    extraction_unavailable: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class File:
    id: int
    dataset_id: int
    filename: str
    file_type: str
    upload_status: str
    created_at: str
    sample_id: Optional[int] = None
    size_bytes: Optional[int] = None
    md5_checksum: Optional[str] = None
    storage_path: Optional[str] = None
    download_url: Optional[str] = None
    file_metadata: Optional[FileMetadata] = None


@dataclass
class Sample:
    id: int
    dataset_id: int
    name: str
    description: Optional[str] = None


@dataclass
class Dataset:
    accession: str
    title: str
    abstract: Optional[str] = None
    status: str = "draft"
    submitter_id: Optional[int] = None
    organism_id: Optional[int] = None
    tissue_id: Optional[int] = None
    disease_id: Optional[int] = None
    technology_id: Optional[int] = None
    collection_id: Optional[int] = None
    qc_status: str = "not_run"
    download_count: int = 0
    published_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    current_version: Optional[int] = None
    doi: Optional[str] = None
    access_type: str = "open"
    # Named classification, as returned by the API since the IDs alone could
    # not be read without a second lookup. ``technology`` is
    # ``{"id", "name", "vendor"}``; the three ontology fields are
    # ``{"id", "ontology", "term_id", "label"}``.
    technology: Optional[dict] = None
    organism: Optional[dict] = None
    tissue: Optional[dict] = None
    disease: Optional[dict] = None


class DatasetPage(list):
    """One page of :meth:`StomicsClient.search` results.

    A plain list of :class:`Dataset`, so existing code that iterates or indexes
    the result keeps working, plus the paging totals the API reports. Without
    ``total`` a caller cannot tell the last page from a full one.
    """

    def __init__(self, items, total: int, page: int, per_page: int) -> None:
        super().__init__(items)
        self.total = total
        self.page = page
        self.per_page = per_page


@dataclass
class VocabularyTerm:
    """A technology or ontology term that datasets are classified with."""

    id: int
    label: str
    kind: str
    ontology: Optional[str] = None
    term_id: Optional[str] = None
    vendor: Optional[str] = None


@dataclass
class DatasetVersion:
    id: int
    dataset_accession: str
    version_number: int
    changelog: str
    published_at: str
    created_by: Optional[int] = None
    file_snapshot: Optional[list] = None
    versioned_doi: Optional[str] = None


@dataclass
class QCCheckResult:
    name: str
    label: str
    status: str
    message: str
    metric: Any = None
    threshold: Any = None


@dataclass
class QCFileReport:
    file_id: int
    filename: str
    file_type: str
    checks: list = field(default_factory=list)


@dataclass
class QCReport:
    id: int
    dataset_accession: str
    status: str
    created_at: str
    sample_id: Optional[int] = None
    report_json: Optional[dict] = None
    completed_at: Optional[str] = None
    curator_override: Optional[str] = None


@dataclass
class QCReportList:
    reports: list = field(default_factory=list)
    dataset_qc_status: str = "not_run"


@dataclass
class DEGeneResult:
    gene: str
    mean_a: float
    mean_b: float
    log2fc: float
    pvalue: float
    padj: float


@dataclass
class DifferentialExpressionResult:
    genes: list = field(default_factory=list)
    n_spots_a: int = 0
    n_spots_b: int = 0
    method: str = "wilcoxon"
    subsampled: bool = False


# ---------------------------------------------------------------------------
# Gene Sets and Module Score (Feature 16)
# ---------------------------------------------------------------------------


@dataclass
class GeneSet:
    id: int
    name: str
    source: str
    gene_symbols: list = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class GeneSetList:
    gene_sets: list = field(default_factory=list)
    total: int = 0


@dataclass
class ModuleScoreSpot:
    barcode: str
    x: float
    y: float
    score: float


@dataclass
class ModuleScoreResult:
    spots: list = field(default_factory=list)
    method: str = "mean_normalized"
    gene_set_name: str = ""
    n_genes_total: int = 0
    n_genes_matched: int = 0
    matched_genes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Community Comments and Q&A (Feature 19)
# ---------------------------------------------------------------------------


@dataclass
class CommentAuthor:
    id: int
    full_name: str
    institution: Optional[str] = None


@dataclass
class CommentReactions:
    helpful: int = 0
    flag: int = 0


@dataclass
class Comment:
    id: int
    dataset_accession: str
    user_id: int
    body: str
    is_deleted: bool
    is_pinned: bool
    created_at: str
    updated_at: str
    parent_id: Optional[int] = None
    author: Optional[CommentAuthor] = None
    reactions: Optional[CommentReactions] = None
    reply_count: int = 0
    replies: list = field(default_factory=list)
    user_reaction: Optional[str] = None


@dataclass
class CommentList:
    items: list = field(default_factory=list)
    total: int = 0


@dataclass
class LaunchResult:
    launch_url: str
    platform: str


@dataclass
class ReviewerToken:
    id: str
    dataset_accession: str
    label: str
    token_prefix: str
    expires_at: str
    access_count: int
    created_at: str
    last_used_at: Optional[str] = None
    revoked_at: Optional[str] = None
    token: Optional[str] = None


# ---------------------------------------------------------------------------
# UMAP / Embedding Visualization (Feature 22)
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingInfo:
    id: int
    method: str
    n_components: int
    created_at: str
    parameters: Optional[dict] = None
    compute_status: Optional[str] = None


@dataclass
class EmbeddingCoordinate:
    barcode: str
    dim_1: float
    dim_2: float
    dim_3: Optional[float] = None


@dataclass
class EmbeddingCoords:
    embedding_id: int
    method: str
    n_components: int
    spots: list = field(default_factory=list)


@dataclass
class EmbeddingUploadResult:
    embedding_id: int
    inserted: int
    sample_id: int


@dataclass
class EmbeddingComputeResult:
    embedding_id: int
    status: str


@dataclass
class EmbeddingComputeStatus:
    status: str
    embedding_id: Optional[int] = None


@dataclass
class SvgGeneResult:
    gene: str
    morans_i: float
    pvalue: float
    padj: float
    mean_expr: float
    n_spots_expressed: int


@dataclass
class SvgResult:
    genes: list
    method: str
    n_genes_tested: int
    sample_id: int
    subsampled: bool
    n_spots: int
    cached: bool


@dataclass
class ConversionJob:
    job_id: str
    status: str
    target_format: str
    size_bytes: int | None = None
    url: str | None = None


# ---------------------------------------------------------------------------
# Gene-Centric Cross-Dataset Expression Search (Feature 26)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Teams (Feature 27)
# ---------------------------------------------------------------------------


@dataclass
class TeamMember:
    id: int
    team_id: int
    user_id: int
    role: str
    invited_at: str
    full_name: str
    email: str
    accepted_at: str | None = None
    institution: str | None = None


@dataclass
class Team:
    id: int
    name: str
    slug: str
    owner_id: int
    created_at: str
    member_count: int
    dataset_count: int
    description: str | None = None


@dataclass
class TeamList:
    items: list
    total: int
    page: int
    per_page: int


@dataclass
class GeneSearchResult:
    dataset_accession: str
    dataset_title: str
    sample_id: int
    mean_expr: float
    pct_spots_expressed: float
    max_expr: float
    organism: str | None = None
    tissue: str | None = None
    technology: str | None = None
    thumbnail_url: str | None = None


@dataclass
class GeneSearchResponse:
    items: list
    total: int
    page: int
    per_page: int


@dataclass
class GeneAutocompleteSuggestion:
    gene_symbol: str
    dataset_count: int


@dataclass
class BulkSampleResponse:
    """Returned by bulk sample creation (Feature 28)."""

    inserted: int
    samples: list = field(default_factory=list)
