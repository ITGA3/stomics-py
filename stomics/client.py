import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import requests

from . import io as stomics_io
from .models import (
    BulkSampleResponse,
    Comment,
    CommentAuthor,
    CommentList,
    CommentReactions,
    ConversionJob,
    Dataset,
    DatasetPage,
    DatasetVersion,
    DEGeneResult,
    DifferentialExpressionResult,
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
    GeneSet,
    GeneSetList,
    LaunchResult,
    ModuleScoreResult,
    ModuleScoreSpot,
    QCReport,
    QCReportList,
    ReviewerToken,
    Sample,
    SvgGeneResult,
    SvgResult,
    Team,
    TeamList,
    TeamMember,
    VocabularyTerm,
)

_DATASET_FIELDS = set(Dataset.__dataclass_fields__)
_DATASET_VERSION_FIELDS = set(DatasetVersion.__dataclass_fields__)
_FILE_FIELDS = set(File.__dataclass_fields__)
_SAMPLE_FIELDS = set(Sample.__dataclass_fields__)
_QC_REPORT_FIELDS = set(QCReport.__dataclass_fields__)


# (connect, read) seconds. The read timeout applies between bytes rather than
# to the whole exchange, so a slow large transfer is fine while a dead
# connection is not. Without any timeout a request hangs until the process is
# killed, which is the failure mode this replaces.
DEFAULT_TIMEOUT = (10, 60)

# A classification filter: an ID, a name, a comma-separated string of either,
# or a list mixing them.
Filter = Union[int, str, Sequence[Union[int, str]], None]

VOCABULARY_KINDS = ("technology", "organism", "tissue", "disease")

# Used for bulk transfers, which can legitimately stall longer between chunks.
TRANSFER_TIMEOUT = (10, 300)


class _TimeoutSession(requests.Session):
    """Session that applies a default timeout to every request.

    ``requests`` has no session-level timeout, so without this each of the
    ~30 call sites has to remember one, and any new call site silently gets no
    timeout at all. Callers can still pass ``timeout=`` explicitly to override.
    """

    def __init__(self, timeout) -> None:
        super().__init__()
        self._timeout = timeout

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", self._timeout)
        return super().request(*args, **kwargs)


class StomicsClient:
    """Client for the Stomics.io REST API.

    Parameters
    ----------
    api_key:
        API key for authenticated access. Falls back to the ``STOMICS_API_KEY``
        environment variable, then unauthenticated (public endpoints only).
    base_url:
        Base URL of the Stomics API. Defaults to ``https://api.stomics.io``.
    timeout:
        ``(connect, read)`` timeout applied to every request. Defaults to
        :data:`DEFAULT_TIMEOUT`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.stomics.io",
        timeout=DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("STOMICS_API_KEY")
        self._session = _TimeoutSession(timeout)
        if self.api_key:
            self._session.headers["Authorization"] = f"Bearer {self.api_key}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _build_dataset(data: dict) -> Dataset:
        return Dataset(**{k: v for k, v in data.items() if k in _DATASET_FIELDS})

    @staticmethod
    def _build_dataset_version(data: dict) -> DatasetVersion:
        return DatasetVersion(**{k: v for k, v in data.items() if k in _DATASET_VERSION_FIELDS})

    @staticmethod
    def _build_file(data: dict) -> File:
        d = {k: v for k, v in data.items() if k in _FILE_FIELDS}
        if isinstance(d.get("file_metadata"), dict):
            fm = d["file_metadata"]
            _FM_FIELDS = set(FileMetadata.__dataclass_fields__)
            d["file_metadata"] = FileMetadata(**{k: v for k, v in fm.items() if k in _FM_FIELDS})
        return File(**d)

    @staticmethod
    def _build_sample(data: dict) -> Sample:
        return Sample(**{k: v for k, v in data.items() if k in _SAMPLE_FIELDS})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_vocabulary(self, kind: str, query: Optional[str] = None) -> List[VocabularyTerm]:
        """List the terms datasets are classified with.

        Parameters
        ----------
        kind:   ``"technology"``, ``"organism"``, ``"tissue"`` or ``"disease"``.
        query:  Optional case-insensitive substring to match against the name.

        Returns
        -------
        List of :class:`VocabularyTerm`.
        """
        if kind not in VOCABULARY_KINDS:
            raise ValueError(f"kind must be one of {', '.join(VOCABULARY_KINDS)}; got {kind!r}")
        params: dict = {}
        if query is not None:
            params["q"] = query
        if kind == "technology":
            data = self._get("/api/v1/vocabulary/technologies", params=params or None)
            return [
                VocabularyTerm(id=t["id"], label=t["name"], kind=kind, vendor=t.get("vendor"))
                for t in data
            ]
        params["kind"] = kind
        data = self._get("/api/v1/vocabulary/terms", params=params)
        return [
            VocabularyTerm(
                id=t["id"],
                label=t["label"],
                kind=kind,
                ontology=t.get("ontology"),
                term_id=t.get("term_id"),
            )
            for t in data
        ]

    def _resolve_ids(self, kind: str, value: Filter) -> Optional[List[int]]:
        """Turn a classification filter into the integer IDs the API expects.

        Names are matched case-insensitively and exactly against the vocabulary;
        an unknown name raises rather than silently filtering on nothing.
        """
        if value is None:
            return None
        if isinstance(value, (int, str)):
            parts: list = str(value).split(",") if isinstance(value, str) else [value]
        else:
            parts = list(value)
        ids: List[int] = []
        for part in parts:
            if isinstance(part, int):
                ids.append(part)
                continue
            text = part.strip()
            if text.isdigit():
                ids.append(int(text))
                continue
            matches = [
                t for t in self.list_vocabulary(kind, query=text) if t.label.lower() == text.lower()
            ]
            if not matches:
                raise ValueError(
                    f"unknown {kind} {text!r}; see client.list_vocabulary({kind!r}) for accepted names"
                )
            ids.append(matches[0].id)
        return ids

    def search(
        self,
        query: Optional[str] = None,
        organism: Filter = None,
        tissue: Filter = None,
        disease: Filter = None,
        technology: Filter = None,
        page: int = 1,
        per_page: int = 20,
    ) -> DatasetPage:
        """Search published datasets with optional filters.

        Parameters
        ----------
        query:       Free-text search term matched against accession, title and abstract.
        organism:    Organism filter: a name (``"Homo sapiens"``), an ID, a
                     comma-separated string of either, or a list.
        tissue:      Tissue filter, in the same forms.
        disease:     Disease filter, in the same forms.
        technology:  Technology filter, in the same forms.
        page:        1-based page number.
        per_page:    Results per page (max 100).

        Returns
        -------
        :class:`DatasetPage`: a list of :class:`Dataset` with ``total``,
        ``page`` and ``per_page`` attributes.
        """
        params: dict = {"page": page, "per_page": per_page}
        if query is not None:
            params["q"] = query
        for kind, value in (
            ("organism", organism),
            ("tissue", tissue),
            ("disease", disease),
            ("technology", technology),
        ):
            ids = self._resolve_ids(kind, value)
            if ids is not None:
                # requests encodes a list as a repeated parameter, which is the
                # form FastAPI parses into list[int]. The comma-joined string
                # sent previously was rejected with 422.
                params[f"{kind}_ids"] = ids
        data = self._get("/api/v1/datasets", params=params)
        return DatasetPage(
            [self._build_dataset(item) for item in data["items"]],
            total=data["total"],
            page=data["page"],
            per_page=data["per_page"],
        )

    def get_dataset(self, accession: str, version: Optional[int] = None) -> Dataset:
        """Fetch metadata for a single dataset by accession number.

        Parameters
        ----------
        accession:  Dataset accession, e.g. ``"STD0001"``.
        version:    Specific version number to retrieve. When ``None`` (default),
                    returns the current (latest) version.

        Returns
        -------
        :class:`Dataset`
        """
        params: dict = {}
        if version is not None:
            params["version"] = version
        return self._build_dataset(
            self._get(f"/api/v1/datasets/{accession}", params=params or None)
        )

    def list_versions(self, accession: str) -> List[DatasetVersion]:
        """Return all published versions for a dataset.

        Parameters
        ----------
        accession:  Dataset accession.

        Returns
        -------
        List of :class:`DatasetVersion` objects ordered by version number
        (oldest first).
        """
        data = self._get(f"/api/v1/datasets/{accession}/versions")
        return [self._build_dataset_version(v) for v in data.get("versions", [])]

    def list_files(self, accession: str, version: Optional[int] = None) -> List[File]:
        """List all files for a dataset, including pre-signed download URLs.

        Parameters
        ----------
        accession:  Dataset accession.
        version:    Return files from a specific version snapshot. When ``None``
                    (default), returns the current files.

        Returns
        -------
        List of :class:`File` objects. Each file's ``download_url`` attribute
        contains a pre-signed S3 URL valid for 1 hour.
        """
        params: dict = {}
        if version is not None:
            params["version"] = version
        data = self._get(f"/api/v1/datasets/{accession}/files", params=params or None)
        return [self._build_file(f) for f in data]

    def download_file(
        self,
        accession: str,
        file_id: int,
        dest_path: str,
        show_progress: bool = True,
    ) -> None:
        """Download a file to disk via pre-signed URL.

        Parameters
        ----------
        accession:      Dataset accession.
        file_id:        File ID as returned by :meth:`list_files`.
        dest_path:      Local path to write the file to. Parent directories are
                        created automatically.
        show_progress:  Display a ``tqdm`` progress bar (requires ``tqdm``).
        """
        data = self._get(f"/api/v1/datasets/{accession}/files/{file_id}/download")
        self._stream_to_disk(data["url"], dest_path, show_progress=show_progress)

    def get_file(self, accession: str, file_id: int) -> File:
        """Return metadata for a single file, including extracted ``file_metadata``.

        Parameters
        ----------
        accession:  Dataset accession.
        file_id:    File ID as returned by :meth:`list_files`.
        """
        data = self._get(f"/api/v1/datasets/{accession}/files/{file_id}")
        return self._build_file(data)

    def trigger_metadata_extraction(self, accession: str, file_id: int) -> File:
        """Manually trigger metadata extraction for a file.

        Extraction runs automatically after upload for expression matrices and
        spatial coordinate files. Use this to re-run extraction or to extract
        from files that were uploaded before Feature 24 was deployed.

        Parameters
        ----------
        accession:  Dataset accession.
        file_id:    File ID as returned by :meth:`list_files`.
        """
        url = f"{self.base_url}/api/v1/datasets/{accession}/files/{file_id}/extract"
        resp = self._session.post(url)
        resp.raise_for_status()
        return self._build_file(resp.json())

    def _stream_to_disk(self, url: str, dest_path: str, show_progress: bool = True) -> None:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=TRANSFER_TIMEOUT) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            if show_progress:
                try:
                    from tqdm import tqdm

                    with open(dest_path, "wb") as fh:
                        with tqdm(total=total or None, unit="B", unit_scale=True) as bar:
                            for chunk in resp.iter_content(chunk_size=8192):
                                fh.write(chunk)
                                bar.update(len(chunk))
                    return
                except ImportError:
                    pass
            with open(dest_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)

    def get_expression(
        self,
        accession: str,
        gene: str,
        sample_id: Optional[str] = None,
    ):
        """Fetch per-spot expression values for a single gene.

        Parameters
        ----------
        accession:  Dataset accession.
        gene:       Gene symbol, e.g. ``"BRCA1"``.
        sample_id:  Restrict to a specific sample (optional).

        Returns
        -------
        :class:`pandas.DataFrame` with columns ``x``, ``y``, and ``expression``.
        """
        import pandas as pd

        params: dict = {"gene": gene}
        if sample_id is not None:
            params["sample_id"] = sample_id
        data = self._get(f"/api/v1/datasets/{accession}/expression", params=params)
        return pd.DataFrame(data.get("spots", []))

    def load_anndata(
        self,
        accession: str,
        sample_id: Optional[str] = None,
    ):
        """Download the expression matrix and spatial coordinates and return an AnnData object.

        Files are cached in ``~/.stomics/cache/{accession}/`` so repeated calls
        skip re-downloading.

        Parameters
        ----------
        accession:  Dataset accession.
        sample_id:  Restrict to a specific sample (optional). When ``None``, the
                    first expression matrix found is used.

        Returns
        -------
        :class:`anndata.AnnData` with ``obsm["spatial"]`` populated when spatial
        coordinates are available.

        Raises
        ------
        ImportError:
            If ``anndata`` is not installed.
        ValueError:
            If no expression matrix file exists for the dataset/sample.
        """
        try:
            import anndata  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'anndata is required. Install with: pip install "stomics-py[anndata] @ git+https://github.com/ITGA3/stomics-py"'
            ) from exc

        cache_dir = stomics_io.ensure_cache_dir(accession)
        files = self.list_files(accession)

        def _matches_sample(f: File) -> bool:
            return sample_id is None or str(f.sample_id) == str(sample_id)

        matrix = next(
            (f for f in files if f.file_type == "expression_matrix" and _matches_sample(f)),
            None,
        )
        if matrix is None:
            raise ValueError(
                f"No expression matrix found for accession={accession!r}, sample_id={sample_id!r}."
            )

        matrix_path = cache_dir / matrix.filename
        if not matrix_path.exists():
            if matrix.download_url:
                self._stream_to_disk(matrix.download_url, str(matrix_path), show_progress=False)
            else:
                self.download_file(accession, matrix.id, str(matrix_path), show_progress=False)

        adata = stomics_io.load_h5ad(str(matrix_path))

        coords = next(
            (f for f in files if f.file_type == "spatial_coordinates" and _matches_sample(f)),
            None,
        )
        if coords is not None:
            coords_path = cache_dir / coords.filename
            if not coords_path.exists():
                if coords.download_url:
                    self._stream_to_disk(coords.download_url, str(coords_path), show_progress=False)
                else:
                    self.download_file(accession, coords.id, str(coords_path), show_progress=False)
            try:
                coords_df = stomics_io.load_spatial_csv(str(coords_path))
                stomics_io.attach_spatial(adata, coords_df)
            except (ValueError, Exception):
                pass

        return adata

    # ------------------------------------------------------------------
    # QC Report methods
    # ------------------------------------------------------------------

    @staticmethod
    def _build_qc_report(data: dict) -> QCReport:
        return QCReport(**{k: v for k, v in data.items() if k in _QC_REPORT_FIELDS})

    def get_qc_reports(self, accession: str) -> QCReportList:
        """Return QC reports and overall QC status for a dataset.

        Parameters
        ----------
        accession:
            Dataset accession identifier.

        Returns
        -------
        :class:`QCReportList` with ``reports`` and ``dataset_qc_status``.
        """
        data = self._get(f"/api/v1/datasets/{accession}/qc")
        reports = [self._build_qc_report(r) for r in data.get("reports", [])]
        return QCReportList(
            reports=reports, dataset_qc_status=data.get("dataset_qc_status", "not_run")
        )

    def run_differential_expression(
        self,
        accession: str,
        sample_id: int,
        region_a: List[str],
        region_b: "str | List[str]" = "rest",
        method: str = "wilcoxon",
        n_top_genes: int = 50,
    ) -> DifferentialExpressionResult:
        """Run differential expression analysis between two spatial regions.

        Requires an authenticated client (API key or session token).

        Parameters
        ----------
        accession:    Dataset accession.
        sample_id:    Sample ID to run DE on.
        region_a:     List of spot barcodes defining region A.
        region_b:     List of spot barcodes for region B, or ``"rest"`` (default)
                      to compare against all spots not in region A.
        method:       Statistical test — ``"wilcoxon"`` (default) or ``"t-test"``.
        n_top_genes:  Maximum number of top genes to return (default 50).

        Returns
        -------
        :class:`DifferentialExpressionResult` with a ``genes`` list of
        :class:`DEGeneResult` objects sorted by adjusted p-value.

        Raises
        ------
        requests.HTTPError:
            401 if not authenticated, 429 if rate limit exceeded.
        """
        path = f"/api/v1/datasets/{accession}/samples/{sample_id}/differential-expression"
        url = f"{self.base_url}{path}"
        payload = {
            "region_a": region_a,
            "region_b": region_b,
            "method": method,
            "n_top_genes": n_top_genes,
        }
        resp = self._session.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        genes = [
            DEGeneResult(**{k: v for k, v in g.items() if k in DEGeneResult.__dataclass_fields__})
            for g in data.get("genes", [])
        ]
        return DifferentialExpressionResult(
            genes=genes,
            n_spots_a=data.get("n_spots_a", 0),
            n_spots_b=data.get("n_spots_b", 0),
            method=data.get("method", method),
            subsampled=data.get("subsampled", False),
        )

    def trigger_qc(self, accession: str) -> QCReport:
        """Trigger a new QC run for a dataset.

        Parameters
        ----------
        accession:
            Dataset accession identifier.

        Returns
        -------
        :class:`QCReport` representing the newly created (pending) report.
        """
        resp = self._session.post(f"{self.base_url}/api/v1/datasets/{accession}/qc/run")
        resp.raise_for_status()
        return self._build_qc_report(resp.json())

    # ---------------------------------------------------------------------------
    # Gene Sets and Module Score (Feature 16)
    # ---------------------------------------------------------------------------

    def get_gene_sets(
        self,
        q: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> GeneSetList:
        """List built-in gene sets from the Stomics catalog.

        Parameters
        ----------
        q:
            Search term matched against gene set name (case-insensitive).
        source:
            Filter by source: ``"MSigDB_Hallmark"``, ``"GO_BP"``, ``"KEGG"``,
            or ``"custom"``.
        limit:
            Maximum number of results (default 50, max 200).
        offset:
            Pagination offset.

        Returns
        -------
        :class:`GeneSetList` with a ``gene_sets`` list and ``total`` count.
        """
        params: dict = {"limit": limit, "offset": offset}
        if q is not None:
            params["q"] = q
        if source is not None:
            params["source"] = source
        resp = self._session.get(f"{self.base_url}/api/v1/gene-sets", params=params)
        resp.raise_for_status()
        data = resp.json()
        gene_sets = [
            GeneSet(
                id=gs["id"],
                name=gs["name"],
                source=gs["source"],
                gene_symbols=gs.get("gene_symbols", []),
                description=gs.get("description"),
            )
            for gs in data.get("gene_sets", [])
        ]
        return GeneSetList(gene_sets=gene_sets, total=data.get("total", 0))

    def compute_module_score(
        self,
        accession: str,
        sample_id: int,
        gene_set_id: Optional[int] = None,
        custom_genes: Optional[List[str]] = None,
    ) -> ModuleScoreResult:
        """Compute per-spot pathway/gene set module scores for a dataset sample.

        Exactly one of ``gene_set_id`` or ``custom_genes`` must be provided.

        The score for each spot is computed as the mean expression of matched
        gene set genes minus the mean expression of an equal-sized random
        background gene set (Seurat ``AddModuleScore`` approach).

        Parameters
        ----------
        accession:
            Dataset accession.
        sample_id:
            Sample ID to score.
        gene_set_id:
            ID of a built-in gene set from :meth:`get_gene_sets`.
        custom_genes:
            List of gene symbols for a custom gene set.

        Returns
        -------
        :class:`ModuleScoreResult` with per-spot scores and gene match metadata.

        Raises
        ------
        requests.HTTPError:
            422 if neither or both parameters are provided, 404 if dataset/sample
            not found, 403 if controlled access.
        """
        if gene_set_id is None and custom_genes is None:
            raise ValueError("Exactly one of gene_set_id or custom_genes must be provided")
        if gene_set_id is not None and custom_genes is not None:
            raise ValueError("Provide gene_set_id or custom_genes, not both")

        payload: dict = {}
        if gene_set_id is not None:
            payload["gene_set_id"] = gene_set_id
        if custom_genes is not None:
            payload["custom_genes"] = custom_genes

        url = f"{self.base_url}/api/v1/datasets/{accession}/samples/{sample_id}/module-score"
        resp = self._session.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        spots = [
            ModuleScoreSpot(
                barcode=s["barcode"],
                x=s["x"],
                y=s["y"],
                score=s["score"],
            )
            for s in data.get("spots", [])
        ]
        return ModuleScoreResult(
            spots=spots,
            method=data.get("method", "mean_normalized"),
            gene_set_name=data.get("gene_set_name", ""),
            n_genes_total=data.get("n_genes_total", 0),
            n_genes_matched=data.get("n_genes_matched", 0),
            matched_genes=data.get("matched_genes", []),
        )

    # ---------------------------------------------------------------------------
    # Cross-Dataset Comparative Viewer (Feature 18)
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Community Comments and Q&A (Feature 19)
    # ---------------------------------------------------------------------------

    @staticmethod
    def _build_comment(data: dict) -> Comment:
        author_data = data.get("author")
        author = (
            CommentAuthor(
                id=author_data["id"],
                full_name=author_data["full_name"],
                institution=author_data.get("institution"),
            )
            if author_data
            else None
        )
        rxn_data = data.get("reactions", {})
        reactions = CommentReactions(
            helpful=rxn_data.get("helpful", 0),
            flag=rxn_data.get("flag", 0),
        )
        replies = [StomicsClient._build_comment(r) for r in data.get("replies", [])]
        return Comment(
            id=data["id"],
            dataset_accession=data["dataset_accession"],
            user_id=data["user_id"],
            body=data["body"],
            is_deleted=data.get("is_deleted", False),
            is_pinned=data.get("is_pinned", False),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            parent_id=data.get("parent_id"),
            author=author,
            reactions=reactions,
            reply_count=data.get("reply_count", 0),
            replies=replies,
            user_reaction=data.get("user_reaction"),
        )

    def get_comments(self, accession: str) -> CommentList:
        """Return the threaded comment list for a dataset.

        Parameters
        ----------
        accession:
            Dataset accession identifier.

        Returns
        -------
        :class:`CommentList` with ``items`` (list of :class:`Comment`) and ``total``.
        """
        data = self._get(f"/api/v1/datasets/{accession}/comments")
        items = [self._build_comment(c) for c in data.get("items", [])]
        return CommentList(items=items, total=data.get("total", 0))

    def post_comment(
        self,
        accession: str,
        body: str,
        parent_id: Optional[int] = None,
    ) -> Comment:
        """Post a comment on a dataset.

        Requires an authenticated client (API key).

        Parameters
        ----------
        accession:
            Dataset accession identifier.
        body:
            Comment text. Must be 1–2000 characters, HTML is stripped.
        parent_id:
            ID of the parent comment when posting a reply. ``None`` for top-level
            comments.

        Returns
        -------
        The created :class:`Comment`.
        """
        payload: dict = {"body": body}
        if parent_id is not None:
            payload["parent_id"] = parent_id
        url = f"{self.base_url}/api/v1/datasets/{accession}/comments"
        resp = self._session.post(url, json=payload)
        resp.raise_for_status()
        return self._build_comment(resp.json())

    # ---------------------------------------------------------------------------
    # One-Click Launch to Cloud Analysis Platforms (Feature 20)
    # ---------------------------------------------------------------------------

    def launch_to_platform(
        self,
        accession: str,
        platform: str,
        file_types: Optional[List[str]] = None,
        workflow_id: Optional[str] = None,
    ) -> LaunchResult:
        """Generate a cloud-platform launch URL for a dataset.

        Opens the dataset directly in Terra, Galaxy, or Seqera Platform
        without requiring manual file transfer.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        platform:
            Target platform: ``"terra"``, ``"galaxy"``, or ``"seqera"``.
        file_types:
            List of file types to include, e.g. ``["expression_matrix",
            "spatial_coordinates"]``. When ``None``, all files are included.
        workflow_id:
            Optional Seqera pipeline ID (only used when ``platform="seqera"``).

        Returns
        -------
        :class:`LaunchResult` with ``launch_url`` and ``platform``.

        Raises
        ------
        requests.HTTPError:
            422 if the platform is not configured (Seqera) or no files match.

        Examples
        --------
        >>> result = client.launch_to_platform("STD0001", platform="terra")
        >>> print(result.launch_url)
        https://app.terra.bio/#import-data?url=...
        """
        payload: dict = {"platform": platform}
        if file_types is not None:
            payload["file_types"] = file_types
        if workflow_id is not None:
            payload["workflow_id"] = workflow_id
        url = f"{self.base_url}/api/v1/datasets/{accession}/launch-bundle"
        resp = self._session.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return LaunchResult(launch_url=data["launch_url"], platform=data["platform"])

    def download_aria2c_manifest(
        self,
        accession: str,
        dest_path: str,
        file_types: Optional[List[str]] = None,
    ) -> None:
        """Download an aria2c URL list for parallel file download.

        The output file can be passed directly to
        ``aria2c -i <dest_path> -x 16 -s 16`` for high-speed parallel
        downloading without a ZIP archive overhead.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        dest_path:
            Local path to write the ``.txt`` manifest to.
        file_types:
            Optional list of file types to include. When ``None``, all
            files are included.

        Examples
        --------
        >>> client.download_aria2c_manifest("STD0001", "STD0001-aria2c.txt")
        >>> # Then in a terminal: aria2c -i STD0001-aria2c.txt -x 16 -s 16
        """
        params: dict = {}
        if file_types:
            params["file_types"] = file_types
        url = f"{self.base_url}/api/v1/datasets/{accession}/aria2c-manifest"
        resp = self._session.get(url, params=params)
        resp.raise_for_status()
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as fh:
            fh.write(resp.text)

    def compare_datasets(self, accessions: List[str]) -> List[Dataset]:
        """Fetch metadata for multiple datasets in a single request.

        Designed for use with the Cross-Dataset Comparative Viewer. Returns
        metadata for each supplied accession in the same order as the input
        list.

        Parameters
        ----------
        accessions:
            List of dataset accession strings, e.g. ``["STD0001", "STD0002"]``.
            Maximum 6 accessions per request.

        Returns
        -------
        List of :class:`Dataset` objects in the same order as *accessions*.

        Raises
        ------
        ValueError
            If *accessions* is empty or contains more than 6 entries.
        requests.HTTPError
            If any accession is not found or is not publicly accessible.
        """
        if not accessions:
            raise ValueError("accessions must not be empty")
        if len(accessions) > 6:
            raise ValueError("Maximum 6 accessions can be compared at once")
        data = self._get("/api/v1/datasets/compare", params={"accessions": ",".join(accessions)})
        return [self._build_dataset(item) for item in data["datasets"]]

    # ------------------------------------------------------------------
    # Reviewer anonymous access links
    # ------------------------------------------------------------------

    def create_reviewer_token(
        self,
        accession: str,
        label: str,
        expires_in_days: int = 90,
    ) -> ReviewerToken:
        """Generate a time-limited anonymous access link for peer review.

        Creates a single-use token that allows a journal reviewer to view an
        embargoed or under-review dataset without needing an account. The raw
        token is returned **once** in the response; it cannot be retrieved
        again. Store it securely before sharing with the reviewer.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        label:
            Human-readable label to identify the reviewer, e.g.
            ``"Nature Reviewer #1"``.
        expires_in_days:
            Number of days until the token expires. Must be between 1 and 180.
            Defaults to 90.

        Returns
        -------
        :class:`ReviewerToken`
            Token metadata including the raw ``token`` field (one-time only).

        Raises
        ------
        requests.HTTPError
            If the dataset is published (tokens not allowed), the user is not
            the dataset owner, or authentication fails.

        Examples
        --------
        >>> token = client.create_reviewer_token(
        ...     "STD0001",
        ...     label="Nature Reviewer #1",
        ...     expires_in_days=90,
        ... )
        >>> reviewer_url = f"https://stomics.io/datasets/STD0001?reviewer_token={token.token}"
        >>> print("Share this link:", reviewer_url)
        """
        url = f"{self.base_url}/api/v1/datasets/{accession}/reviewer-tokens"
        resp = self._session.post(url, json={"label": label, "expires_in_days": expires_in_days})
        resp.raise_for_status()
        data = resp.json()
        _fields = set(ReviewerToken.__dataclass_fields__)
        return ReviewerToken(**{k: v for k, v in data.items() if k in _fields})

    def list_reviewer_tokens(self, accession: str) -> List[ReviewerToken]:
        """List all active reviewer tokens for a dataset.

        Returns tokens that have not been revoked, including expired ones so
        you can see their history. Revoked tokens are excluded.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.

        Returns
        -------
        List of :class:`ReviewerToken` objects.

        Raises
        ------
        requests.HTTPError
            If the dataset is not found or the user is not the owner.

        Examples
        --------
        >>> tokens = client.list_reviewer_tokens("STD0001")
        >>> for t in tokens:
        ...     print(t.label, "expires", t.expires_at, "—", t.access_count, "views")
        """
        data = self._get(f"/api/v1/datasets/{accession}/reviewer-tokens")
        _fields = set(ReviewerToken.__dataclass_fields__)
        return [
            ReviewerToken(**{k: v for k, v in item.items() if k in _fields})
            for item in data["items"]
        ]

    def revoke_reviewer_token(self, accession: str, token_id: str) -> ReviewerToken:
        """Revoke a reviewer access token immediately.

        After revocation the token's link stops working. The token record is
        kept for audit purposes but ``revoked_at`` is set to the current time.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        token_id:
            UUID of the token to revoke, as returned by
            :meth:`create_reviewer_token` or :meth:`list_reviewer_tokens`.

        Returns
        -------
        :class:`ReviewerToken`
            Updated token record with ``revoked_at`` populated.

        Raises
        ------
        requests.HTTPError
            If the token is not found or the user is not the dataset owner.

        Examples
        --------
        >>> revoked = client.revoke_reviewer_token("STD0001", token_id="abc-123")
        >>> print("Revoked at:", revoked.revoked_at)
        """
        url = f"{self.base_url}/api/v1/datasets/{accession}/reviewer-tokens/{token_id}"
        resp = self._session.delete(url)
        resp.raise_for_status()
        data = resp.json()
        _fields = set(ReviewerToken.__dataclass_fields__)
        return ReviewerToken(**{k: v for k, v in data.items() if k in _fields})

    # -----------------------------------------------------------------------
    # UMAP / Embedding Visualization (Feature 22)
    # -----------------------------------------------------------------------

    def list_embeddings(self, accession: str, sample_id: int) -> List[EmbeddingInfo]:
        """List available embeddings for a sample.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        sample_id:
            Sample ID whose embeddings to list.

        Returns
        -------
        list of :class:`EmbeddingInfo`
            Embedding metadata records (method, status, etc.).

        Examples
        --------
        >>> embeddings = client.list_embeddings("STD0001", sample_id=1)
        >>> for emb in embeddings:
        ...     print(emb.method, "—", emb.compute_status)
        """
        data = self._get(f"/api/v1/datasets/{accession}/samples/{sample_id}/embeddings")
        _fields = set(EmbeddingInfo.__dataclass_fields__)
        return [
            EmbeddingInfo(**{k: v for k, v in item.items() if k in _fields})
            for item in data["embeddings"]
        ]

    def get_embedding(self, accession: str, sample_id: int, embedding_id: int) -> EmbeddingCoords:
        """Fetch UMAP/tSNE coordinates for a specific embedding.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        sample_id:
            Sample ID.
        embedding_id:
            ID of the embedding to fetch.

        Returns
        -------
        :class:`EmbeddingCoords`
            Embedding coordinates with ``spots`` list of
            :class:`EmbeddingCoordinate`.

        Raises
        ------
        requests.HTTPError
            If embedding not found or still computing.

        Examples
        --------
        >>> coords = client.get_embedding("STD0001", sample_id=1, embedding_id=1)
        >>> print(len(coords.spots), "spots")
        """
        data = self._get(
            f"/api/v1/datasets/{accession}/samples/{sample_id}/embeddings/{embedding_id}"
        )
        _coord_fields = set(EmbeddingCoordinate.__dataclass_fields__)
        spots = [
            EmbeddingCoordinate(**{k: v for k, v in s.items() if k in _coord_fields})
            for s in data.get("spots", [])
        ]
        return EmbeddingCoords(
            embedding_id=data["embedding_id"],
            method=data["method"],
            n_components=data["n_components"],
            spots=spots,
        )

    def upload_embedding(
        self, accession: str, sample_id: int, csv_path: str
    ) -> EmbeddingUploadResult:
        """Upload embedding coordinates from a CSV file.

        The CSV must have a ``barcode`` column and at least two dimension
        columns whose names signal the method (e.g. ``UMAP_1``/``UMAP_2``,
        ``tSNE_1``/``tSNE_2``, ``PC_1``/``PC_2``).

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        sample_id:
            Sample ID to attach the embedding to.
        csv_path:
            Path to the CSV file on the local filesystem.

        Returns
        -------
        :class:`EmbeddingUploadResult`
            Result with ``embedding_id``, ``inserted`` count, and ``sample_id``.

        Examples
        --------
        >>> result = client.upload_embedding("STD0001", sample_id=1, csv_path="umap.csv")
        >>> print(f"Inserted {result.inserted} spots (embedding {result.embedding_id})")
        """
        url = f"{self.base_url}/api/v1/datasets/{accession}/samples/{sample_id}/embeddings/upload"
        with open(csv_path, "rb") as fh:
            resp = self._session.post(url, files={"file": (Path(csv_path).name, fh, "text/csv")})
        resp.raise_for_status()
        data = resp.json()
        return EmbeddingUploadResult(
            embedding_id=data["embedding_id"],
            inserted=data["inserted"],
            sample_id=data["sample_id"],
        )

    def compute_embedding(self, accession: str, sample_id: int) -> EmbeddingComputeResult:
        """Trigger server-side UMAP computation for a sample.

        The computation runs asynchronously. Poll
        :meth:`get_embedding_compute_status` to check progress.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        sample_id:
            Sample ID.

        Returns
        -------
        :class:`EmbeddingComputeResult`
            Initial result with ``embedding_id`` and ``status="pending"``.

        Raises
        ------
        requests.HTTPError
            If a computation is already running (409 Conflict).

        Examples
        --------
        >>> result = client.compute_embedding("STD0001", sample_id=1)
        >>> print("Job started, embedding_id:", result.embedding_id)
        """
        url = f"{self.base_url}/api/v1/datasets/{accession}/samples/{sample_id}/embeddings/compute"
        resp = self._session.post(url)
        resp.raise_for_status()
        data = resp.json()
        return EmbeddingComputeResult(
            embedding_id=data["embedding_id"],
            status=data["status"],
        )

    def get_embedding_compute_status(
        self, accession: str, sample_id: int
    ) -> EmbeddingComputeStatus:
        """Check the status of the most recent UMAP computation.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        sample_id:
            Sample ID.

        Returns
        -------
        :class:`EmbeddingComputeStatus`
            Status object; ``status`` is one of ``"not_started"``,
            ``"pending"``, ``"computing"``, ``"ready"``, or ``"failed"``.

        Examples
        --------
        >>> import time
        >>> client.compute_embedding("STD0001", sample_id=1)
        >>> while True:
        ...     s = client.get_embedding_compute_status("STD0001", sample_id=1)
        ...     if s.status in ("ready", "failed"):
        ...         break
        ...     time.sleep(5)
        >>> print("Done:", s.status)
        """
        data = self._get(
            f"/api/v1/datasets/{accession}/samples/{sample_id}/embeddings/compute/status"
        )
        return EmbeddingComputeStatus(
            status=data["status"],
            embedding_id=data.get("embedding_id"),
        )

    def spatially_variable_genes(
        self,
        accession: str,
        sample_id: int,
        *,
        method: str = "moran_i",
        n_top_genes: int = 50,
        min_expr_fraction: float = 0.1,
    ) -> SvgResult:
        """Identify spatially variable genes for a sample using Moran's I.

        Results are cached server-side for 24 hours; repeated calls with the
        same parameters return almost instantly.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        sample_id:
            Sample ID.
        method:
            Spatial autocorrelation method. Currently ``"moran_i"`` (Moran's I).
        n_top_genes:
            Number of top-ranked genes to return (1–500, default 50).
        min_expr_fraction:
            Minimum fraction of spots expressing a gene (0–1, default 0.1).
            Genes below this threshold are excluded before testing.

        Returns
        -------
        :class:`SvgResult`
            Result with a ``genes`` list of :class:`SvgGeneResult` objects
            sorted by Moran's I descending.

        Raises
        ------
        requests.HTTPError
            429 if the per-user hourly rate limit is exceeded. The limit is
            server-configured; the response body states the current value.

        Examples
        --------
        >>> result = client.spatially_variable_genes("STD0001", sample_id=1)
        >>> for g in result.genes[:5]:
        ...     print(f"{g.gene}: I={g.morans_i:.3f}, padj={g.padj:.2e}")
        """
        url = (
            f"{self.base_url}/api/v1/datasets/{accession}/samples/{sample_id}"
            "/spatially-variable-genes"
        )
        resp = self._session.post(
            url,
            json={
                "method": method,
                "n_top_genes": n_top_genes,
                "min_expr_fraction": min_expr_fraction,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        _gene_fields = set(SvgGeneResult.__dataclass_fields__)
        genes = [
            SvgGeneResult(**{k: v for k, v in g.items() if k in _gene_fields})
            for g in data.get("genes", [])
        ]
        return SvgResult(
            genes=genes,
            method=data["method"],
            n_genes_tested=data["n_genes_tested"],
            sample_id=data["sample_id"],
            subsampled=data["subsampled"],
            n_spots=data["n_spots"],
            cached=data["cached"],
        )

    def start_format_conversion(
        self,
        accession: str,
        file_id: int,
        target_format: str,
    ) -> ConversionJob:
        """Request on-the-fly conversion of an expression matrix file.

        If a non-expired job for the same ``(file_id, target_format)`` pair
        already exists it is returned immediately without re-running the
        conversion.  Supported target formats: ``h5ad``, ``mex``, ``csv_long``,
        ``h5seurat``.  (``loom`` is not offered — the server does not ship
        loompy — and is rejected with a 400.)

        The job starts asynchronously.  Poll :meth:`get_format_conversion_status`
        until ``status == "ready"``, then use :meth:`download_converted_file` to
        stream the result.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        file_id:
            ID of the expression-matrix file to convert.
        target_format:
            Desired output format (``h5ad``, ``mex``, ``csv_long``, or
            ``h5seurat``).

        Returns
        -------
        :class:`ConversionJob`
            Job descriptor with ``job_id`` and initial ``status``
            (``"pending"``, ``"converting"``, or ``"ready"``).

        Raises
        ------
        requests.HTTPError
            404 if the file or dataset is not found; 400 for unsupported formats.

        Examples
        --------
        >>> job = client.start_format_conversion("STD0001", file_id=1, target_format="h5ad")
        >>> print(job.job_id, job.status)
        """
        url = f"{self.base_url}/api/v1/datasets/{accession}/files/{file_id}/convert"
        resp = self._session.post(url, json={"target_format": target_format})
        resp.raise_for_status()
        data = resp.json()
        return ConversionJob(
            job_id=data["job_id"],
            status=data["status"],
            target_format=data["target_format"],
            size_bytes=data.get("size_bytes"),
            url=data.get("url"),
        )

    def get_format_conversion_status(
        self,
        accession: str,
        file_id: int,
        job_id: str,
    ) -> ConversionJob:
        """Poll the status of a format conversion job.

        When ``status`` is ``"ready"`` the returned object includes a presigned
        ``url`` valid for one hour that can be passed to
        :meth:`download_converted_file`.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        file_id:
            ID of the source expression-matrix file.
        job_id:
            Conversion job ID returned by :meth:`start_format_conversion`.

        Returns
        -------
        :class:`ConversionJob`
            Updated job descriptor.

        Raises
        ------
        requests.HTTPError
            404 if the job is not found.

        Examples
        --------
        >>> import time
        >>> while True:
        ...     job = client.get_format_conversion_status("STD0001", file_id=1, job_id=job.job_id)
        ...     if job.status in ("ready", "failed"):
        ...         break
        ...     time.sleep(5)
        """
        url = f"{self.base_url}/api/v1/datasets/{accession}/files/{file_id}/convert/{job_id}"
        resp = self._session.get(url)
        resp.raise_for_status()
        data = resp.json()
        return ConversionJob(
            job_id=data["job_id"],
            status=data["status"],
            target_format=data["target_format"],
            size_bytes=data.get("size_bytes"),
            url=data.get("url"),
        )

    def download_converted_file(
        self,
        accession: str,
        file_id: int,
        job_id: str,
        dest: str | Path,
    ) -> Path:
        """Download the output of a completed format conversion job.

        Polls until the job is ready, then streams the converted file to
        *dest*.

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        file_id:
            ID of the source expression-matrix file.
        job_id:
            Conversion job ID returned by :meth:`start_format_conversion`.
        dest:
            Local filesystem path to write the downloaded file.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the downloaded file (same as *dest*).

        Raises
        ------
        requests.HTTPError
            404 if the job is not found; raised if status is ``"failed"``.
        RuntimeError
            If the job completed but the download URL is missing.

        Examples
        --------
        >>> path = client.download_converted_file("STD0001", file_id=1, job_id=job.job_id, dest="output.h5ad")
        >>> print(f"Downloaded to {path}")
        """
        import time

        job = self.get_format_conversion_status(accession, file_id, job_id)
        while job.status not in ("ready", "failed"):
            time.sleep(5)
            job = self.get_format_conversion_status(accession, file_id, job_id)

        if job.status == "failed":
            raise requests.HTTPError(f"Conversion job {job_id} failed")
        if not job.url:
            raise RuntimeError(f"Conversion job {job_id} is ready but has no download URL")

        dest = Path(dest)
        # A plain request, not self._session: the presigned URL is for the storage
        # host, which must not receive the API key.
        with requests.get(job.url, stream=True, timeout=TRANSFER_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
        return dest

    # -----------------------------------------------------------------------
    # Gene-Centric Cross-Dataset Expression Search (Feature 26)
    # -----------------------------------------------------------------------

    def search_by_gene(
        self,
        gene: str,
        *,
        organism_id: Optional[int] = None,
        tissue_id: Optional[int] = None,
        technology_id: Optional[int] = None,
        min_pct_expressed: float = 0.1,
        sort_by: str = "mean_expr",
        page: int = 1,
        per_page: int = 20,
    ) -> GeneSearchResponse:
        """Search for datasets expressing a gene across the entire catalogue.

        Parameters
        ----------
        gene:
            Gene symbol to search for (e.g. ``"GAPDH"``). Case-insensitive.
        organism_id:
            Optional ontology-term ID to filter by organism.
        tissue_id:
            Optional ontology-term ID to filter by tissue.
        technology_id:
            Optional technology ID to filter by assay technology.
        min_pct_expressed:
            Minimum fraction of spots that must express the gene (0–1).
            Defaults to 0.1 (10 %).
        sort_by:
            Sort results by ``"mean_expr"`` (default) or
            ``"pct_spots_expressed"``.
        page:
            Page number (1-based).
        per_page:
            Results per page (max 100).

        Returns
        -------
        :class:`GeneSearchResponse`
            Paginated list of :class:`GeneSearchResult` objects plus total.

        Examples
        --------
        >>> resp = client.search_by_gene("GAPDH", min_pct_expressed=0.2)
        >>> for ds in resp.items:
        ...     print(ds.dataset_accession, ds.pct_spots_expressed)
        """
        params: dict = {
            "q": gene,
            "min_pct_expressed": min_pct_expressed,
            "sort_by": sort_by,
            "page": page,
            "per_page": per_page,
        }
        if organism_id is not None:
            params["organism_id"] = organism_id
        if tissue_id is not None:
            params["tissue_id"] = tissue_id
        if technology_id is not None:
            params["technology_id"] = technology_id

        data = self._get("/api/v1/search/genes", params=params)
        items = [
            GeneSearchResult(
                dataset_accession=r["dataset_accession"],
                dataset_title=r["dataset_title"],
                sample_id=r["sample_id"],
                mean_expr=r["mean_expr"],
                pct_spots_expressed=r["pct_spots_expressed"],
                max_expr=r["max_expr"],
                organism=r.get("organism"),
                tissue=r.get("tissue"),
                technology=r.get("technology"),
                thumbnail_url=r.get("thumbnail_url"),
            )
            for r in data["items"]
        ]
        return GeneSearchResponse(
            items=items,
            total=data["total"],
            page=data["page"],
            per_page=data["per_page"],
        )

    def gene_autocomplete(
        self,
        prefix: str,
        per_page: int = 10,
    ) -> List[GeneAutocompleteSuggestion]:
        """Return autocomplete suggestions for a gene symbol prefix.

        Parameters
        ----------
        prefix:
            Partial gene symbol (e.g. ``"GAP"`` → ``"GAPDH"``, …).
        per_page:
            Maximum number of suggestions to return (default 10).

        Returns
        -------
        list of :class:`GeneAutocompleteSuggestion`

        Examples
        --------
        >>> suggestions = client.gene_autocomplete("FOX")
        >>> [s.gene_symbol for s in suggestions]
        ['FOXP2', 'FOXP1', 'FOXA1']
        """
        data = self._get(
            "/api/v1/search/genes/autocomplete",
            params={"q": prefix, "per_page": per_page},
        )
        return [
            GeneAutocompleteSuggestion(
                gene_symbol=s["gene_symbol"],
                dataset_count=s["dataset_count"],
            )
            for s in data
        ]

    # ------------------------------------------------------------------
    # Teams (Feature 27)
    # ------------------------------------------------------------------

    def _build_team(self, data: dict) -> Team:
        return Team(
            id=data["id"],
            name=data["name"],
            slug=data["slug"],
            owner_id=data["owner_id"],
            created_at=data["created_at"],
            member_count=data.get("member_count", 0),
            dataset_count=data.get("dataset_count", 0),
            description=data.get("description"),
        )

    def _build_team_member(self, data: dict) -> TeamMember:
        return TeamMember(
            id=data["id"],
            team_id=data["team_id"],
            user_id=data["user_id"],
            role=data["role"],
            invited_at=data["invited_at"],
            full_name=data.get("full_name", ""),
            email=data.get("email", ""),
            accepted_at=data.get("accepted_at"),
            institution=data.get("institution"),
        )

    def list_teams(self, page: int = 1, per_page: int = 20) -> TeamList:
        """List all public teams.

        Parameters
        ----------
        page:
            Page number (1-based).
        per_page:
            Results per page (max 100).

        Returns
        -------
        :class:`TeamList`
        """
        data = self._get("/api/v1/teams", params={"page": page, "per_page": per_page})
        return TeamList(
            items=[self._build_team(t) for t in data["items"]],
            total=data["total"],
            page=data["page"],
            per_page=data["per_page"],
        )

    def get_team(self, slug: str) -> Team:
        """Retrieve a team by its URL slug.

        Parameters
        ----------
        slug:
            Team slug, e.g. ``"smith-lab"``.

        Returns
        -------
        :class:`Team`
        """
        data = self._get(f"/api/v1/teams/{slug}")
        return self._build_team(data)

    def get_my_teams(self) -> List[Team]:
        """List teams the authenticated user belongs to.

        Returns
        -------
        list of :class:`Team`
        """
        data = self._get("/api/v1/teams/my")
        return [self._build_team(t) for t in data]

    def create_team(self, name: str, description: Optional[str] = None) -> Team:
        """Create a new team.

        Parameters
        ----------
        name:
            Human-readable team name.
        description:
            Optional description of the lab or group.

        Returns
        -------
        :class:`Team`
        """
        payload: dict = {"name": name}
        if description is not None:
            payload["description"] = description
        resp = self._session.post(
            f"{self.base_url}/api/v1/teams",
            json=payload,
        )
        resp.raise_for_status()
        return self._build_team(resp.json())

    def update_team(
        self,
        slug: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Team:
        """Update team name or description.

        Parameters
        ----------
        slug:
            Team slug.
        name:
            New display name.
        description:
            New description.

        Returns
        -------
        :class:`Team`
        """
        payload: dict = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        resp = self._session.patch(
            f"{self.base_url}/api/v1/teams/{slug}",
            json=payload,
        )
        resp.raise_for_status()
        return self._build_team(resp.json())

    def delete_team(self, slug: str) -> None:
        """Permanently delete a team (owner only).

        Parameters
        ----------
        slug:
            Team slug.
        """
        resp = self._session.delete(f"{self.base_url}/api/v1/teams/{slug}")
        resp.raise_for_status()

    def list_team_members(self, slug: str) -> List[TeamMember]:
        """List accepted and pending members of a team.

        Parameters
        ----------
        slug:
            Team slug.

        Returns
        -------
        list of :class:`TeamMember`
        """
        data = self._get(f"/api/v1/teams/{slug}/members")
        return [self._build_team_member(m) for m in data]

    def invite_team_member(self, slug: str, email: str, role: str = "member") -> TeamMember:
        """Invite a user to join a team.

        Parameters
        ----------
        slug:
            Team slug.
        email:
            Email address of the user to invite.
        role:
            Role to assign: ``"member"`` (default) or ``"admin"``.

        Returns
        -------
        :class:`TeamMember`
        """
        resp = self._session.post(
            f"{self.base_url}/api/v1/teams/{slug}/invitations",
            json={"email": email, "role": role},
        )
        resp.raise_for_status()
        return self._build_team_member(resp.json())

    def accept_team_invitation(self, slug: str) -> TeamMember:
        """Accept a pending team invitation for the authenticated user.

        Parameters
        ----------
        slug:
            Team slug.

        Returns
        -------
        :class:`TeamMember`
        """
        resp = self._session.post(
            f"{self.base_url}/api/v1/teams/{slug}/invitations/accept",
        )
        resp.raise_for_status()
        return self._build_team_member(resp.json())

    def remove_team_member(self, slug: str, user_id: int) -> None:
        """Remove a member from a team.

        Parameters
        ----------
        slug:
            Team slug.
        user_id:
            ID of the user to remove.
        """
        resp = self._session.delete(
            f"{self.base_url}/api/v1/teams/{slug}/members/{user_id}",
        )
        resp.raise_for_status()

    def change_team_member_role(self, slug: str, user_id: int, role: str) -> TeamMember:
        """Change a team member's role.

        Parameters
        ----------
        slug:
            Team slug.
        user_id:
            ID of the user whose role to change.
        role:
            New role: ``"member"``, ``"admin"``, or ``"owner"``.

        Returns
        -------
        :class:`TeamMember`
        """
        resp = self._session.patch(
            f"{self.base_url}/api/v1/teams/{slug}/members/{user_id}",
            json={"role": role},
        )
        resp.raise_for_status()
        return self._build_team_member(resp.json())

    def assign_dataset_team(self, accession: str, team_id: Optional[int]) -> Dataset:
        """Assign a dataset to a team (or remove team assignment with ``None``).

        Parameters
        ----------
        accession:
            Dataset accession, e.g. ``"STD0001"``.
        team_id:
            ID of the team to assign, or ``None`` to remove the team.

        Returns
        -------
        :class:`Dataset`
        """
        resp = self._session.patch(
            f"{self.base_url}/api/v1/datasets/{accession}/team",
            json={"team_id": team_id},
        )
        resp.raise_for_status()
        return self._build_dataset(resp.json())

    # ------------------------------------------------------------------
    # Programmatic submission (Feature 28)
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        title: str,
        abstract: str,
        organism: str,
        tissue: str,
        technology: str,
        disease: Optional[str] = None,
        embargo_days: int = 365,
    ) -> Dataset:
        """Create a new draft dataset programmatically.

        Parameters
        ----------
        title:
            Human-readable dataset title.
        abstract:
            Dataset abstract / description.
        organism:
            Organism name or ID.
        tissue:
            Tissue name or ID.
        technology:
            Spatial technology name or ID (e.g. ``"10x Visium"``).
        disease:
            Disease name or ID (optional).
        embargo_days:
            Days until automatic public release (default 365).

        Returns
        -------
        :class:`Dataset`
        """
        from stomics import submission as _sub

        return _sub.create_dataset(
            self,
            title=title,
            abstract=abstract,
            organism=organism,
            tissue=tissue,
            technology=technology,
            disease=disease,
            embargo_days=embargo_days,
        )

    def add_sample(
        self,
        accession: str,
        title: str,
        description: Optional[str] = None,
        section_id: Optional[str] = None,
    ) -> Sample:
        """Add a single sample to a dataset.

        Parameters
        ----------
        accession:
            Dataset accession number.
        title:
            Sample name / title.
        description:
            Optional free-text description.
        section_id:
            Optional section identifier.

        Returns
        -------
        :class:`Sample`
        """
        from stomics import submission as _sub

        return _sub.add_sample(
            self, accession=accession, title=title, description=description, section_id=section_id
        )

    def add_samples(self, accession: str, samples: list) -> "BulkSampleResponse":
        """Bulk-add up to 500 samples in a single request.

        Parameters
        ----------
        accession:
            Dataset accession number.
        samples:
            List of dicts, each with at least a ``"title"`` key.

        Returns
        -------
        :class:`BulkSampleResponse`
        """
        from stomics import submission as _sub

        return _sub.add_samples(self, accession=accession, samples=samples)

    def upload_file(
        self,
        accession: str,
        local_path: str,
        file_type: str,
        show_progress: bool = True,
        sample_id: Optional[int] = None,
    ) -> File:
        """Upload a local file directly to S3 via presigned URL.

        Parameters
        ----------
        accession:
            Dataset accession number.
        local_path:
            Path to the file on disk.
        file_type:
            One of ``expression_matrix``, ``spatial_coordinates``,
            ``tissue_image``, ``raw_reads``, ``supplementary``.
        show_progress:
            Show a tqdm progress bar while uploading (requires ``tqdm``).
        sample_id:
            Associate the file with a specific sample (optional).

        Returns
        -------
        :class:`File`
        """
        from stomics import submission as _sub

        return _sub.upload_file(
            self,
            accession=accession,
            local_path=local_path,
            file_type=file_type,
            show_progress=show_progress,
            sample_id=sample_id,
        )

    def submit(self, accession: str) -> Dataset:
        """Submit a dataset for curator review.

        Parameters
        ----------
        accession:
            Dataset accession number.

        Returns
        -------
        :class:`Dataset`
            Updated dataset with status ``"under_review"``.
        """
        from stomics import submission as _sub

        return _sub.submit(self, accession=accession)
