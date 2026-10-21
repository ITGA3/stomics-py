"""Programmatic dataset submission helpers (Feature 28).

These methods are mixed into :class:`stomics.client.StomicsClient` and enable
end-to-end dataset submission without a browser:

.. code-block:: python

    client = StomicsClient(api_key="sk-...")
    ds = client.create_dataset(
        title="My Atlas",
        abstract="...",
        organism="Mus musculus",
        tissue="brain cortex",
        technology="10x Visium",
    )
    client.add_sample(ds.accession, title="Section 1")
    client.upload_file(ds.accession, "matrix.h5ad", file_type="expression_matrix")
    client.submit(ds.accession)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from stomics.client import StomicsClient
    from stomics.models import BulkSampleResponse, Dataset, File, Sample


def create_dataset(
    self: "StomicsClient",
    title: str,
    abstract: str,
    organism: "int | str",
    tissue: "int | str",
    technology: "int | str",
    disease: "Optional[int | str]" = None,
    embargo_days: int = 365,
) -> "Dataset":
    """Create a new draft dataset.

    Parameters
    ----------
    title:
        Human-readable title for the dataset.
    abstract:
        Description / abstract (free text).
    organism:
        Organism name or ID (e.g. ``"Mus musculus"`` or ``1``).
    tissue:
        Tissue name or ID.
    technology:
        Sequencing / spatial technology name or ID (e.g. ``"10x Visium"``).
    disease:
        Disease name or ID (optional).
    embargo_days:
        Requested embargo in days. Sent for the curator's information; the
        embargo itself is set when a curator publishes the dataset.

    Names are resolved against :meth:`StomicsClient.list_vocabulary`, and an
    unknown name raises :class:`ValueError` before anything is created. The
    API takes only IDs: it used to receive the names under keys it does not
    read, so every dataset created this way lost its classification silently.

    Returns
    -------
    Dataset
        The newly created dataset with its accession number.
    """
    from stomics.client import StomicsClient

    def one(kind: str, value) -> int:
        return self._resolve_ids(kind, value)[0]

    payload: dict = {
        "title": title,
        "abstract": abstract,
        "organism_id": one("organism", organism),
        "tissue_id": one("tissue", tissue),
        "technology_id": one("technology", technology),
        "embargo_days": embargo_days,
    }
    if disease is not None:
        payload["disease_id"] = one("disease", disease)

    resp = self._session.post(f"{self.base_url}/api/v1/datasets", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return StomicsClient._build_dataset(data)


def add_sample(
    self: "StomicsClient",
    accession: str,
    title: str,
    description: Optional[str] = None,
    section_id: Optional[str] = None,
) -> "Sample":
    """Add a single sample to a dataset.

    Parameters
    ----------
    accession:
        Dataset accession (e.g. ``"STD0001"``).
    title:
        Sample title / name.
    description:
        Optional free-text description.
    section_id:
        Optional section identifier (e.g. tissue block ID).

    Returns
    -------
    Sample
        The created sample record.
    """
    from stomics.client import StomicsClient

    payload = {"title": title}
    if description is not None:
        payload["description"] = description
    if section_id is not None:
        payload["section_id"] = section_id

    resp = self._session.post(
        f"{self.base_url}/api/v1/datasets/{accession}/samples",
        json=[payload],
    )
    resp.raise_for_status()
    data = resp.json()
    samples = data.get("samples", [])
    if not samples:
        raise RuntimeError("Server returned no samples after bulk insert")
    return StomicsClient._build_sample(samples[0])


def add_samples(
    self: "StomicsClient",
    accession: str,
    samples: List[dict],
) -> "BulkSampleResponse":
    """Bulk-add up to 500 samples to a dataset in a single request.

    Parameters
    ----------
    accession:
        Dataset accession.
    samples:
        List of sample dicts, each with at least a ``"title"`` key and
        optionally ``"description"`` and ``"section_id"``.

    Returns
    -------
    BulkSampleResponse
        Contains ``inserted`` count and the created ``Sample`` objects.
    """
    from stomics.client import StomicsClient
    from stomics.models import BulkSampleResponse

    resp = self._session.post(
        f"{self.base_url}/api/v1/datasets/{accession}/samples",
        json=samples,
    )
    resp.raise_for_status()
    data = resp.json()
    built_samples = [StomicsClient._build_sample(s) for s in data.get("samples", [])]
    return BulkSampleResponse(inserted=data.get("inserted", 0), samples=built_samples)


def upload_file(
    self: "StomicsClient",
    accession: str,
    local_path: str,
    file_type: str,
    show_progress: bool = True,
    sample_id: Optional[int] = None,
) -> "File":
    """Upload a local file to a dataset via presigned S3 PUT URL.

    This method:
    1. Requests a presigned PUT URL from the API.
    2. Streams the file bytes directly to S3 (no server relay).
    3. Calls the ``complete`` endpoint to register the upload and trigger QC.

    Parameters
    ----------
    accession:
        Dataset accession.
    local_path:
        Path to the local file to upload.
    file_type:
        One of ``expression_matrix``, ``spatial_coordinates``,
        ``tissue_image``, ``raw_reads``, or ``supplementary``.
    show_progress:
        When True and ``tqdm`` is installed, display a progress bar.
    sample_id:
        Associate the file with a specific sample (optional).

    Returns
    -------
    File
        The registered file record after upload completion.
    """
    import requests

    from stomics.client import TRANSFER_TIMEOUT, StomicsClient

    path = Path(local_path)
    size_bytes = path.stat().st_size
    filename = path.name

    url_resp = self._session.post(
        f"{self.base_url}/api/v1/datasets/{accession}/files/upload-url",
        json={
            "filename": filename,
            "file_type": file_type,
            "size_bytes": size_bytes,
        },
    )
    url_resp.raise_for_status()
    url_data = url_resp.json()
    upload_url: str = url_data["upload_url"]
    file_id: int = url_data["file_id"]

    with open(path, "rb") as fh:
        if show_progress:
            try:
                from tqdm import tqdm

                with tqdm(total=size_bytes, unit="B", unit_scale=True, desc=filename) as pbar:

                    class _ProgressReader:
                        def __init__(self, f, bar):
                            self._f = f
                            self._bar = bar

                        def read(self, n=-1):
                            chunk = self._f.read(n)
                            self._bar.update(len(chunk))
                            return chunk

                    data: object = _ProgressReader(fh, pbar)
                    put_resp = requests.put(upload_url, data=data, timeout=TRANSFER_TIMEOUT)
            except ImportError:
                put_resp = requests.put(upload_url, data=fh, timeout=TRANSFER_TIMEOUT)
        else:
            put_resp = requests.put(upload_url, data=fh, timeout=TRANSFER_TIMEOUT)
    put_resp.raise_for_status()

    complete_resp = self._session.post(
        f"{self.base_url}/api/v1/datasets/{accession}/files/{file_id}/complete",
        json={},
    )
    complete_resp.raise_for_status()
    return StomicsClient._build_file(complete_resp.json())


def submit(self: "StomicsClient", accession: str) -> "Dataset":
    """Submit a dataset for curator review.

    Parameters
    ----------
    accession:
        Dataset accession to submit.

    Returns
    -------
    Dataset
        The updated dataset (status will be ``"under_review"``).
    """
    from stomics.client import StomicsClient

    resp = self._session.post(
        f"{self.base_url}/api/v1/datasets/{accession}/submit",
    )
    resp.raise_for_status()
    return StomicsClient._build_dataset(resp.json())
