"""Command-line interface for the Stomics Python SDK.

Entry point: ``stomics`` (configured via ``[project.scripts]`` in pyproject.toml).

Supported sub-commands
----------------------
stomics submit --config submission.yaml
    Read metadata and file paths from a YAML file and submit a dataset.
    Useful for Snakemake / Nextflow pipeline integration.

YAML schema
-----------
.. code-block:: yaml

    api_key: sk-...            # optional — falls back to STOMICS_API_KEY env var
    base_url: https://api.stomics.io   # optional

    dataset:
      title: "My Spatial Atlas"
      abstract: "..."
      organism: "Mus musculus"
      tissue: "brain cortex"
      technology: "10x Visium"
      disease: null              # optional
      embargo_days: 365          # optional, default 365

    samples:                     # optional list
      - title: "Section 1"
        section_id: "A1"
        description: "Frontal cortex"

    files:
      - path: ./matrix.h5ad
        file_type: expression_matrix
      - path: ./tissue_positions.csv
        file_type: spatial_coordinates
"""

import argparse
import os
import sys
from typing import Any, Dict
from urllib.parse import urlsplit

from stomics.client import StomicsClient


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover
        print("PyYAML is required for 'stomics submit'. Install with: pip install pyyaml")
        sys.exit(1)
    with open(path) as fh:
        return yaml.safe_load(fh)


def _site_url(base_url: str) -> str:
    """The web site for an API base URL: https://api.stomics.io -> https://stomics.io."""
    parts = urlsplit(base_url.rstrip("/"))
    return f"{parts.scheme}://{parts.netloc.removeprefix('api.')}"


def cmd_submit(args: argparse.Namespace) -> int:
    """Execute the ``stomics submit`` command."""
    config = _load_yaml(args.config)

    api_key = config.get("api_key") or os.environ.get("STOMICS_API_KEY")
    base_url = config.get("base_url", "https://api.stomics.io")

    client = StomicsClient(api_key=api_key, base_url=base_url)

    ds_cfg: Dict[str, Any] = config.get("dataset", {})
    if not ds_cfg:
        print("Error: 'dataset' section is required in the YAML config.", file=sys.stderr)
        return 1

    required = ("title", "abstract", "organism", "tissue", "technology")
    missing = [k for k in required if not ds_cfg.get(k)]
    if missing:
        print(f"Error: missing required dataset fields: {', '.join(missing)}", file=sys.stderr)
        return 1

    print(f"Creating dataset: {ds_cfg['title']!r} ...")
    dataset = client.create_dataset(
        title=ds_cfg["title"],
        abstract=ds_cfg["abstract"],
        organism=ds_cfg["organism"],
        tissue=ds_cfg["tissue"],
        technology=ds_cfg["technology"],
        disease=ds_cfg.get("disease"),
        embargo_days=int(ds_cfg.get("embargo_days", 365)),
    )
    print(f"  Created {dataset.accession}")

    samples_cfg = config.get("samples") or []
    if samples_cfg:
        print(f"Adding {len(samples_cfg)} sample(s) ...")
        result = client.add_samples(dataset.accession, samples_cfg)
        print(f"  Inserted {result.inserted} sample(s)")

    files_cfg = config.get("files") or []
    for file_cfg in files_cfg:
        fpath = file_cfg.get("path")
        ftype = file_cfg.get("file_type")
        if not fpath or not ftype:
            print(
                "Warning: skipping file entry with missing 'path' or 'file_type'", file=sys.stderr
            )
            continue
        print(f"Uploading {fpath!r} as {ftype!r} ...")
        uploaded = client.upload_file(
            accession=dataset.accession,
            local_path=fpath,
            file_type=ftype,
            show_progress=True,
        )
        print(f"  Registered file {uploaded.id}: {uploaded.filename}")

    print(f"Submitting {dataset.accession} for review ...")
    updated = client.submit(dataset.accession)
    print(f"  Status: {updated.status}")
    print(
        f"\nDone. Dataset URL: {_site_url(base_url)}/datasets/{dataset.accession}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="stomics",
        description="Stomics.io command-line client",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    submit_p = sub.add_parser(
        "submit",
        help="Submit a dataset from a YAML config file",
    )
    submit_p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the submission YAML config file",
    )

    args = parser.parse_args()
    if args.command == "submit":
        sys.exit(cmd_submit(args))


if __name__ == "__main__":  # pragma: no cover
    main()
