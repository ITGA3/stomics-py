# stomics-py

Python client for [Stomics.io](https://stomics.io), a public repository of spatial transcriptomics
datasets. Search the catalogue, download files, load datasets into AnnData and submit your own data
from notebooks or pipelines.

## Install

Python 3.10 or later and `git` are required.

```bash
pip install "git+https://github.com/ITGA3/stomics-py"
# with AnnData support
pip install "stomics-py[anndata] @ git+https://github.com/ITGA3/stomics-py"
# a specific release
pip install "git+https://github.com/ITGA3/stomics-py@v0.1.1"
```

## Quick start

```python
from stomics import StomicsClient

client = StomicsClient()  # reads STOMICS_API_KEY; public data needs no key
for ds in client.search(organism="Homo sapiens", per_page=10):
    print(ds.accession, ds.title)

adata = client.load_anndata("STD0100")

# What the processed expression holds, then any per-spot field and raw counts
manifest = client.expression_manifest("STD0100")
clusters = client.get_obs("STD0100", "leiden")
counts = client.get_expression("STD0100", gene="GAPDH", layer="counts")

# Clusters, markers and spatially variable genes from the stomics.io standard pipeline
run = client.get_standard_analysis("STD0100", sample_id="12")
clusters = client.get_obs("STD0100", "standard:leiden_1.0", sample_id="12")
markers = client.get_markers("STD0100", sample_id="12", resolution=1.0)
mean, fraction = client.get_dotplot("STD0100", sample_id="12")
svgs = client.get_spatially_variable_genes("STD0100", sample_id="12", limit=20)

# Files: filter by name, type or sample, or download them all, resuming if interrupted
matrices = client.list_files("STD0716", q="GSM5", file_types=["expression_matrix"])
client.download_dataset("STD0716", "./STD0716")
```

Create an API key at <https://stomics.io/account/api-keys> for anything tied to your account, such as
submitting data. The client talks to `https://api.stomics.io` unless you pass `base_url`.

The package also installs `stomics submit --config submission.yaml`, which runs a whole submission
from a YAML file.

## Documentation

- SDK reference: <https://stomics.io/docs/sdk>
- Programmatic submission: <https://stomics.io/docs/submit>
- REST API: <https://api.stomics.io/api/v1/docs>
- Help Center: <https://stomics.io/help>

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Support

Report bugs in this repository's issues or write to support@stomics.io. Pull requests are welcome;
CI must pass, including the 100% test coverage gate.

## License

MIT. Copyright (c) 2026 The Allman Institute for Personalized Medicine. See [LICENSE](LICENSE).
