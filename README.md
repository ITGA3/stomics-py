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
