"""Tests for stomics.submission and stomics.cli (Feature 28)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from stomics import StomicsClient
from stomics.models import BulkSampleResponse

BASE = "https://api.stomics.io"


@pytest.fixture
def client():
    return StomicsClient(api_key="sk-test", base_url=BASE)


_DS_JSON = {
    "id": 1,
    "accession": "STD0001",
    "title": "T",
    "abstract": "A",
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
_SAMPLE_JSON = {
    "id": 1,
    "dataset_id": 1,
    "name": "S1",
    "title": "S1",
    "description": None,
    "section_id": None,
}
_FILE_JSON = {
    "id": 5,
    "dataset_id": 1,
    "sample_id": None,
    "filename": "matrix.h5ad",
    "file_type": "expression_matrix",
    "size_bytes": 10,
    "md5_checksum": None,
    "storage_path": "STD0001/matrix.h5ad",
    "upload_status": "complete",
    "created_at": "2026-01-01T00:00:00Z",
    "download_url": None,
    "file_metadata": None,
}


# ---------------------------------------------------------------------------
# BulkSampleResponse model
# ---------------------------------------------------------------------------


def test_bulk_sample_response_defaults():
    bsr = BulkSampleResponse(inserted=0)
    assert bsr.inserted == 0
    assert bsr.samples == []


def test_bulk_sample_response_with_samples():
    bsr = BulkSampleResponse(inserted=2, samples=["a", "b"])
    assert bsr.inserted == 2
    assert len(bsr.samples) == 2


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_main_entrypoint_help():
    """Running 'stomics' with no args exits with non-zero (argparse error)."""
    from stomics.cli import main

    with pytest.raises(SystemExit) as exc_info:
        sys.argv = ["stomics"]
        main()
    assert exc_info.value.code != 0


def test_cli_submit_missing_config(tmp_path):
    from stomics.cli import cmd_submit

    class FakeArgs:
        config = str(tmp_path / "nonexistent.yaml")

    with pytest.raises((FileNotFoundError, OSError)):
        cmd_submit(FakeArgs())


def test_cli_submit_missing_dataset_section(tmp_path):
    import yaml

    from stomics.cli import cmd_submit

    cfg = tmp_path / "sub.yaml"
    cfg.write_text(yaml.dump({}))

    class FakeArgs:
        config = str(cfg)

    result = cmd_submit(FakeArgs())
    assert result == 1


def test_cli_submit_missing_required_fields(tmp_path):
    import yaml

    from stomics.cli import cmd_submit

    cfg = tmp_path / "sub.yaml"
    cfg.write_text(yaml.dump({"dataset": {"title": "T"}}))

    class FakeArgs:
        config = str(cfg)

    result = cmd_submit(FakeArgs())
    assert result == 1


def test_cli_submit_skips_incomplete_file_entries(tmp_path, capsys):
    from unittest.mock import MagicMock, patch

    import yaml

    from stomics.cli import cmd_submit

    data_file = tmp_path / "matrix.h5ad"
    data_file.write_bytes(b"data")

    cfg_data = {
        "api_key": "sk-test",
        "dataset": {
            "title": "T",
            "abstract": "A",
            "organism": "O",
            "tissue": "t",
            "technology": "tech",
        },
        "files": [
            {"file_type": "expression_matrix"},  # missing path → skip
            {"path": str(data_file)},  # missing file_type → skip
        ],
    }
    cfg = tmp_path / "sub.yaml"
    cfg.write_text(yaml.dump(cfg_data))

    class FakeArgs:
        config = str(cfg)

    with patch("stomics.cli.StomicsClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.create_dataset.return_value = MagicMock(accession="STD0001")
        mock_instance.submit.return_value = MagicMock(status="under_review")
        MockClient.return_value = mock_instance

        result = cmd_submit(FakeArgs())

    assert result == 0
    # upload_file should not have been called (both entries were skipped)
    mock_instance.upload_file.assert_not_called()
    captured = capsys.readouterr()
    assert "Warning" in captured.err


def test_cli_submit_full_workflow(tmp_path, capsys):
    import yaml

    from stomics.cli import cmd_submit

    data_file = tmp_path / "matrix.h5ad"
    data_file.write_bytes(b"data")

    cfg_data = {
        "api_key": "sk-test",
        "base_url": BASE,
        "dataset": {
            "title": "Atlas",
            "abstract": "Abs",
            "organism": "Mus musculus",
            "tissue": "brain",
            "technology": "10x Visium",
            "embargo_days": 365,
        },
        "samples": [{"title": "S1", "section_id": "A1"}],
        "files": [
            {"path": str(data_file), "file_type": "expression_matrix"},
        ],
    }
    cfg = tmp_path / "sub.yaml"
    cfg.write_text(yaml.dump(cfg_data))

    class FakeArgs:
        config = str(cfg)

    with patch("stomics.cli.StomicsClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.create_dataset.return_value = MagicMock(accession="STD0001")
        mock_instance.add_samples.return_value = MagicMock(inserted=1)
        mock_instance.upload_file.return_value = MagicMock(id=5, filename="matrix.h5ad")
        mock_instance.submit.return_value = MagicMock(status="under_review")
        MockClient.return_value = mock_instance

        result = cmd_submit(FakeArgs())

    assert result == 0
    mock_instance.create_dataset.assert_called_once()
    mock_instance.add_samples.assert_called_once_with(
        "STD0001", [{"title": "S1", "section_id": "A1"}]
    )
    mock_instance.upload_file.assert_called_once()
    mock_instance.submit.assert_called_once_with("STD0001")

    captured = capsys.readouterr()
    assert "STD0001" in captured.out
    assert "under_review" in captured.out
    assert "Dataset URL: https://stomics.io/datasets/STD0001" in captured.out


def test_cli_submit_no_samples_no_files(tmp_path, capsys):
    import yaml

    from stomics.cli import cmd_submit

    cfg_data = {
        "api_key": "sk-test",
        "dataset": {
            "title": "T",
            "abstract": "A",
            "organism": "O",
            "tissue": "t",
            "technology": "tech",
        },
    }
    cfg = tmp_path / "sub.yaml"
    cfg.write_text(yaml.dump(cfg_data))

    class FakeArgs:
        config = str(cfg)

    with patch("stomics.cli.StomicsClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.create_dataset.return_value = MagicMock(accession="STD0001")
        mock_instance.submit.return_value = MagicMock(status="under_review")
        MockClient.return_value = mock_instance

        result = cmd_submit(FakeArgs())

    assert result == 0
    mock_instance.add_samples.assert_not_called()
    mock_instance.upload_file.assert_not_called()


def test_cli_submit_uses_env_api_key(tmp_path, monkeypatch, capsys):
    import yaml

    from stomics.cli import cmd_submit

    monkeypatch.setenv("STOMICS_API_KEY", "sk-from-env")

    cfg_data = {
        "dataset": {
            "title": "T",
            "abstract": "A",
            "organism": "O",
            "tissue": "t",
            "technology": "tech",
        },
    }
    cfg = tmp_path / "sub.yaml"
    cfg.write_text(yaml.dump(cfg_data))

    class FakeArgs:
        config = str(cfg)

    with patch("stomics.cli.StomicsClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.create_dataset.return_value = MagicMock(accession="STD0001")
        mock_instance.submit.return_value = MagicMock(status="under_review")
        MockClient.return_value = mock_instance

        result = cmd_submit(FakeArgs())

    assert result == 0
    MockClient.assert_called_once_with(api_key="sk-from-env", base_url="https://api.stomics.io")


def test_cli_no_yaml_import(tmp_path):
    """If yaml is not importable, exit(1) is called."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    from stomics.cli import _load_yaml

    cfg = tmp_path / "sub.yaml"
    cfg.write_text("dataset: {}")

    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(SystemExit):
            _load_yaml(str(cfg))


def test_cli_main_submit_command(tmp_path, capsys):
    import yaml

    from stomics.cli import main

    cfg_data = {
        "api_key": "sk-test",
        "dataset": {
            "title": "T",
            "abstract": "A",
            "organism": "O",
            "tissue": "t",
            "technology": "tech",
        },
    }
    cfg = tmp_path / "sub.yaml"
    cfg.write_text(yaml.dump(cfg_data))

    with patch("stomics.cli.StomicsClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.create_dataset.return_value = MagicMock(accession="STD0001")
        mock_instance.submit.return_value = MagicMock(status="under_review")
        MockClient.return_value = mock_instance

        sys.argv = ["stomics", "submit", "--config", str(cfg)]
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 0
