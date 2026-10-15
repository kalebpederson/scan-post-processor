import hashlib
import os

import pytest

from helpers import make_digital_pdf, make_scan_pdf, text_page
from scanpp import cli, pipeline, safety


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_backup_default_location_and_naming(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"one")
    b1 = safety.backup_original(f)
    assert b1 == tmp_path / ".originals" / "doc.pdf" and b1.read_bytes() == b"one"
    f.write_bytes(b"two")
    b2 = safety.backup_original(f)
    assert b2 != b1 and b2.parent == b1.parent and b2.name.startswith("doc.") and b2.suffix == ".pdf"
    assert b1.read_bytes() == b"one" and b2.read_bytes() == b"two"
    b3 = safety.backup_original(f)               # same second: still unique
    assert len({b1, b2, b3}) == 3


def test_backup_custom_dir(tmp_path):
    f = tmp_path / "x.png"
    f.write_bytes(b"z")
    b = safety.backup_original(f, tmp_path / "elsewhere")
    assert b == tmp_path / "elsewhere" / "x.png"


def test_atomic_replace_keeps_mode(tmp_path):
    f = tmp_path / "a"
    f.write_text("old")
    os.chmod(f, 0o640)
    t = safety.make_temp_path(f)
    t.write_text("new")
    safety.atomic_replace(t, f)
    assert f.read_text() == "new" and (f.stat().st_mode & 0o777) == 0o640 and not t.exists()


def test_exception_mid_pipeline_leaves_original(tmp_path, monkeypatch):
    f = make_scan_pdf(tmp_path / "s.pdf", [text_page(), text_page()])
    before = sha(f)

    def boom(*a, **k):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(pipeline.pdfio, "write_pdf", boom)
    rc = cli.main(["application/pdf", "false", str(f), "--jobs 1 --foreground"])
    assert rc == 1
    assert sha(f) == before
    assert sha(tmp_path / ".originals" / "s.pdf") == before
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".scanpp-")] == []   # temp cleaned up


def test_exception_in_worker_leaves_original(tmp_path, monkeypatch):
    f = make_scan_pdf(tmp_path / "s.pdf", [text_page()])
    before = sha(f)
    monkeypatch.setattr(pipeline, "clean_page", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    assert cli.main([str(f), "--jobs", "1"]) == 1
    assert sha(f) == before and (tmp_path / ".originals" / "s.pdf").exists()


def test_invalid_output_is_not_installed(tmp_path, monkeypatch):
    f = make_scan_pdf(tmp_path / "s.pdf", [text_page()])
    before = sha(f)
    monkeypatch.setattr(pipeline.pdfio, "validate_pdf", lambda *a: (_ for _ in ()).throw(ValueError("bad")))
    assert cli.main([str(f), "--jobs", "1"]) == 1
    assert sha(f) == before


def test_digital_pdf_untouched_no_backup(tmp_path):
    f = make_digital_pdf(tmp_path / "d.pdf")
    before = sha(f)
    assert cli.main(["application/pdf", "true", str(f)]) == 0
    assert sha(f) == before and not (tmp_path / ".originals").exists()


def test_missing_file_exit_1(tmp_path):
    assert cli.main(["application/pdf", "true", str(tmp_path / "nope.pdf")]) == 1


def test_log_file_written(tmp_path):
    f = make_digital_pdf(tmp_path / "d.pdf")
    cli.main(["application/pdf", "true", str(f)])
    assert "leaving untouched" in (tmp_path / "_log" / "scanpp.log").read_text()


def test_no_backup_and_backup_dir(tmp_path):
    f = make_scan_pdf(tmp_path / "s.pdf", [text_page()])
    assert cli.main([str(f), "--jobs", "1", "--no-backup"]) == 0
    assert not (tmp_path / ".originals").exists()
    g = make_scan_pdf(tmp_path / "t.pdf", [text_page()])
    assert cli.main([str(g), "--jobs", "1", "--backup-dir", str(tmp_path / "bk")]) == 0
    assert (tmp_path / "bk" / "t.pdf").exists()
