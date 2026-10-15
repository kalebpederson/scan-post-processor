import pytest

from scanpp.cli import is_simple_scan_protocol, options_from, parse_args


def test_protocol_detection():
    assert is_simple_scan_protocol(["application/pdf", "true", "/x/a.pdf"])
    assert is_simple_scan_protocol(["image/jpeg", "false", "/x/a.jpg", "--paper keep"])
    assert not is_simple_scan_protocol(["scan.pdf"])
    assert not is_simple_scan_protocol(["--paper", "keep", "a.pdf"])
    assert not is_simple_scan_protocol(["application/pdf", "maybe", "/x"])


def test_protocol_single_string_args():
    ns, keep = parse_args(["application/pdf", "false", "/x/a b.pdf",
                           "--paper keep --ocr --dpi 150 --jobs 3 --keep-blank --report '/tmp/r x.json'"])
    assert keep == "false"
    assert ns.inputs == ["/x/a b.pdf"]
    o = options_from(ns)
    assert (o.paper, o.ocr, o.dpi, o.jobs, o.drop_blank) == ("keep", True, 150, 3, False)
    assert ns.report == "/tmp/r x.json"
    assert ns.mime == "application/pdf"


def test_protocol_no_args_defaults():
    ns, keep = parse_args(["image/png", "true", "/x/a.png"])
    o = options_from(ns)
    assert keep == "true" and o.paper == "flatten" and o.dpi is None and o.drop_blank and o.deskew
    assert o.edge_clean and o.backup and o.encoder == "auto" and o.jobs == 0


def test_normal_mode_multiple_inputs_and_flags():
    ns, keep = parse_args(["--no-deskew", "--no-edge-clean", "--no-backup", "--backup-dir", "/b", "a.pdf", "b.pdf"])
    o = options_from(ns)
    assert keep is None and ns.inputs == ["a.pdf", "b.pdf"]
    assert not o.deskew and not o.edge_clean and not o.backup and o.backup_dir == "/b"


def test_normal_mode_output():
    ns, _ = parse_args(["a.pdf", "-o", "out.pdf"])
    assert ns.output == "out.pdf"


def test_output_requires_single_input():
    with pytest.raises(SystemExit):
        parse_args(["a.pdf", "b.pdf", "-o", "out.pdf"])


def test_no_input_is_error():
    with pytest.raises(SystemExit):
        parse_args(["--paper", "keep"])


def test_bad_option_in_protocol_args():
    with pytest.raises(SystemExit):
        parse_args(["application/pdf", "true", "/x/a.pdf", "--paper purple"])


def test_blank_threshold_default_pinned_and_help():
    from scanpp.options import DEFAULT_BLANK_THRESHOLD, Options
    from scanpp import geometry
    from scanpp.cli import build_parser
    assert DEFAULT_BLANK_THRESHOLD == 15.0              # calibrated, see README "Blank pages"; change deliberately
    assert Options().blank_threshold == 15.0 == geometry.BLANK_THRESHOLD
    ns, _ = parse_args(["a.pdf"])
    assert options_from(ns).blank_threshold == 15.0
    assert "--blank-threshold" in build_parser().format_help()


def test_blank_threshold_cli_and_simple_scan_args():
    ns, _ = parse_args(["--blank-threshold", "25", "a.pdf"])
    assert options_from(ns).blank_threshold == 25.0
    ns, keep = parse_args(["application/pdf", "true", "/x/a.pdf", "--paper keep --blank-threshold 7.5 --keep-blank"])
    o = options_from(ns)
    assert o.blank_threshold == 7.5 and o.paper == "keep" and not o.drop_blank
    ns, _ = parse_args(["application/pdf", "true", "/x/a.pdf", "--blank-threshold=0"])
    assert options_from(ns).blank_threshold == 0.0


@pytest.mark.parametrize("bad", ["-1", "abc", "nan", "inf"])
def test_blank_threshold_rejects_garbage(bad):
    with pytest.raises(SystemExit):
        parse_args([f"--blank-threshold={bad}", "a.pdf"])
