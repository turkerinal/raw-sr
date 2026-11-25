"""The acceptance verdict must decide, and must not decide more than it knows.

``scripts/acceptance_verdict.py`` turns acceptance metrics into an explicit
PASS / FAIL / N-A per criterion, each citing the SPEC line it comes from. The
tests below pin the two properties that make such a table trustworthy:

* every criterion is *decided* by the measurement it names -- no criterion can
  silently pass on a missing or unrelated value;
* a criterion whose data does not exist is N/A with the substitute written out,
  never PASS (the flat-field 15+ frame row is the case that matters: the
  available burst has 9 frames of real scene, not a flat target).
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


def _load(name, filename):
    if SCRIPTS not in sys.path:
        sys.path.insert(0, SCRIPTS)
    path = os.path.join(SCRIPTS, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def av():
    return _load("acceptance_verdict_test", "acceptance_verdict.py")


@pytest.fixture(scope="module")
def acc():
    return _load("sr_acceptance_verdict_test", "sr_acceptance.py")


def _metrics(**over):
    base = {
        "roi": [1650, 1240, 1024, 1024],
        "ratio": 2.0,
        "seconds": 13.0,
        "frames_used": 9,
        "frames_total": 9,
        "shifts": [[0.0, 0.0], [1.2, -0.4], [0.5, 1.1], [2.4, 0.9],
                   [3.3, -1.4], [1.9, 2.8], [4.4, 0.2], [0.7, 3.1],
                   [5.2, -2.2]],
        "hf_gain_db_vs_single_frames": [5.2, 4.8, 5.5, 4.1, 5.0, 4.4, 5.3,
                                        4.9, 5.1],
        "mean_align_score": 0.997,
        "align_residual_mean_px": 0.034,
        "heldout_gain_mean_db": 0.014,
        "heldout_gain_min_db": -0.024,
        "flat_snr": {"snr_gain_db": 4.0, "frames_used": 9,
                     "flat_block_lr": [32, 32, 64]},
    }
    base.update(over)
    return base


def _ctx(**over):
    ctx = {
        "parallax": {"crashed": False, "frames": 5, "robust_active": True,
                     "robust_reason": "moving subject detected", "finite_hr": True},
        "low_coverage": {"crashed": False, "used": 1, "total": 9,
                         "fraction": 1 / 9, "warning": True},
        "threshold_scaling": {"1x": 0.2, "2x": 0.1414, "4x": 0.1},
    }
    ctx.update(over)
    return ctx


def _row(table, cid):
    return next(c for c in table["criteria"] if c["id"] == cid)


# --------------------------------------------------------------------------- #
# The table itself
# --------------------------------------------------------------------------- #

def test_every_criterion_cites_a_spec_location_and_a_threshold(av):
    ids = [c["id"] for c in av.CRITERIA]
    assert len(ids) == len(set(ids))
    for c in av.CRITERIA:
        assert c["spec"], c["id"]
        assert all(s.strip() for s in c["spec"])
        assert c["threshold"] and c["unit"]
    # The SPEC's own rows must all be represented, §7 and §11.
    assert any("§7" in s for c in av.CRITERIA for s in c["spec"])
    assert any("§11" in s for c in av.CRITERIA for s in c["spec"])


def test_spec_line_numbers_match_the_document(av):
    """The cited §7/§11 lines really carry the criterion they claim.

    A citation that silently drifts (SPEC edited, table not) would make the
    verdict point a reader at the wrong sentence, so each row's requirement
    words must appear in the SPEC text around its cited lines.
    """
    import re
    lines = open(os.path.join(ROOT, "docs", "SPEC.md"),
                 encoding="utf-8").read().splitlines()
    all_text = "\n".join(lines).lower()
    for c in av.CRITERIA:
        located = 0
        for ref in c["spec"]:
            m = re.search(r"\bL(\d+)(?:-L(\d+))?", ref)
            if not m:
                continue
            first = int(m.group(1))
            last = int(m.group(2) or first)
            assert 1 <= first <= last <= len(lines), ref
            window = " ".join(lines[first - 1:last]).lower()
            words = [w for w in re.findall(r"[a-z][a-z\-]{4,}",
                                           c["requirement"].lower())]
            assert any(w in window for w in words), (c["id"], ref, window)
            located += 1
        # Every SPEC row is cited by line number, not just by section.
        if any("SPEC.md" in r for r in c["spec"]):
            assert located >= 1, c["id"]
    # And the two acceptance sections still exist under those headings.
    assert "## 7. roadmap" in all_text
    assert "## 11. acceptance criteria" in all_text


def test_verdict_needs_every_row(av):
    table = av.evaluate(_metrics(), _ctx())
    assert [c["id"] for c in table["criteria"]] == [c["id"] for c in av.CRITERIA]
    assert table["counts"]["PASS"] + table["counts"]["FAIL"] + \
        table["counts"]["N/A"] == len(av.CRITERIA)


def test_self_checks_cover_the_three_verdict_inputs(av):
    """The bundle the acceptance script stores must feed S4, S5 and S6."""
    checks = av.run_self_checks(ratio=2.0)
    for key in ("parallax", "low_coverage", "heatmap"):
        assert key in checks, key
        assert "error" not in checks[key], (key, checks[key])
    assert checks["parallax"]["crashed"] is False
    assert checks["low_coverage"]["crashed"] is False
    table = av.evaluate(_metrics(self_checks=checks))
    for cid in ("S4_parallax_no_crash", "S5_alignment_heatmap",
                "S6_low_coverage_warning"):
        assert _row(table, cid)["status"] != av.STATUS_NA, cid

# --------------------------------------------------------------------------- #
# S1: the flat-field row must not be bought with a substitute
# --------------------------------------------------------------------------- #

def test_flat_snr_is_na_with_substitute_when_frames_below_15(av):
    row = _row(av.evaluate(_metrics(), _ctx()), "S1_snr_flat_15f")
    assert row["status"] == av.STATUS_NA
    assert "substitute" in row
    sub = row["substitute"]
    assert sub["frames"] == 9
    # sqrt(N) scaling to 15 frames is arithmetic, not a measurement.
    assert sub["scaled_15f_db"] == pytest.approx(
        sub["snr_gain_db"] + 20 * np.log10(np.sqrt(15 / 9)), abs=1e-6)
    assert "15" in row["detail"]


def test_flat_snr_decides_when_a_15_frame_flat_measurement_exists(av):
    flat = {"snr_gain_db": 4.2, "frames_used": 15, "flat_block_lr": [0, 0, 64]}
    ok = _row(av.evaluate(_metrics(), _ctx(flat_snr=flat)), "S1_snr_flat_15f")
    assert ok["status"] == av.STATUS_PASS and ok["value"] == 4.2
    bad = _row(av.evaluate(_metrics(), _ctx(
        flat_snr=dict(flat, snr_gain_db=1.9))), "S1_snr_flat_15f")
    assert bad["status"] == av.STATUS_FAIL


def test_flat_snr_missing_is_na_not_pass(av):
    m = _metrics()
    del m["flat_snr"]
    row = _row(av.evaluate(m, _ctx(flat_snr=None)), "S1_snr_flat_15f")
    assert row["status"] == av.STATUS_NA and row["value"] is None


def test_flat_region_snr_reports_a_denoising_gain(av):
    rng = np.random.default_rng(4)
    lr_shape = (64, 64)
    ref = np.full(lr_shape + (3,), 0.3, np.float32)
    ref[10:20, 10:20] = 0.6                       # a small feature elsewhere
    noisy = np.clip(ref + rng.normal(0, 0.01, ref.shape), 0, 1).astype(np.float32)
    naive = np.repeat(np.repeat(noisy, 2, axis=0), 2, axis=1)
    clean = np.repeat(np.repeat(ref, 2, axis=0), 2, axis=1)
    sr = np.clip(clean + rng.normal(0, 0.002, clean.shape), 0, 1).astype(np.float32)
    out = av.flat_region_snr(sr, naive, ref[:, :, 1], 2.0, block_lr=16)
    assert out["snr_gain_db"] > 6.0               # 5x less noise
    assert out["flat_block_lr"][2] == 16
    assert out["pitch"] == "lr"
    assert "snr_gain_db_hr_pitch" in out


def test_flat_snr_is_measured_at_input_pitch_not_at_hr_pitch(av):
    """Bicubic upsampling smooths a single frame's noise; only the LR-pitch
    comparison measures the fusion's denoising rather than that smoothing."""
    rng = np.random.default_rng(9)
    lr = np.full((64, 64, 3), 0.3, np.float32)
    sigma = 0.01
    truth_frames = [np.clip(lr + rng.normal(0, sigma, lr.shape), 0, 1)
                    .astype(np.float32) for _ in range(9)]
    averaged = np.mean(truth_frames, axis=0).astype(np.float32)
    gains_lr, gains_hr = [], []
    for r in (2, 3, 4):
        hr = np.repeat(np.repeat(averaged, r, axis=0), r, axis=1)
        naive = __import__("cv2").resize(
            truth_frames[0], (hr.shape[1], hr.shape[0]),
            interpolation=__import__("cv2").INTER_CUBIC)
        d = av.flat_region_snr(hr, naive, lr[:, :, 1], float(r), block_lr=32)
        gains_lr.append(d["snr_gain_db"])
        gains_hr.append(d["snr_gain_db_hr_pitch"])
    # A plain 9-frame average is ~9.5 dB quieter at input pitch, at every ratio.
    assert all(g > 6.0 for g in gains_lr), gains_lr
    assert max(gains_lr) - min(gains_lr) < 1.5, gains_lr
    # The per-HR-pixel variant drifts with ratio for a reason that is not the
    # fusion: the bicubic baseline is smoothed more at higher ratios.
    assert gains_hr[-1] < gains_hr[0] - 2.0, gains_hr


def test_flattest_block_finds_the_quiet_area(av):
    rng = np.random.default_rng(2)
    g = rng.normal(0.3, 0.02, (64, 64)).astype(np.float32)   # noisy
    g[16:48, 0:32] = rng.normal(0.3, 0.001, (32, 32))        # quiet
    y, x, sigma = av.flattest_block(g, block=32, stride=8)
    assert 8 <= y <= 24 and x <= 8


# --------------------------------------------------------------------------- #
# S2: "any single input frame" really means every frame
# --------------------------------------------------------------------------- #

def test_hf_gain_uses_the_worst_frame_not_the_reference(av):
    gains = [6.0, 6.2, -0.4, 5.9, 6.1, 5.8, 6.0, 6.3, 5.7]
    table = av.evaluate(_metrics(hf_gain_db_vs_single_frames=gains), _ctx())
    row = _row(table, "S2_hf_vs_any_single")
    assert row["status"] == av.STATUS_FAIL
    assert row["value"] == -0.4
    assert "+6.30" in row["detail"] and "-0.40" in row["detail"]


def test_hf_gain_falls_back_to_the_single_reference_number(av):
    m = _metrics()
    del m["hf_gain_db_vs_single_frames"]
    m["hf_gain_db_vs_single"] = 0.89
    row = _row(av.evaluate(m, _ctx()), "S2_hf_vs_any_single")
    assert row["status"] == av.STATUS_PASS and row["value"] == 0.89
    assert "reference frame only" in row["source"]


def test_hf_gain_missing_is_na(av):
    m = _metrics()
    del m["hf_gain_db_vs_single_frames"]
    assert _row(av.evaluate(m, _ctx()), "S2_hf_vs_any_single")["status"] == av.STATUS_NA


def test_single_frame_hf_gains_covers_every_frame(av):
    from raw_sr.core.superres import RawFrame, high_frequency_gain_db, to_view_float
    rng = np.random.default_rng(3)
    lr = rng.random((32, 32, 3)).astype(np.float32)
    frames = [RawFrame(name=str(i), rgb=lr.copy(), green=lr[:, :, 1].copy())
              for i in range(4)]
    hr = np.repeat(np.repeat(lr, 2, axis=0), 2, axis=1)
    gains = av.single_frame_hf_gains(hr, frames, high_frequency_gain_db,
                                     to_view_float)
    assert len(gains) == 4 and all(np.isfinite(gains))
    assert av.single_frame_hf_gains(
        hr, frames, high_frequency_gain_db, to_view_float, max_frames=2) == gains[:2]


# --------------------------------------------------------------------------- #
# S3: time only applies to the SPEC's ~500x500 ROI
# --------------------------------------------------------------------------- #

def test_time_is_na_for_a_large_roi_and_decides_at_512(av):
    big = _row(av.evaluate(_metrics(), _ctx()), "S3_time_roi500")
    assert big["status"] == av.STATUS_NA and "not the SPEC" in big["detail"]
    small = _row(av.evaluate(_metrics(roi=[0, 0, 512, 512], seconds=6.0), _ctx()),
                 "S3_time_roi500")
    assert small["status"] == av.STATUS_PASS and small["value"] == 6.0
    slow = _row(av.evaluate(_metrics(roi=[0, 0, 512, 512], seconds=75.0), _ctx()),
                "S3_time_roi500")
    assert slow["status"] == av.STATUS_FAIL


# --------------------------------------------------------------------------- #
# S4 / S6: crash and abort are FAILs; unexercised paths are N/A
# --------------------------------------------------------------------------- #

def test_parallax_crash_is_a_failure(av):
    ctx = _ctx(parallax={"crashed": True, "error": "cv2.error: boom"})
    row = _row(av.evaluate(_metrics(), ctx), "S4_parallax_no_crash")
    assert row["status"] == av.STATUS_FAIL and "boom" in row["detail"]


def test_parallax_without_a_reason_fails_the_indication_half(av):
    ctx = _ctx(parallax={"crashed": False, "frames": 5, "robust_active": False,
                         "robust_reason": "", "finite_hr": True})
    assert _row(av.evaluate(_metrics(), ctx),
                "S4_parallax_no_crash")["status"] == av.STATUS_FAIL


def test_low_coverage_abort_is_a_failure(av):
    ctx = _ctx(low_coverage={"crashed": True, "error": "RuntimeError: abort"})
    row = _row(av.evaluate(_metrics(), ctx), "S6_low_coverage_warning")
    assert row["status"] == av.STATUS_FAIL and "abort" in row["detail"]


def test_low_coverage_that_never_drops_below_20pct_is_na(av):
    ctx = _ctx(low_coverage={"crashed": False, "used": 5, "total": 9,
                             "fraction": 5 / 9, "warning": False})
    assert _row(av.evaluate(_metrics(), ctx),
                "S6_low_coverage_warning")["status"] == av.STATUS_NA


def test_low_coverage_below_threshold_but_no_warning_fails(av):
    ctx = _ctx(low_coverage={"crashed": False, "used": 1, "total": 9,
                             "fraction": 1 / 9, "warning": False})
    assert _row(av.evaluate(_metrics(), ctx),
                "S6_low_coverage_warning")["status"] == av.STATUS_FAIL


def test_alignment_heatmap_decides_when_the_shipped_map_is_available(av):
    """S5 stops being N/A when the shipped map can be exercised headlessly."""
    chk = av.selfcheck_heatmap(roi=128)
    if not chk.get("available"):
        pytest.skip("desktop backend not importable here")
    assert chk["low_inside"], chk
    assert chk["ratio"] < 0.8
    assert chk["frames_used"] >= 5
    row = _row(av.evaluate(_metrics(), _ctx(heatmap=chk)), "S5_alignment_heatmap")
    assert row["status"] == av.STATUS_PASS
    assert row["unit"] == "-"


def test_alignment_heatmap_is_na_when_the_backend_is_missing(av):
    ctx = _ctx(heatmap={"available": False, "error": "desktop backend unavailable"})
    row = _row(av.evaluate(_metrics(), ctx), "S5_alignment_heatmap")
    assert row["status"] == av.STATUS_NA
    assert "unavailable" in row["detail"]
    # A map that does NOT mark the region is a failure, not an N/A.
    ctx = _ctx(heatmap={"available": True, "low_inside": False, "ratio": 1.0,
                        "inside_red_mean": 50.0, "outside_red_mean": 50.0,
                        "region": [0, 0, 16, 16]})
    assert _row(av.evaluate(_metrics(), ctx),
                "S5_alignment_heatmap")["status"] == av.STATUS_FAIL


def test_stored_self_checks_reproduce_the_verdict_without_a_ctx(av):
    """An archived JSON must yield the same verdict as the run that wrote it.

    Without this, re-reading a JSON would silently drop S4/S5/S6 back to N/A
    and the table would under-report what was actually checked.
    """
    heat = {"available": True, "low_inside": True, "ratio": 0.4,
            "inside_red_mean": 60.0, "outside_red_mean": 156.0,
            "region": [115, 134, 153, 153], "frames_used": 9}
    m = _metrics(self_checks={"parallax": _ctx()["parallax"],
                              "low_coverage": _ctx()["low_coverage"],
                              "heatmap": heat})
    from_ctx = av.evaluate(m, _ctx(heatmap=heat))
    from_json = av.evaluate(json.loads(json.dumps(m)))
    for cid in ("S4_parallax_no_crash", "S5_alignment_heatmap",
                "S6_low_coverage_warning"):
        assert _row(from_ctx, cid)["status"] == _row(from_json, cid)["status"]
        assert _row(from_json, cid)["status"] == av.STATUS_PASS


# --------------------------------------------------------------------------- #
# S7 / S8 / P1
# --------------------------------------------------------------------------- #

def test_ratio_reporting_decides_on_both_numbers(av):
    row = _row(av.evaluate(_metrics(), _ctx()), "S7_ratio_reported")
    assert row["status"] == av.STATUS_PASS
    assert row["detail"].count("x") >= 2
    ctx = _ctx(ratio_feedback={"requested": 2.0, "achievable": 1.4})
    bad = _row(av.evaluate(_metrics(), ctx), "S7_ratio_reported")
    assert bad["status"] == av.STATUS_FAIL


def test_ratio_feedback_is_derived_from_the_run_shifts(av):
    info = av.ratio_feedback([[0, 0], [0.3, 0.6], [0.9, 0.1]], 2.0)
    assert info["requested"] == 2.0 and 1.0 <= info["achievable"] <= 2.0
    flat = av.ratio_feedback([[0, 0], [2, 0], [4, 0]], 2.0)
    assert flat["achievable"] < info["achievable"]


def test_threshold_scaling_is_strictly_tighter_with_ratio(av):
    row = _row(av.evaluate(_metrics(), _ctx()), "S8_threshold_scaling")
    assert row["status"] == av.STATUS_PASS
    t = av.threshold_scaling()
    assert t["1x"] > t["2x"] > t["4x"] > 0
    assert t["1x"] == pytest.approx(t["2x"] * np.sqrt(2.0), rel=1e-9)


def test_heldout_gain_decides_on_the_minimum(av):
    row = _row(av.evaluate(_metrics(), _ctx()), "P1_heldout_gain")
    assert row["status"] == av.STATUS_FAIL and row["value"] == -0.024
    ok = _row(av.evaluate(_metrics(heldout_gain_min_db=0.01), _ctx()),
              "P1_heldout_gain")
    assert ok["status"] == av.STATUS_PASS


def test_single_heldout_frame_fallback(av):
    m = _metrics(heldout_gain_db=0.33)
    del m["heldout_gain_mean_db"], m["heldout_gain_min_db"]
    row = _row(av.evaluate(m, _ctx()), "P1_heldout_gain")
    assert row["status"] == av.STATUS_PASS and row["value"] == 0.33


# --------------------------------------------------------------------------- #
# Overall verdict + rendering
# --------------------------------------------------------------------------- #

def test_overall_is_fail_if_any_row_fails_and_incomplete_if_only_na(av):
    t = av.evaluate(_metrics(), _ctx())
    assert t["overall"] == av.STATUS_FAIL
    good = _metrics(heldout_gain_min_db=0.02,
                    hf_gain_db_vs_single_frames=[1.0] * 9)
    t2 = av.evaluate(good, _ctx())
    # S1 (9 frames) and S5 (qualitative) and S3 (large ROI) stay N/A.
    assert t2["counts"]["FAIL"] == 0
    assert t2["overall"] == "INCOMPLETE"


def test_table_rendering_names_every_row_and_the_counts(av):
    t = av.evaluate(_metrics(), _ctx())
    text = av.format_table(t)
    for c in av.CRITERIA:
        assert c["id"] in text
    assert "PASS" in text and "FAIL" in text and "N/A" in text
    assert "verdict: FAIL" in text


def test_evaluate_set_adds_a_global_timing_table(av):
    tables = av.evaluate_set(
        [("2x", _metrics(), {})],
        self_checks={"parallax": _ctx()["parallax"],
                     "low_coverage": _ctx()["low_coverage"]},
        flat_snr=_metrics()["flat_snr"],
        timing={"roi": [0, 0, 512, 512], "seconds": 6.0})
    assert tables[0]["label"] == "2x"
    assert tables[-1]["label"] == "GLOBAL_ROI512"
    assert tables[-1]["criteria"][0]["id"] == "S3_time_roi500"
    assert tables[-1]["counts"]["PASS"] == 1
    assert "2x" in av.format_set(tables)


# --------------------------------------------------------------------------- #
# The CLI reads a stored verdict, or recomputes it
# --------------------------------------------------------------------------- #

def test_cli_prefers_the_stored_verdict(tmp_path, av, capsys):
    stored = av.evaluate(_metrics(), _ctx(), label="2x")
    path = tmp_path / "sr_acceptance_2x.json"
    path.write_text(json.dumps({"acceptance": stored, "seconds": 13.0}))
    assert av.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "verdict: FAIL" in out and "S1_snr_flat_15f" in out


def test_cli_strict_exit_code(tmp_path, av):
    stored = av.evaluate(_metrics(), _ctx(), label="2x")
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"acceptance": stored}))
    assert av.main([str(path), "--strict"]) == 1
    ok = _metrics(heldout_gain_min_db=0.02,
                  hf_gain_db_vs_single_frames=[1.0] * 9,
                  flat_snr={"snr_gain_db": 4.0, "frames_used": 15,
                            "flat_block_lr": [0, 0, 64]})
    good = av.evaluate(ok, _ctx(), label="2x")
    path.write_text(json.dumps({"acceptance": good}))
    assert av.main([str(path), "--strict"]) == 0


def test_cli_recomputes_from_a_plain_metrics_json(tmp_path, av):
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(_metrics()))
    out_path = tmp_path / "verdict.json"
    assert av.main([str(path), "--json-out", str(out_path)]) == 0
    saved = json.loads(out_path.read_text())
    assert saved["tables"][0]["criteria"]
    assert saved["generated_from"] == [str(path)]


# --------------------------------------------------------------------------- #
# The acceptance script exposes the verdict
# --------------------------------------------------------------------------- #

def test_acceptance_cli_has_verdict_switches(acc):
    ap = acc.build_parser()
    a = ap.parse_args([])
    assert a.verdict is True and a.self_checks is True
    assert a.hf_all_frames == 0
    b = ap.parse_args(["--no-verdict", "--no-self-checks",
                       "--hf-all-frames", "3"])
    assert b.verdict is False and b.self_checks is False
    assert b.hf_all_frames == 3


def test_acceptance_verdict_keys_are_documented_in_the_module(acc):
    """The three metric keys the script adds are the ones the verdict reads."""
    src = open(acc.__file__, encoding="utf-8").read()
    for key in ("hf_gain_db_vs_single_frames", "flat_snr", "self_checks",
                "acceptance", "acceptance_failed_ids"):
        assert key in src, key
