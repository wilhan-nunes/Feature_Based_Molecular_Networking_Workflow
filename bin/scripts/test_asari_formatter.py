import os
import re
import pandas as pd
import asari_formatter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Asari_Raw")
FEATURE_TABLE = os.path.join(DATA_DIR, "full_Feature_table.tsv")
MSP = os.path.join(DATA_DIR, "ms2_spectra.msp")


# --------------------------------------------------------------------------
# Synthetic tests: pin the matching *rules* independent of the real dataset.
# --------------------------------------------------------------------------

def _features(rows):
    """rows: list of (row_id, mz, rtime, left, right)."""
    return pd.DataFrame(
        [{"row ID": r[0], "id_number": "F{}".format(r[0] - 1), "mz": r[1],
          "rtime": r[2], "rtime_left_base": r[3], "rtime_right_base": r[4]}
         for r in rows]
    )


def _spec(pmz, rt):
    return {"id": "s", "precursor_mz": pmz, "rt": rt, "peaks": [("100.0", "1.0")]}


def test_ppm_gate_excludes_outside_10ppm():
    feats = _features([(1, 200.0000, 100.0, 90.0, 110.0)])
    # 15 ppm away -> excluded; 5 ppm away -> included
    match_far, unmatched_far = asari_formatter._match_spectra_to_features([_spec(200.0030, 100.0)], feats)
    match_near, unmatched_near = asari_formatter._match_spectra_to_features([_spec(200.0010, 100.0)], feats)
    assert match_far == {} and unmatched_far == 1
    assert set(match_near.keys()) == {1} and unmatched_near == 0


def test_rt_bounds_gate_is_strict():
    feats = _features([(1, 200.0, 100.0, 95.0, 105.0)])
    # RT inside bounds matches; RT just outside does not
    inside, u_in = asari_formatter._match_spectra_to_features([_spec(200.0, 105.0)], feats)
    outside, u_out = asari_formatter._match_spectra_to_features([_spec(200.0, 105.01)], feats)
    assert set(inside.keys()) == {1} and u_in == 0
    assert outside == {} and u_out == 1


def test_closest_rt_wins_among_candidates():
    # Two same-m/z features whose windows both contain the spectrum RT.
    feats = _features([(1, 400.0, 100.0, 90.0, 160.0),
                       (2, 400.0, 150.0, 90.0, 160.0)])
    match, _ = asari_formatter._match_spectra_to_features([_spec(400.0, 140.0)], feats)
    # RT 140 is closer to feature 2's rtime (150) than feature 1's (100).
    assert set(match.keys()) == {2}


def test_mz_breaks_rt_ties():
    # Two features with identical rtime; RT diff ties -> closest m/z wins.
    feats = _features([(1, 300.0000, 100.0, 90.0, 110.0),
                       (2, 300.0010, 100.0, 90.0, 110.0)])
    match, _ = asari_formatter._match_spectra_to_features([_spec(300.0008, 100.0)], feats)
    # pmz 300.0008 is closer to feature 2 (300.0010) than feature 1 (300.0000).
    assert set(match.keys()) == {2}


def test_collision_keeps_closest_rt_spectrum():
    feats = _features([(1, 200.0, 100.0, 80.0, 120.0)])
    far = _spec(200.0, 110.0)   # rt_diff 10
    near = _spec(200.0, 98.0)   # rt_diff 2  -> should win
    match, _ = asari_formatter._match_spectra_to_features([far, near], feats)
    assert set(match.keys()) == {1}
    assert match[1][1] is near


def test_convert_mgf_monotonic_and_feature_id(tmp_path):
    feats = _features([(1, 200.0, 100.0, 90.0, 110.0),
                       (2, 300.0, 200.0, 190.0, 210.0),
                       (3, 400.0, 300.0, 290.0, 310.0)])  # feature 2 gets no spectrum
    msp = tmp_path / "in.msp"
    msp.write_text(
        "ID: track_a\nPRECURSORMZ: 400.0\nRETENTIONTIME: 300.0\nNum Peaks: 1\n10.0 5.0\n\n"
        "ID: track_b\nPRECURSORMZ: 200.0\nRETENTIONTIME: 100.0\nNum Peaks: 1\n20.0 6.0\n\n"
    )
    out = tmp_path / "out.mgf"
    written = asari_formatter.convert_mgf(str(msp), str(out), feats)
    content = out.read_text()

    assert written == 2
    scans = [int(s) for s in re.findall(r"SCANS=(\d+)", content)]
    assert scans == [1, 3]                       # sorted ascending, feature 2 absent
    assert "ASARI_FEATURE_ID=F0" in content      # row ID 1 -> id_number F0
    assert "ASARI_FEATURE_ID=F2" in content      # row ID 3 -> id_number F2


# --------------------------------------------------------------------------
# Real-data tests against data/Asari_Raw (present locally, untracked).
# --------------------------------------------------------------------------

def test_feature_csv_headers_and_rows(tmp_path):
    out = str(tmp_path / "ft.csv")
    asari_formatter.convert_to_feature_csv(FEATURE_TABLE, out)
    df = pd.read_csv(out)

    assert list(df.columns[:3]) == ["row ID", "row m/z", "row retention time"]

    peak_cols = [c for c in df.columns if c.endswith(" Peak area")]
    assert len(peak_cols) == 13
    assert len(df.columns) == 3 + 13

    # All 41452 features kept, row IDs 1..N, unique, no zero.
    assert len(df) == 41452
    assert df["row ID"].min() == 1
    assert df["row ID"].max() == 41452
    assert df["row ID"].is_unique

    # Spot-check feature 1 (asari F0): m/z 150.0552, RT 61.27.
    row1 = df[df["row ID"] == 1].iloc[0]
    assert abs(row1["row m/z"] - 150.0552) < 1e-3
    assert abs(row1["row retention time"] - 61.27) < 1e-2


def test_mgf_linkage_monotonic_and_known_match(tmp_path):
    csv_out = str(tmp_path / "ft.csv")
    mgf_out = str(tmp_path / "specs.mgf")
    feature_df = asari_formatter.convert_to_feature_csv(FEATURE_TABLE, csv_out)
    written = asari_formatter.convert_mgf(MSP, mgf_out, feature_df)

    content = open(mgf_out).read()
    assert written == 1627
    assert content.count("BEGIN IONS") == 1627
    assert content.count("END IONS") == 1627

    scans = [int(s) for s in re.findall(r"SCANS=(\d+)", content)]
    # Every SCANS is a valid row ID, strictly increasing, unique.
    row_ids = set(pd.read_csv(csv_out)["row ID"].tolist())
    assert set(scans).issubset(row_ids)
    assert scans == sorted(scans)
    assert len(scans) == len(set(scans))

    # Known match: track_150.1280_clus0 (pmz 150.128, RT 147.16) -> row ID 36
    # (asari F35), whose RT window [141.75, 159.51] contains 147.16. F34's
    # window ends at 141.75, so it is correctly rejected.
    first_block = content.split("END IONS")[0]
    assert "SCANS=36" in first_block
    assert "TITLE=SCAN=36" in first_block
    assert "PEPMASS=150.128" in first_block
    assert "ASARI_FEATURE_ID=F35" in first_block
