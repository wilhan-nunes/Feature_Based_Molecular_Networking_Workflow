#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-07-27

@purpose: convert raw asari output (aligned feature-table TSV + MS/MS MSP) into
          the GNPS-standard quantification CSV and MGF consumed by FBMN.

asari ships the feature table and the MSP with no shared feature id -- the MSP
is keyed by mass-track/cluster (track_<mz>_clus<N>), not by feature. So each
MS/MS spectrum is matched to a feature by precursor m/z (+/- 10 ppm gate) and
retention time (must fall inside the feature's RT bounds), with the closest RT
winning. row ID is assigned 1..N by table position and reused as the MGF SCANS,
so the two outputs share one numbering (the downstream FBMN join is
cluster index == row ID == MGF SCANS).
"""
import sys
import bisect
import pandas as pd

# In an asari aligned feature table, columns up to and including
# "detection_counts" are metadata; every column after it is a per-sample
# intensity.
LAST_METADATA_COLUMN = "detection_counts"
DEFAULT_METADATA_COLUMN_COUNT = 11

# Precursor m/z gate for matching a spectrum to a feature.
PPM_TOLERANCE = 10.0


def convert_to_feature_csv(input_filename, output_filename):
    """Write the GNPS quantification CSV and return an in-memory feature table
    (row ID + mz + rtime + RT bounds + id_number) for the MGF matcher, so both
    outputs share the exact same row ID assignment."""
    df = pd.read_csv(input_filename, sep="\t")
    df = df.reset_index(drop=True)
    columns = list(df.columns)

    if LAST_METADATA_COLUMN in columns:
        boundary = columns.index(LAST_METADATA_COLUMN) + 1
    else:
        print("Warning: '{}' column not found; falling back to fixed metadata "
              "column count {}".format(LAST_METADATA_COLUMN,
                                       DEFAULT_METADATA_COLUMN_COUNT))
        boundary = DEFAULT_METADATA_COLUMN_COUNT

    sample_names = columns[boundary:]

    # row ID 1..N by table position (no zero -- asari's native id starts at F0).
    row_ids = list(range(1, len(df) + 1))

    output_records = []
    for row_id, record in zip(row_ids, df.to_dict(orient="records")):
        output_record = {
            "row ID": str(row_id),
            "row m/z": record["mz"],
            "row retention time": record["rtime"],
        }
        for sample_name in sample_names:
            output_record[sample_name + " Peak area"] = record[sample_name]
        output_records.append(output_record)

    output_headers = ["row ID", "row m/z", "row retention time"]
    output_headers += [sample_name + " Peak area" for sample_name in sample_names]

    output_df = pd.DataFrame(output_records)
    output_df.to_csv(output_filename, sep=",", index=False, columns=output_headers)

    feature_df = pd.DataFrame({
        "row ID": row_ids,
        "id_number": df["id_number"].astype(str).values,
        "mz": df["mz"].astype(float).values,
        "rtime": df["rtime"].astype(float).values,
        "rtime_left_base": df["rtime_left_base"].astype(float).values,
        "rtime_right_base": df["rtime_right_base"].astype(float).values,
    })
    return feature_df


def _parse_msp(input_msp):
    """Parse an asari MSP into a list of spectra, each a dict with id,
    precursor_mz (float), rt (float, seconds) and peaks (list of (mz, intensity)
    strings, preserved verbatim). Empty spectra are dropped."""
    spectra = []
    spec_id = None
    precursor_mz = None
    rt = None
    peaks = []
    reading_peaks = False

    def _flush():
        if precursor_mz is not None and rt is not None and peaks:
            spectra.append({
                "id": spec_id,
                "precursor_mz": precursor_mz,
                "rt": rt,
                "peaks": peaks,
            })

    with open(input_msp) as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if line == "":
                _flush()
                spec_id = None
                precursor_mz = None
                rt = None
                peaks = []
                reading_peaks = False
                continue

            if reading_peaks:
                parts = line.split()
                if len(parts) >= 2:
                    peaks.append((parts[0], parts[1]))
                continue

            lower = line.lower()
            if lower.startswith("id:"):
                spec_id = line.split(":", 1)[1].strip()
            elif lower.startswith("precursormz:"):
                precursor_mz = float(line.split(":", 1)[1].strip())
            elif lower.startswith("retentiontime:"):
                rt = float(line.split(":", 1)[1].strip())
            elif lower.startswith("num peaks:"):
                reading_peaks = True

        # Flush the final block if the file doesn't end with a blank line.
        _flush()

    return spectra


def _match_spectra_to_features(spectra, feature_df):
    """Assign each spectrum to its best feature (m/z +/- 10 ppm gate, RT inside
    the feature's bounds; closest RT wins, then closest m/z), then resolve
    collisions (2+ spectra on one feature) by the same criterion.

    Returns (best_for_feature, n_unmatched):
      best_for_feature: row_id -> (key, spectrum) where key = (rt_diff, mz_diff)
      n_unmatched:      spectra that matched no feature at all
    """
    feats = feature_df.sort_values("mz").reset_index(drop=True)
    mz_arr = feats["mz"].values
    rtime_arr = feats["rtime"].values
    left_arr = feats["rtime_left_base"].values
    right_arr = feats["rtime_right_base"].values
    row_id_arr = feats["row ID"].values

    best_for_feature = {}
    n_unmatched = 0

    for spec in spectra:
        pmz = spec["precursor_mz"]
        rt = spec["rt"]

        # Binary-search a slightly padded m/z window, then apply the exact
        # ppm test (relative to the feature m/z) and RT-bounds test inside.
        window = pmz * (PPM_TOLERANCE + 1.0) * 1e-6
        lo = bisect.bisect_left(mz_arr, pmz - window)
        hi = bisect.bisect_right(mz_arr, pmz + window)

        best = None  # (rt_diff, mz_diff, row_id)
        for j in range(lo, hi):
            fmz = mz_arr[j]
            if abs(fmz - pmz) / fmz * 1e6 > PPM_TOLERANCE:
                continue
            if not (left_arr[j] <= rt <= right_arr[j]):
                continue
            cand = (abs(rtime_arr[j] - rt), abs(fmz - pmz), int(row_id_arr[j]))
            if best is None or cand < best:
                best = cand

        if best is None:
            n_unmatched += 1
            continue

        rt_diff, mz_diff, row_id = best
        key = (rt_diff, mz_diff)
        prev = best_for_feature.get(row_id)
        if prev is None or key < prev[0]:
            best_for_feature[row_id] = (key, spec)

    return best_for_feature, n_unmatched


def convert_mgf(input_msp, output_mgf, feature_df):
    """Match the MSP spectra to features and write one MGF block per matched
    feature, sorted by row ID ascending (so SCANS is monotonically increasing).
    SCANS == row ID; asari's native id_number is carried as ASARI_FEATURE_ID."""
    spectra = _parse_msp(input_msp)
    best_for_feature, n_unmatched = _match_spectra_to_features(spectra, feature_df)

    id_by_row = dict(zip(feature_df["row ID"].astype(int),
                         feature_df["id_number"].astype(str)))

    written = 0
    with open(output_mgf, "w") as out:
        for row_id in sorted(best_for_feature.keys()):
            _, spec = best_for_feature[row_id]
            out.write("BEGIN IONS\n")
            out.write("TITLE=SCAN={}\n".format(row_id))
            out.write("SCANS={}\n".format(row_id))
            out.write("PEPMASS={}\n".format(spec["precursor_mz"]))
            out.write("RTINSECONDS={}\n".format(spec["rt"]))
            out.write("ASARI_FEATURE_ID={}\n".format(id_by_row.get(row_id, "")))
            for mz, intensity in spec["peaks"]:
                out.write("{} {}\n".format(mz, intensity))
            out.write("END IONS\n\n")
            written += 1

    matched = len(spectra) - n_unmatched
    collision_dropped = matched - written
    print("ASARI: {} spectra parsed | {} matched a feature | {} unmatched "
          "(no feature) | {} dropped in collision | {} MGF blocks written".format(
              len(spectra), matched, n_unmatched, collision_dropped, written))
    return written


if __name__ == "__main__":
    # Usage: asari_formatter.py <feature_table.tsv> <out.csv> [<spectra.msp> <out.mgf>]
    feature_df = convert_to_feature_csv(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 4:
        convert_mgf(sys.argv[3], sys.argv[4], feature_df)
