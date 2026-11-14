# sarsang_model.py   (VERSION v16, reads sarsang_dataset_v9.csv)
#
# THE ONLY PYTHON FILE. Colab: paste the whole file in ONE cell and run it.
# If something is corrected, this same file is edited and VERSION goes up.
#   v2: prints the elevation of the weak-water points per date (shadow-filter check).
#   v3: THREE sets: TRAIN / VALIDATION / TEST. The TEST set is touched only once.
#   v4: DATASET_VERSION follows sarsang_dataset.js v9 (the file name of the CSV).
#   v5: prints what is inside EACH val_*.csv (rows, date, matched points, distance to the
#       dataset points), so a short, empty or mismatching label file is found at once.
#   v6: label files are read SAFELY. Title lines ('--- CSV ...'), repeated headers, cut or
#       wrapped lines and duplicate points (several Console blocks pasted into one file) are
#       handled and COUNTED. Fixes the 'float < str' error caused by empty annotator cells
#       (rows read wrongly); such rows are now reported, not silently dropped.
#   v7: label points are matched to the dataset by DISTANCE (nearest point of the same date
#       within MATCH_TOL_M), not by coordinates rounded to 5 decimals. The label tool and the
#       dataset differ by ~0.2 m in coordinates, so the rounded key lost 15-20 % of the labels.
#   v8: 2020-12-27 moved from TEST to VAL (VAL 5 dates, TEST 3 dates) because its test numbers were
#       seen when the test was opened too early. The test now stays closed until every TEST date
#       has at least MIN_TEST_POINTS_PER_DATE labels. Openings are NOT logged in this version.
#   v9: the weak-label check shows the Sentinel-1 group of each date and LISTS the points where
#       the automatic label and the manual label disagree (with their values), to find why.
#   v10: a label file that cannot be read now shows its FIRST LINE and size (header written with
#       ';' or quotes by Excel/Sheets is accepted); the disagreement list shows exact coordinates;
#       two annotators' disagreements are listed point by point (second blind pass).
#   v11: optional label_corrections.csv (date,lon,lat,label,reason). Corrections are applied on
#       top of the label files, printed one by one (old -> new, reason) and the weak-label
#       precision is shown before and after. The label files themselves are never edited.
#   v12: DIAGNOSTICS section (validation dates only, switch RUN_DIAGNOSTICS): where the water is,
#       skill on hard pixels, prior shift and threshold, extra features, what a few manual
#       labels on hard pixels would add (leave-date-out), and the effect of the Sentinel-1 date offset.
#   v13: two COMPACT feature sets are new model candidates (mndwi + smoothed VV, and the same with
#       ndwi, ndvi, VV-VH). v12 showed that two engineered features beat the 8 raw bands (F1 0.80 vs 0.76).
#       They contain the rule inputs on purpose: see the comment at FEATURE_SETS.
#   v14: diagnostics part 7: how much information each single feature carries on HARD pixels (AUC,
#       water vs land), incl. an ORACLE relative elevation (elevation minus the date's water level
#       taken from the manual labels; a diagnostic, not a model).
#   v15: dataset gets water_level (q99 of the confident-water TRAIN rows of each date; no manual label
#       is used) and elev_rel = elev - water_level. Diagnostics part 8 tests terrain: rules that use
#       elev_rel on hard pixels, a terrain cap on the model, and leave-date-out models with terrain features.
#   v16: diagnostics 8d/8e: LOCAL water level (q90 of the elevation of the 15 nearest confident-water
#       TRAIN points of the same date) instead of one number per date, and the paired F1 difference
#       (with bootstrap CI) of every terrain rule against the rule.
#   v8: the TEST set cannot be opened until EVERY test date is labeled (a premature test run
#       showed the results of one test date). That date (2020-12-27) is now a VALIDATION date,
#       because its test labels were already seen. TEST = the three dates nobody has looked at.
#
# THE THREE SETS
#   TRAIN : weakly labeled points (sarsang_dataset_<version>.csv, set = train), from the TRAIN
#           blocks, all 8 dates. The models are fitted here and nowhere else.
#   VAL   : manual labels of the dates in VAL_DATES. Used to choose the model, the
#           feature set and the best classical rule, and to check the weak labels.
#   TEST  : manual labels of the dates in TEST_DATES. Used ONCE, at the very end, for the
#           numbers that go into the thesis. It stays closed until RUN_FINAL_TEST = True.
#   All manual labels lie in TEST blocks, away from the TRAIN blocks (300 m buffer), so VAL
#   and TEST are spatially independent of TRAIN. VAL and TEST are separated by DATE, so
#   neighbouring points can never sit in both.
#   The final independent check later is round R2: new labels on the hold-out dates
#   (2025-2026).
#
# HOW TO USE
#   1. Label all 8 dates (sarsang_dataset.js, MODE = 'label'), put the val_*.csv files
#      in the Drive folder.
#   2. Run this file with RUN_FINAL_TEST = False. Read the VALIDATION part. Choose the
#      model (CHOSEN_MODEL). Do not look for anything on the test dates.
#   3. Set RUN_FINAL_TEST = True and run again. Write the TEST numbers down. After that,
#      do not change the model or the settings because of the test result.
#
# NOTE: the shore zone is where the methods differ. The stratum "outside the zone" is not
#   reported as accuracy (10 points per date for 91 % of the frame). The weak-label rule
#   is not a fair reference for the models: only manual labels are.

import os, glob, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("SARSANG_DIR", ".")

# ---------------- settings ----------------
DATASET_VERSION = "v9"
DATASET_CSV = os.path.join(DATA_DIR, f"sarsang_dataset_{DATASET_VERSION}.csv")
VAL_GLOB = os.path.join(DATA_DIR, "val_*.csv")       # manual label files from the labeling tool
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Dates are split so that each set has low, medium and high water and different seasons.
# 2020-12-27 was TEST first. A test run opened too early showed its results, so it moved to VAL
# (a date whose test results were seen cannot be a clean test date). Write this in the thesis log.
VAL_DATES = ["2023-03-24", "2023-07-05", "2021-11-19", "2024-06-21", "2020-12-27"]   # 4.1, 6.9, 7.5, 10.5, 5.9 km2
TEST_DATES = ["2023-04-28", "2024-04-27", "2024-11-21"]                              # 2.7, 8.6, 9.1 km2
MIN_TEST_POINTS_PER_DATE = 60   # the test stays closed until every test date has at least this many labels
RUN_FINAL_TEST = False       # True only AFTER the model is chosen. Then do not change anything
CHOSEN_MODEL = "S2 + S1"     # name from FEATURE_SETS; decide it from the VALIDATION part
RUN_DIAGNOSTICS = True       # extra analysis on the validation dates (never touches the test dates)

MAIN_ANNOTATOR = "A1"
# Sentinel-1 group of each date, copied from DEV_DATES in sarsang_dataset.js: orbit pass, days from the Sentinel-2 day.
S1_INFO = {"2023-04-28": "79 DESC, -1 d", "2023-03-24": "79 DESC, -2 d", "2020-12-27": "79 DESC, -1 d",
           "2023-07-05": "79 DESC, +3 d", "2021-11-19": "174 ASC, +2 d", "2024-04-27": "174 ASC, 0 d",
           "2024-11-21": "79 DESC, +2 d", "2024-06-21": "101 ASC, 0 d"}
CORRECTIONS_CSV = os.path.join(DATA_DIR, "label_corrections.csv")   # optional; see the v11 note at the top
MATCH_TOL_M = 3.0             # a label belongs to the dataset point within this distance (points are >= 10 m apart)
SEED = 42
N_BOOT = 1000
MIN_DELTA_F1 = 0.03          # success criterion for H1, written BEFORE seeing results
GATE_B_PRECISION = 0.90      # weak labels should be right in at least 90 % of agree strata
S1_WATER_DB = -16.0
S2 = ["B2", "B3", "B4", "B8", "B11", "B12"]
# The raw feature sets do not contain mndwi / vv_s (the weak labels were made from them).
# The two "Compact" sets DO contain them on purpose: the v12 diagnostics showed that the raw
# bands made the model worse than two engineered features. The evaluation uses MANUAL labels,
# so this does not inflate the result, but a model fed with the rule inputs is a learned,
# smoother version of the rule and not an independent detector (say so in the thesis).
# jrc_occ must never appear.
FEATURE_SETS = {
    "S2 (6 bands)": S2,
    "S1 (VV, VH)": ["VV", "VH"],
    "S2 + S1": S2 + ["VV", "VH"],
    "S2 (no B3, B11) + S1": ["B2", "B4", "B8", "B12", "VV", "VH"],
    "S2 + S1 + elev + slope (ablation)": S2 + ["VV", "VH", "elev", "slope"],
    "Compact: mndwi + smoothed VV": ["mndwi", "vv_s"],
    "Compact + indices: mndwi, ndwi, ndvi, smoothed VV, VV-VH": ["mndwi", "ndwi", "ndvi", "vv_s", "vv_vh"],
}
TIER_MASS = {"T2_water": 1.0, "T2_land": 1.0, "T1_water": 0.5, "T1_land": 0.5}
ALLF = S2 + ["VV", "VH", "mndwi", "vv_s", "elev", "slope"]
os.makedirs(RESULTS_DIR, exist_ok=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from scipy.spatial import cKDTree


def add_idx_cols(d):
    """Indices computed from the bands of the dataset (no new data needed)."""
    d = d.copy()
    d["ndwi"] = (d.B3 - d.B8) / (d.B3 + d.B8).clip(lower=1e-4)
    d["ndvi"] = (d.B8 - d.B4) / (d.B8 + d.B4).clip(lower=1e-4)
    d["swir_ratio"] = d.B11 / d.B12.clip(lower=1e-4)
    d["vv_vh"] = d.VV - d.VH
    return d


def make_rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED)


# ---------------- 1. load and check ----------------
assert os.path.exists(DATASET_CSV), f"Not found: {DATASET_CSV}"
assert CHOSEN_MODEL in FEATURE_SETS, f"CHOSEN_MODEL must be one of {list(FEATURE_SETS)}"
assert not (set(VAL_DATES) & set(TEST_DATES)), "VAL_DATES and TEST_DATES must not share a date"
ds = pd.read_csv(DATASET_CSV)
assert "jrc_occ" not in ds.columns, "jrc_occ must not be in the dataset"
ds = add_idx_cols(ds)
LEVEL_Q = 0.99      # water level of a date = this quantile of the elevation of its confident-water TRAIN rows
_lev = ds[(ds["set"] == "train") & (ds.tier == "T2_water")].groupby("date").elev.quantile(LEVEL_Q)
ds["water_level"] = ds.date.map(_lev)          # from the image of that date only; no manual label is used
ds["elev_rel"] = ds.elev - ds.water_level      # negative = below the date's water level (can be flooded)
tr = ds[ds["set"] == "train"].reset_index(drop=True)
lb = ds[ds["set"] == "label"].reset_index(drop=True)
print(f"Dataset {os.path.basename(DATASET_CSV)}: {len(ds)} rows | train {len(tr)} | label {len(lb)} | dates {ds.date.nunique()}")
unassigned = sorted(set(ds.date) - set(VAL_DATES) - set(TEST_DATES))
if unassigned:
    print("WARNING: dates in neither VAL nor TEST (not evaluated):", unassigned)

print("\n--- train rows per date and tier (target 150 / 150 / 40 / 40 per date) ---")
print(tr.pivot_table(index="date", columns="tier", values="lon", aggfunc="size", fill_value=0).to_string())

ref_elev = tr[tr.tier == "T1_water"].elev.median()
conf_w = ((tr.tier == "T1_water") & (tr.mndwi < -0.1) & (tr.vv_s > S1_WATER_DB)).sum()
shadow = ((tr.tier == "T2_water") & (tr.elev > ref_elev + 60)).sum()
print(f"\nConflicts left in the training rows (must be 0): dry 'permanent water' {conf_w}, terrain-shadow water {shadow}")
print(f"\n--- T2_water elevation per date (m). The shadow filter removes weak water above {ref_elev + 60:.0f} m ---")
print(tr[tr.tier == "T2_water"].groupby("date").elev.agg(max="max", p99=lambda s_: s_.quantile(0.99)).round(0).to_string())
print("If 'max' on the high-water dates is within ~10 m of the cutoff, real water may be cut: raise SHADOW_ABOVE_M in the JS (and the version).")

def to_xy(d, lat0):
    return np.column_stack([d.lon.values * 111320 * np.cos(lat0), d.lat.values * 110540])
lat0 = np.radians(ds.lat.mean())
dist, _ = cKDTree(to_xy(tr, lat0)).query(to_xy(lb, lat0))
print("\nNearest TRAIN point to a LABEL point: min %.0f m | median %.0f m | within 100 m: %.1f %%, within 300 m: %.1f %%" % (
    dist.min(), np.median(dist), 100 * (dist < 100).mean(), 100 * (dist < 300).mean()))

# ---------------- 2. train the models (TRAIN set only) ----------------
tr["w"] = tr.tier.map(TIER_MASS) / tr.groupby("tier").lon.transform("size")
models = {}
for name, cols in FEATURE_SETS.items():
    d = tr.dropna(subset=cols)
    models[name] = (make_rf().fit(d[cols], d.label_weak, sample_weight=d.w), cols)
print(f"\nTrained {len(models)} Random Forest models on the TRAIN set ({len(tr)} weakly labeled rows).")
imp = pd.Series(models["S2 + S1"][0].feature_importances_, index=models["S2 + S1"][1]).sort_values(ascending=False)
print("Feature importance, S2 + S1:", {k: round(v, 3) for k, v in imp.items()})

# ---------------- manual labels ----------------
val_files = sorted(glob.glob(VAL_GLOB))
if not val_files:
    print("\n" + "!" * 78)
    print("No manual labels (val_*.csv) found, so NO accuracy can be reported.")
    print("Label-free look: how often the model says 'water' in each stratum (NOT accuracy):")
    m_, c_ = models[CHOSEN_MODEL]
    d = lb.dropna(subset=c_).copy(); d["pred_water"] = m_.predict(d[c_])
    print(d.groupby("stratum").pred_water.mean().round(2).to_string())
    print("Next: label the 8 dates, put the val_*.csv files in", DATA_DIR, "and run this file again.")
    print("!" * 78)
    raise SystemExit

import io

def uniq(series):
    """Sorted unique values as text; empty cells are left out (they made sorted() fail)."""
    return sorted(str(x) for x in series.dropna().unique())

def read_val(f):
    """Reads one label file safely. Returns (DataFrame, info about what was skipped)."""
    info = dict(before_header=0, titles=0, repeated_header=0, malformed=0, bad_numbers=0, duplicates=0,
                example="", size=0, n_lines=0, first_line="")
    try:
        text = open(f, encoding="utf-8-sig", errors="replace").read()
    except Exception as e:
        print(f"  could not read {os.path.basename(f)}: {e}")
        return pd.DataFrame(), info
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    info["size"] = len(text); info["n_lines"] = len(lines); info["first_line"] = lines[0][:110] if lines else ""
    hi = next((i for i, ln in enumerate(lines) if ln.lstrip("\"' ").lower().startswith("annotator")), None)
    if hi is None:
        return pd.DataFrame(), info
    info["before_header"] = hi
    header = lines[hi]
    delim = max([",", ";", "\t"], key=header.count)          # Excel in some languages saves with ';'
    n_delim = header.count(delim)
    kept = [header]
    for ln in lines[hi + 1:]:
        if ln == header:
            info["repeated_header"] += 1
        elif ln.startswith("---"):
            info["titles"] += 1
        elif ln.count(delim) != n_delim:
            info["malformed"] += 1                           # cut or wrapped line
            if not info["example"]:
                info["example"] = ln[:90]
        else:
            kept.append(ln)
    body = "\n".join(kept)
    v = pd.read_csv(io.StringIO(body), sep=delim)
    v.columns = [str(c).strip().strip('"').lower() for c in v.columns]
    if delim == ";" and "lon" in v.columns and not pd.api.types.is_numeric_dtype(v["lon"]):
        v = pd.read_csv(io.StringIO(body), sep=";", decimal=",")      # decimal comma
        v.columns = [str(c).strip().strip('"').lower() for c in v.columns]
    if any(c not in v.columns for c in ["date", "lon", "lat", "label"]):
        return pd.DataFrame(), info                          # header exists but required columns are missing
    for c in ["lon", "lat", "label", "stratum"]:
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce")
    bad = v[["lon", "lat", "label"]].isna().any(axis=1)
    info["bad_numbers"] = int(bad.sum()); v = v[~bad]
    n0 = len(v)
    v = v.drop_duplicates(["date", "lon", "lat"], keep="last")    # a later block overrides an earlier one
    info["duplicates"] = n0 - len(v)
    return v.reset_index(drop=True), info

print("\n--- manual label files: what each file contains (expected about 90 rows per date) ---")
print(f"Each label is matched to the nearest dataset label point of the same date within {MATCH_TOL_M:.0f} m.")
frames = []
lb["pid"] = lb.index
for f in val_files:
    v, info = read_val(f)
    name = os.path.basename(f)
    if len(v) == 0 or "lon" not in v.columns:
        print(f"{name}: NO USABLE ROWS | {info['size']} characters, {info['n_lines']} lines | first line: '{info['first_line']}'")
        if info["size"] == 0:
            print("   Colab sees this file as EMPTY. If it is fine on Drive, the copy Colab sees is not updated yet: Runtime > Restart session, then run again.")
        else:
            print("   A label file needs a header line starting with 'annotator' (annotator,date,order,lon,lat,label,...). Send me the first line shown above.")
        continue
    v = v.copy(); v["pid"] = np.nan; v["match_dist"] = np.nan
    for d_, g in v.groupby("date"):
        ref = lb[lb.date == d_]
        if len(ref) == 0:
            print(f"   date {d_} in {name} does not exist in the dataset"); continue
        dd, ii = cKDTree(to_xy(ref, lat0)).query(to_xy(g, lat0))
        ok = dd <= MATCH_TOL_M
        v.loc[g.index[ok], "pid"] = ref.pid.values[ii[ok]]
        v.loc[g.index, "match_dist"] = dd
    frames.append(v)
    matched = int(v.pid.notna().sum())
    md = v.match_dist.dropna()
    lab = v.label.value_counts().to_dict()
    blank_ann = int(v.annotator.isna().sum())
    print(f"{name}: rows {len(v)} | date {uniq(v.date)} | annotator {uniq(v.annotator)} | "
          f"water/land/unsure {lab.get(1, 0)}/{lab.get(0, 0)}/{lab.get(-1, 0)} | "
          f"matched {matched} of {len(v)}" + (f" (distance of matched rows: median {v.loc[v.pid.notna(), 'match_dist'].median():.2f} m, "
          f"max {v.loc[v.pid.notna(), 'match_dist'].max():.2f} m)" if matched else ""))
    skipped = {k_: v_ for k_, v_ in info.items() if k_ not in ("example", "size", "n_lines", "first_line") and v_}
    if skipped:
        print(f"   file needed cleaning: {skipped}" + (f" | first bad line: '{info['example']}'" if info["example"] else ""))
    if blank_ann:
        print(f"   WARNING: {blank_ann} rows have an EMPTY annotator cell; they are NOT used (check the file)")
    if matched < len(v):
        print(f"   -> {len(v) - matched} rows are farther than {MATCH_TOL_M:.0f} m from every dataset label point of their date "
              f"(nearest: {md[v.pid.isna()].min():.1f} m). The label asset and the dataset do not belong together, or the date in the file is wrong")
    same = int(v.pid.dropna().duplicated().sum())
    if same:
        print(f"   NOTE: {same} rows point to the same dataset point as another row; the last one is used")
if not frames:
    print("No usable label file. Check the files in", DATA_DIR); raise SystemExit
val = pd.concat(frames, ignore_index=True)
is_main = val.annotator.astype(str).str.strip() == MAIN_ANNOTATOR
main = val[is_main & val.pid.notna()][["pid", "label", "stratum"]].rename(columns={"stratum": "stratum_val"})
main["pid"] = main.pid.astype(int)
main = main.drop_duplicates("pid", keep="last")
main["label_original"] = main.label

def pooled_prec(e):
    """'agree water' and 'agree land' points whose manual label agrees with the rule, validation dates only."""
    out = []
    for s_, good in [(1, 1), (0, 0)]:
        g = e[(e.stratum == s_) & e.label.isin([0, 1]) & e.date.isin(VAL_DATES)]
        out.append(f"{int((g.label == good).sum())}/{len(g)}")
    return out

if os.path.exists(CORRECTIONS_CSV):
    try:
        corr = pd.read_csv(CORRECTIONS_CSV, sep=None, engine="python", skipinitialspace=True)
        corr.columns = [str(c).strip().lower() for c in corr.columns]
    except Exception as e:
        corr = None
        print(f"WARNING: could not read {os.path.basename(CORRECTIONS_CSV)}: {e}. It was NOT used.")
    if corr is not None and not all(c in corr.columns for c in ["date", "lon", "lat", "label"]):
        print(f"WARNING: {os.path.basename(CORRECTIONS_CSV)} needs the columns date, lon, lat, label (and reason). It was NOT used.")
    elif corr is not None:
        if "reason" not in corr.columns:
            corr["reason"] = ""
        applied = []
        for _, r in corr.iterrows():
            d_ = str(r["date"]).strip()
            ref = lb[lb.date == d_]
            if len(ref) == 0:
                print(f"   correction ignored: date {d_} is not in the dataset"); continue
            dd, ii = cKDTree(to_xy(ref, lat0)).query(to_xy(pd.DataFrame({"lon": [float(r["lon"])], "lat": [float(r["lat"])]}), lat0))
            if dd[0] > MATCH_TOL_M:
                print(f"   correction ignored: {d_} {r['lon']} {r['lat']} has no dataset point within {MATCH_TOL_M:.0f} m (nearest {dd[0]:.1f} m)"); continue
            pid_ = int(ref.pid.values[ii[0]])
            if not (main.pid == pid_).any():
                print(f"   correction ignored: {d_} {r['lon']} {r['lat']} has no label from {MAIN_ANNOTATOR}"); continue
            old_ = int(main.loc[main.pid == pid_, "label"].iloc[0])
            main.loc[main.pid == pid_, "label"] = int(r["label"])
            applied.append(dict(date=d_, lon=float(ref.lon.values[ii[0]]), lat=float(ref.lat.values[ii[0]]), old=old_, new=int(r["label"]), reason=str(r["reason"])))
        if applied:
            ap = pd.DataFrame(applied)
            before = pooled_prec(lb.merge(main.assign(label=main.label_original), on="pid"))
            after = pooled_prec(lb.merge(main, on="pid"))
            print(f"\nLABEL CORRECTIONS from {os.path.basename(CORRECTIONS_CSV)}: {len(ap)} applied, {int((ap.old != ap.new).sum())} change the label (0 = land, 1 = water, -1 = unsure)")
            print(ap.round({"lon": 6, "lat": 6}).to_string(index=False))
            print(f"Validation dates, weak-label precision: agree water {before[0]} -> {after[0]} | agree land {before[1]} -> {after[1]}   (before -> after the corrections)")
            hit = ap[ap.date.isin(TEST_DATES)]
            if len(hit):
                print(f"WARNING: {len(hit)} corrections are on TEST dates. Test labels must not be changed after the test results are known.")
ev = lb.merge(main, on="pid", how="inner")
print(f"\nManual label files: {len(val_files)} | annotators {uniq(val.annotator)} | "
      f"label rows with a label from {MAIN_ANNOTATOR}: {len(ev)} of {len(lb)}")
bad_s = ev[ev.stratum != ev.stratum_val]
if len(bad_s):
    print("WARNING: stratum differs between the dataset and the label file for", len(bad_s), "points. First ones:")
    print(bad_s[["date", "lon", "lat", "stratum", "stratum_val"]].head(3).to_string(index=False))
ev_val = ev[ev.date.isin(VAL_DATES)].reset_index(drop=True)
ev_test = ev[ev.date.isin(TEST_DATES)].reset_index(drop=True)
per_date = ev.groupby("date").size().reindex(sorted(set(VAL_DATES) | set(TEST_DATES)), fill_value=0)
print("labeled points per date:", per_date.to_dict())
miss = [d for d in VAL_DATES + TEST_DATES if per_date.get(d, 0) == 0]
if miss:
    print("WARNING: no manual labels yet for:", miss)
print(f"VALIDATION set: {len(ev_val)} points on {len(VAL_DATES)} dates | TEST set: {len(ev_test)} points on {len(TEST_DATES)} dates (closed unless RUN_FINAL_TEST)")

# ---------------- estimators (area-weighted, stratified) ----------------
def make_cells(df):
    """Cells = date x stratum inside the shore zone. Weight = pixel count of the cell."""
    df = df.reset_index(drop=True)
    cells, N = [], []
    for _, g in df.groupby(["date", "stratum"]):
        cells.append(g.index.values); N.append(g.stratum_px.iloc[0])
    N = np.array(N, float)
    return df, cells, N / N.sum()

def est(x, z, cells, W):
    """Ratio estimator sum(W*mean(x)) / sum(W*mean(z)) with its standard error."""
    num = sum(w * x[c].mean() for w, c in zip(W, cells))
    den = sum(w * z[c].mean() for w, c in zip(W, cells))
    R = num / den
    var = 0.0
    for w, c in zip(W, cells):
        n = len(c)
        if n < 2:
            continue
        xc, zc = x[c], z[c]
        cov = np.cov(xc, zc, ddof=1)[0, 1] if zc.std() > 0 else 0.0
        var += w ** 2 * (xc.var(ddof=1) + R ** 2 * zc.var(ddof=1) - 2 * R * cov) / n
    return R, np.sqrt(max(var, 0.0)) / den

def fmt(R, se):
    return f"{R:.3f} ({max(R - 1.96 * se, 0):.3f}-{min(R + 1.96 * se, 1):.3f})"

def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h

# classical rules. The Otsu threshold comes from the TRAIN set only.
otsu_src = tr.mndwi.dropna().values
hist, edges = np.histogram(otsu_src, bins=256); centers = (edges[:-1] + edges[1:]) / 2
w0 = np.cumsum(hist); w1 = w0[-1] - w0; s_ = np.cumsum(hist * centers)
m0 = s_ / np.maximum(w0, 1); m1 = (s_[-1] - s_) / np.maximum(w1, 1)
otsu_t = centers[np.argmax(w0 * w1 * (m0 - m1) ** 2)]

def get_frame(ev_set):
    d = ev_set[ev_set.stratum.isin([1, 0, -1]) & ev_set.label.isin([0, 1])].dropna(subset=ALLF)
    if len(d) == 0:
        return None
    sh, cells, W = make_cells(d)
    return sh, cells, W, sh.label.values.astype(float)

def get_preds(sh):
    p = {}
    for name, (m, cols) in models.items():
        p["RF: " + name] = (m.predict_proba(sh[cols])[:, 1] >= 0.5).astype(float)
    p["Rule: MNDWI > 0"] = (sh.mndwi > 0).astype(float).values
    p[f"Rule: MNDWI > Otsu on train ({otsu_t:.2f})"] = (sh.mndwi > otsu_t).astype(float).values
    p[f"Rule: S1 VV (smoothed) < {S1_WATER_DB:.0f} dB"] = (sh.vv_s < S1_WATER_DB).astype(float).values
    p["Rule: MNDWI > 0.1 and VV < -16 dB"] = ((sh.mndwi > 0.1) & (sh.vv_s < S1_WATER_DB)).astype(float).values
    return p

def gate_b(sh, tag):
    print(f"\n=== [{tag}] Weak-label precision: how often the manual label agrees with the rule ===")
    rows = []
    for d, g in sh[sh.stratum.isin([1, 0])].groupby("date"):
        for s, name, good in [(1, "agree water", 1), (0, "agree land", 0)]:
            gg = g[g.stratum == s]; k = int((gg.label == good).sum()); n = len(gg)
            lo, hi = wilson(k, n)
            rows.append(dict(date=d, s1=S1_INFO.get(d, ""), stratum=name, n=n, correct=k, precision=k / n if n else np.nan, lo=lo, hi=hi))
    gb = pd.DataFrame(rows); print(gb.round(3).to_string(index=False))
    for s, name, good in [(1, "agree water", 1), (0, "agree land", 0)]:
        gg = sh[sh.stratum == s]; k = int((gg.label == good).sum()); n = len(gg); lo, hi = wilson(k, n)
        verdict = "OK" if lo >= GATE_B_PRECISION else ("BELOW the gate" if hi < GATE_B_PRECISION else "unclear")
        print(f"POOLED {name}: {k}/{n} = {k / max(n, 1):.3f} (95 % CI {lo:.3f}-{hi:.3f}); gate {GATE_B_PRECISION:.2f}: {verdict}")
    return gb

def gate_failures(sh, tag):
    """Lists the points where the automatic label (the rule) and the manual label disagree."""
    cols = ["date", "lon", "lat", "mndwi", "vv_s", "B3", "B8", "B11", "elev", "slope"]
    for s_, name, wrong, wtxt in [(1, "agree water", 0, "land"), (0, "agree land", 1, "water")]:
        bad = sh[(sh.stratum == s_) & (sh.label == wrong)]
        if len(bad) == 0:
            continue
        ok = sh[(sh.stratum == s_) & (sh.label != wrong)]
        print(f"\n--- [{tag}] '{name}' points that the manual label calls {wtxt}: {len(bad)} (first 20) ---")
        print(bad[cols].round({c_: (6 if c_ in ("lon", "lat") else 3) for c_ in cols if c_ != "date"}).head(20).to_string(index=False))
        print("median of the points that were RIGHT in the same group:", ok[cols[3:]].median().round(3).to_dict())
        print("count of wrong points per date:", bad.groupby("date").size().to_dict())

def results_table(sh, cells, W, y, preds, tag):
    print(f"\n=== [{tag}] Shore zone: area-weighted accuracy and F1 (water), 95 % CI ===")
    rows = []
    hard = sh.stratum.values == -1
    for name, p in preds.items():
        ok = (p == y).astype(float)
        a, sa = est(ok, np.ones_like(ok), cells, W); f, sf = est(2 * p * y, p + y, cells, W)
        rows.append(dict(method=name, accuracy=fmt(a, sa), F1_water=fmt(f, sf), F1=f, acc_hard_stratum=round(float(ok[hard].mean()), 3)))
    res = pd.DataFrame(rows); print(res.drop(columns="F1").to_string(index=False))
    return res

def wratio(x, z, groups, W):
    return sum(w * x[g].mean() for w, g in zip(W, groups)) / sum(w * z[g].mean() for w, g in zip(W, groups))

def h1_table(sh, cells, W, y, preds, rule, model_names, tag):
    print(f"\n=== [{tag}] H1: model minus the rule '{rule}', F1 of water in the shore zone ===")
    print(f"Criterion (set before the results): difference >= {MIN_DELTA_F1} and the 95 % CI above 0")
    rng = np.random.default_rng(SEED)
    boots = [[rng.choice(c, len(c)) for c in cells] for _ in range(N_BOOT)]
    xr, zr = 2 * preds[rule] * y, preds[rule] + y
    rows = []
    for name in model_names:
        xa, za = 2 * preds[name] * y, preds[name] + y
        d0 = wratio(xa, za, cells, W) - wratio(xr, zr, cells, W)
        bd = np.array([wratio(xa, za, b, W) - wratio(xr, zr, b, W) for b in boots])
        lo, hi = np.percentile(bd, [2.5, 97.5])
        rows.append(dict(model=name, delta_F1=round(d0, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3), criterion_met=bool((d0 >= MIN_DELTA_F1) and (lo > 0))))
    h = pd.DataFrame(rows); print(h.to_string(index=False)); return h

def lodo(sh, cells, W, y, model_name, tag, std_row):
    print(f"\n=== [{tag}] Leave-one-date-out ({model_name}): each date is predicted by a model that never saw that date ===")
    cols = FEATURE_SETS[model_name]; p = np.zeros(len(sh))
    for d in sh.date.unique():
        d_tr = tr[tr.date != d].dropna(subset=cols)
        m = make_rf().fit(d_tr[cols], d_tr.label_weak, sample_weight=d_tr.w)
        idx = (sh.date == d).values
        p[idx] = (m.predict_proba(sh.loc[idx, cols])[:, 1] >= 0.5)
    ok = (p == y).astype(float)
    a, sa = est(ok, np.ones_like(ok), cells, W); f, sf = est(2 * p * y, p + y, cells, W)
    print(f"accuracy {fmt(a, sa)} | F1 water {fmt(f, sf)}   (standard model: {std_row})")

def unsure_case(ev_set, model_name, tag):
    print(f"\n=== [{tag}] Unsure labels: best case / worst case accuracy ({model_name}) ===")
    cols = FEATURE_SETS[model_name]
    un = ev_set[ev_set.stratum.isin([1, 0, -1]) & ev_set.label.isin([0, 1, -1])].dropna(subset=cols + ["stratum_px"])
    if len(un) == 0:
        print("no points"); return
    un, cells_u, W_u = make_cells(un)
    pu = (models[model_name][0].predict_proba(un[cols])[:, 1] >= 0.5).astype(float)
    decided = un.label.isin([0, 1]).values
    ok_dec = (pu == un.label.values.astype(float)) & decided
    ones = np.ones(len(un))
    best = est((ok_dec | ~decided).astype(float), ones, cells_u, W_u); worst = est(ok_dec.astype(float), ones, cells_u, W_u)
    print(f"share of Unsure in the shore zone: {100 * (~decided).mean():.1f} % | accuracy: worst case {worst[0]:.3f} | best case {best[0]:.3f}")

def kappa_report(ev_set, tag):
    for a2 in [a for a in uniq(val.annotator) if a != MAIN_ANNOTATOR]:
        v2 = val[(val.annotator.astype(str).str.strip() == a2) & val.pid.notna()][["pid", "label"]].rename(columns={"label": "label2"})
        v2["pid"] = v2.pid.astype(int); v2 = v2.drop_duplicates("pid", keep="last")
        pair = ev_set[["pid", "label"]].merge(v2, on="pid", how="inner")
        if len(pair) > 5:
            print(f"[{tag}] Cohen's kappa {MAIN_ANNOTATOR} vs {a2}: {cohen_kappa_score(pair.label, pair.label2):.3f} on {len(pair)} common points")
            dis = pair[pair.label != pair.label2]
            if len(dis):
                dd_ = dis.merge(lb[["pid", "date", "lon", "lat"]], on="pid").round({"lon": 6, "lat": 6})
                print(f"   {len(dis)} points where {MAIN_ANNOTATOR} and {a2} disagree (first 15): label = {MAIN_ANNOTATOR}, label2 = {a2}")
                print(dd_[["date", "lon", "lat", "label", "label2"]].head(15).to_string(index=False))

# ======================================================================
# VALIDATION  (always runs)
# ======================================================================
print("\n" + "=" * 78)
print(f"VALIDATION SET, dates {VAL_DATES}: used to choose the model")
print("=" * 78)
fv = get_frame(ev_val)
if fv is None:
    print("No validation labels yet. Label the validation dates first."); raise SystemExit
sh_v, cells_v, W_v, y_v = fv
print(f"Shore-zone frame: {len(sh_v)} points in {len(cells_v)} cells (date x stratum), unsure labels excluded")
gb_v = gate_b(sh_v, "VAL")
gate_failures(sh_v, "VAL")
preds_v = get_preds(sh_v)
res_v = results_table(sh_v, cells_v, W_v, y_v, preds_v, "VAL")
rf_names = [k for k in preds_v if k.startswith("RF:")]
rule_names = [k for k in preds_v if k.startswith("Rule:")]
best_model_val = max(rf_names, key=lambda r: res_v.set_index("method").loc[r, "F1"])[4:]
best_rule_val = max(rule_names, key=lambda r: res_v.set_index("method").loc[r, "F1"])
print(f"\nBest model on VALIDATION (F1 water): {best_model_val}   | best classical rule on VALIDATION: {best_rule_val}")
print(f"CHOSEN_MODEL in this file: {CHOSEN_MODEL}" + ("" if CHOSEN_MODEL == best_model_val else "   <- differs from the best on validation; fine if you have a reason"))
h1_v = h1_table(sh_v, cells_v, W_v, y_v, preds_v, best_rule_val, rf_names, "VAL (development comparison)")
lodo(sh_v, cells_v, W_v, y_v, CHOSEN_MODEL, "VAL", res_v.set_index("method").loc["RF: " + CHOSEN_MODEL, "F1_water"])
unsure_case(ev_val, CHOSEN_MODEL, "VAL")
kappa_report(ev_val, "VAL")
res_v.drop(columns="F1").to_csv(os.path.join(RESULTS_DIR, f"val_results_{DATASET_VERSION}.csv"), index=False)
h1_v.to_csv(os.path.join(RESULTS_DIR, f"val_h1_{DATASET_VERSION}.csv"), index=False)
gb_v.to_csv(os.path.join(RESULTS_DIR, f"val_weak_label_precision_{DATASET_VERSION}.csv"), index=False)
try:
    import joblib
    joblib.dump({"model": models[CHOSEN_MODEL][0], "features": FEATURE_SETS[CHOSEN_MODEL]},
                os.path.join(RESULTS_DIR, f"rf_chosen_{DATASET_VERSION}.joblib"))
except Exception as e:
    print("model not saved:", e)

# ======================================================================
# DIAGNOSTICS  (validation dates only; the test dates are never used here)
# ======================================================================
if RUN_DIAGNOSTICS:
    import re
    print("\n" + "=" * 78)
    print("DIAGNOSTICS on the validation dates: where does the model lose against the rule?")
    print("=" * 78)
    grp = {1: "agree water", 0: "agree land", -1: "hard"}
    strat_v = sh_v.stratum.values
    hard = strat_v == -1
    hw_m, hl_m = hard & (y_v == 1), hard & (y_v == 0)

    def eval_row(p):
        ok = (p == y_v).astype(float)
        a, sa = est(ok, np.ones_like(ok), cells_v, W_v); f, sf = est(2 * p * y_v, p + y_v, cells_v, W_v)
        return dict(accuracy=fmt(a, sa), F1_water=fmt(f, sf), F1=round(float(f), 3),
                    hard_water_found=round(float(p[hw_m].mean()), 2) if hw_m.any() else np.nan,
                    hard_land_called_water=round(float(p[hl_m].mean()), 2) if hl_m.any() else np.nan)

    # ---- 1. where the water is ----
    mass = {}
    for w, c in zip(W_v, cells_v):
        s_ = int(strat_v[c[0]]); mass[s_] = mass.get(s_, 0.0) + w * y_v[c].mean()
    tot = sum(mass.values())
    print("\n--- 1. Share of ALL water (area-weighted) that lies in each group of the shore zone ---")
    for s_ in (1, -1, 0):
        print(f"   {grp[s_]:12s} {100 * mass.get(s_, 0) / tot:5.1f} %")
    print(f"   A method that calls water ONLY where both sensors agree finds at most {100 * mass.get(1, 0) / tot:.0f} % of the water (recall ceiling).")
    print(f"   hard points: {int(hard.sum())}, of which water by hand: {int(hw_m.sum())} ({100 * hw_m.sum() / max(hard.sum(), 1):.0f} %)")

    # ---- 2. skill by group ----
    def by_group(p):
        ok = (p == y_v); out = {}
        for s_ in (1, 0, -1):
            m = strat_v == s_
            out[f"acc {grp[s_]}"] = round(float(ok[m].mean()), 3) if m.any() else np.nan
        out["hard: water found"] = round(float(p[hw_m].mean()), 2) if hw_m.any() else np.nan
        out["hard: land called water"] = round(float(p[hl_m].mean()), 2) if hl_m.any() else np.nan
        return out
    rows_g = {}
    for nm in dict.fromkeys(["RF: " + CHOSEN_MODEL, "RF: " + best_model_val, best_rule_val, "Rule: MNDWI > 0"]):
        rows_g[nm] = by_group(preds_v[nm])
    rows_g["constant: everything land"] = by_group(np.zeros(len(sh_v)))
    print("\n--- 2. Accuracy by group (points). On hard pixels the rule says 'land' everywhere ---")
    print(pd.DataFrame(rows_g).T.to_string())
    print("   A model that is worse than 'everything land' on hard pixels has NO skill there: it adds false water.")

    # ---- 3. prior shift and threshold ----
    cols_c = FEATURE_SETS[CHOSEN_MODEL]
    proba = models[CHOSEN_MODEL][0].predict_proba(sh_v[cols_c])[:, 1]
    pi_t = float((tr.w * tr.label_weak).sum() / tr.w.sum())
    pi_e = float(sum(w * y_v[c].mean() for w, c in zip(W_v, cells_v)))
    r_ = (pi_e / (1 - pi_e)) / (pi_t / (1 - pi_t))
    p_adj = proba * r_ / (proba * r_ + (1 - proba))
    print(f"\n--- 3. Prior shift: share of water in training {100 * pi_t:.0f} %  vs  in the validation shore zone {100 * pi_e:.0f} % ---")
    grid = np.arange(0.1, 0.91, 0.05)
    best_t = max(grid, key=lambda t: eval_row((proba >= t).astype(float))["F1"])
    rows_t = {"threshold 0.5 (as trained)": eval_row((proba >= 0.5).astype(float)),
              "prior-corrected (threshold 0.5 after correction)": eval_row((p_adj >= 0.5).astype(float)),
              f"best fixed threshold {best_t:.2f} (tuned on validation: optimistic)": eval_row((proba >= best_t).astype(float)),
              best_rule_val: eval_row(preds_v[best_rule_val])}
    print(pd.DataFrame(rows_t).T.to_string())

    # ---- 4. features (weak labels only) ----
    def add_idx(d):
        d = d.copy()
        d["ndwi"] = (d.B3 - d.B8) / (d.B3 + d.B8).clip(lower=1e-4)
        d["ndvi"] = (d.B8 - d.B4) / (d.B8 + d.B4).clip(lower=1e-4)
        d["swir_ratio"] = d.B11 / d.B12.clip(lower=1e-4)
        d["vv_vh"] = d.VV - d.VH
        return d
    tr_i, sh_i = add_idx(tr), add_idx(sh_v)
    IDX = ["ndwi", "ndvi", "swir_ratio", "vv_vh"]
    variants = {"S2 + S1 (chosen)": cols_c,
                "S2 + S1 + indices (ndwi, ndvi, swir ratio, VV-VH)": cols_c + IDX,
                "S2 + S1 + rule inputs (mndwi, smoothed VV)": cols_c + ["mndwi", "vv_s"],
                "only rule inputs (mndwi, smoothed VV)": ["mndwi", "vv_s"]}
    rows_f = {}
    for nm, fc in variants.items():
        d_ = tr_i.dropna(subset=fc)
        m_ = make_rf().fit(d_[fc], d_.label_weak, sample_weight=d_.w)
        rows_f[nm] = eval_row((m_.predict_proba(sh_i[fc])[:, 1] >= 0.5).astype(float))
    rows_f[best_rule_val] = eval_row(preds_v[best_rule_val])
    print("\n--- 4. Same weak-label training, more or different features ---")
    print(pd.DataFrame(rows_f).T.to_string())
    print("   'only rule inputs' reproduces the rule: it shows what the weak labels can teach at most.")

    # ---- 5. what manual labels on hard pixels would add (leave-date-out) ----
    def lodo_variant(fc, manual_frac, use_weak=True):
        p = np.zeros(len(sh_i))
        for d in sh_i.date.unique():
            te = (sh_i.date == d).values
            man = sh_i[~te].dropna(subset=fc)
            Xs, ys, ws = [], [], []
            wk_mass = 0.0
            if use_weak:
                wk = tr_i[tr_i.date != d].dropna(subset=fc)
                Xs.append(wk[fc]); ys.append(wk.label_weak.values); ws.append(wk.w.values); wk_mass = float(wk.w.sum())
            if (manual_frac > 0 or not use_weak) and len(man):
                mw = manual_frac * wk_mass / len(man) if use_weak else 1.0
                Xs.append(man[fc]); ys.append(man.label.values); ws.append(np.full(len(man), mw))
            m_ = make_rf().fit(pd.concat(Xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
            p[te] = (m_.predict_proba(sh_i.loc[te, fc])[:, 1] >= 0.5)
        return p
    rows_m = {}
    for fname, fc in [("S2+S1", cols_c), ("S2+S1+indices", cols_c + IDX)]:
        rows_m[f"{fname}: weak labels only"] = eval_row(lodo_variant(fc, 0.0))
        rows_m[f"{fname}: weak + manual (manual = 50 % of weak weight)"] = eval_row(lodo_variant(fc, 0.5))
        rows_m[f"{fname}: weak + manual (manual = 100 % of weak weight)"] = eval_row(lodo_variant(fc, 1.0))
        rows_m[f"{fname}: manual labels only"] = eval_row(lodo_variant(fc, 0.0, use_weak=False))
    rows_m[best_rule_val] = eval_row(preds_v[best_rule_val])
    print("\n--- 5. Leave-one-date-out: each date is predicted by a model that never saw that date (not its weak rows, not its manual labels) ---")
    print("    'manual' = the validation labels of the OTHER dates, hard pixels included (a small simulation of labeling hard pixels)")
    print(pd.DataFrame(rows_m).T.to_string())

    # ---- 6. Sentinel-1 date offset ----
    rows_o = []
    for d in sorted(sh_v.date.unique()):
        m = (sh_v.date == d).values; hm_d = m & hard
        info_ = S1_INFO.get(d, ""); mm = re.search(r"([+-]?\d+) d", info_)
        off = abs(int(mm.group(1))) if mm else np.nan
        rows_o.append(dict(date=d, s1=info_, abs_offset_days=off, n_hard=int(hm_d.sum()),
                           hard_water_share=round(float(y_v[hm_d].mean()), 3) if hm_d.any() else np.nan))
    print("\n--- 6. Sentinel-1 date offset and the share of water among hard pixels (descriptive, only a few dates) ---")
    print(pd.DataFrame(rows_o).sort_values("abs_offset_days").to_string(index=False))

    # ---- 7. how much information is there on hard pixels? (AUC, water vs land, hard points only) ----
    from sklearn.metrics import roc_auc_score
    hd = sh_i[hard].copy(); yh = y_v[hard]
    lvl = sh_i[sh_i.label == 1].groupby("date").elev.quantile(0.9)       # ORACLE: level of each date from the manual water labels
    hd["elev_rel_ORACLE"] = hd.elev - hd.date.map(lvl)
    single = ["mndwi", "ndwi", "ndvi", "vv_s", "VV", "VH", "vv_vh", "B3", "B8", "B11", "B12", "elev", "slope", "elev_rel_ORACLE"]
    rows_a = []
    for f_ in single:
        x_ = hd[f_].values.astype(float)
        if not np.isfinite(x_).all() or len(np.unique(yh)) < 2:
            continue
        a_ = roc_auc_score(yh, x_)
        rows_a.append(dict(score=f_, AUC=round(max(a_, 1 - a_), 3), water_is="higher" if a_ >= 0.5 else "lower"))
    for nm_ in ["Compact: mndwi + smoothed VV", CHOSEN_MODEL]:
        if nm_ in models and len(np.unique(yh)) == 2:
            pr_ = models[nm_][0].predict_proba(sh_v.loc[hard, models[nm_][1]])[:, 1]
            rows_a.append(dict(score="MODEL score: " + nm_, AUC=round(roc_auc_score(yh, pr_), 3), water_is="higher"))
    print(f"\n--- 7. Information on HARD pixels only ({int(hard.sum())} points, {int((yh == 1).sum())} water): AUC of one score, 0.5 = nothing, 0.7 = weak, 0.8+ = good ---")
    print(pd.DataFrame(rows_a).sort_values("AUC", ascending=False).to_string(index=False))
    print("   elev_rel_ORACLE uses the manual labels to get each date's water level: it shows what a good relative-elevation feature COULD give; it is not a usable model.")
    print("   Few water points (about 40): AUC values are uncertain by about +-0.05.")

    # ---- 8. terrain: elevation relative to the water level of the date ----
    rel = sh_v.elev_rel.values.astype(float)
    mn, vs = sh_v.mndwi.values.astype(float), sh_v.vv_s.values.astype(float)
    rule_w = (mn > 0.1) & (vs < S1_WATER_DB)
    rule_l = (mn < -0.1) & (vs > -12.0)
    hard_f = ~(rule_w | rule_l)                      # pixels where the two sensors do not agree (from the features)
    base_pred = rule_w.astype(float)
    print("\n" + "-" * 78)
    print(f"--- 8. Terrain. water level of a date = q{int(LEVEL_Q * 100)} of the elevation of its confident-water TRAIN rows; elev_rel = elev - level ---")
    rows_d = []
    for d in sorted(sh_v.date.unique()):
        m = (sh_v.date == d).values
        hw_d, hl_d = m & hard & (y_v == 1), m & hard & (y_v == 0)
        rows_d.append(dict(date=d, level_m=round(float(sh_v.loc[m, "water_level"].iloc[0]), 1),
                           hard_water_rel_median=round(float(np.median(rel[hw_d])), 1) if hw_d.any() else np.nan,
                           hard_land_rel_median=round(float(np.median(rel[hl_d])), 1) if hl_d.any() else np.nan,
                           hard_water_rel_le_5=round(float((rel[hw_d] <= 5).mean()), 2) if hw_d.any() else np.nan,
                           hard_land_rel_le_5=round(float((rel[hl_d] <= 5).mean()), 2) if hl_d.any() else np.nan))
    print("8a. Per date, hard pixels. 'rel_le_5' = share of the pixels that lie at most 5 m above the water level (terrain does not reject them).")
    print("    Good terrain signal: water close to 1, land close to 0.")
    print(pd.DataFrame(rows_d).to_string(index=False))

    rows_h = {"rule (reference)": eval_row(base_pred)}
    for tol_ in (0.0, 5.0, 10.0):
        ok_t = rel <= tol_
        rows_h[f"rule + terrain on hard pixels (elev <= level + {tol_:.0f} m)"] = eval_row(np.where(hard_f & ok_t, 1.0, base_pred))
        rows_h[f"rule + terrain + mndwi > 0 on hard pixels (level + {tol_:.0f} m)"] = eval_row(np.where(hard_f & ok_t & (mn > 0), 1.0, base_pred))
    cname = "RF: Compact: mndwi + smoothed VV"
    if cname in preds_v:
        rows_h["Compact model, water only if elev <= level + 5 m"] = eval_row(preds_v[cname] * (rel <= 5.0))
        rows_h["Compact model (reference)"] = eval_row(preds_v[cname])
    print("\n8b. Rules and a model with terrain (no training; the level comes from the TRAIN rows, not from manual labels):")
    print(pd.DataFrame(rows_h).T.to_string())
    print("    The tolerance (0, 5, 10 m) is shown, not tuned: choose it from physics (elevation error of the DEM), not from this table.")

    terr_a = ["mndwi", "vv_s", "elev_rel", "slope"]
    terr_b = cols_c + ["elev_rel", "slope"]
    rows_k = {}
    for fname, fc in [("Compact + terrain", terr_a), ("S2+S1 + terrain", terr_b)]:
        rows_k[f"{fname}: weak labels only"] = eval_row(lodo_variant(fc, 0.0))
        rows_k[f"{fname}: weak + manual (100 %)"] = eval_row(lodo_variant(fc, 1.0))
        rows_k[f"{fname}: manual labels only"] = eval_row(lodo_variant(fc, 0.0, use_weak=False))
    rows_k[best_rule_val] = eval_row(preds_v[best_rule_val])
    print("\n8c. Leave-one-date-out models with terrain features (each date predicted by a model that never saw it):")
    print(pd.DataFrame(rows_k).T.to_string())

    # ---- 8d. LOCAL water level: the k nearest confident-water TRAIN points of the same date ----
    K_LOC, Q_LOC = 15, 0.90          # fixed in advance, not tuned
    lvl_loc = np.full(len(sh_v), np.nan); rad_loc = np.full(len(sh_v), np.nan)
    for d in sh_v.date.unique():
        m = (sh_v.date == d).values
        trw = tr[(tr.date == d) & (tr.tier == "T2_water")]
        if len(trw) < K_LOC:
            continue
        dd_, ii_ = cKDTree(to_xy(trw, lat0)).query(to_xy(sh_v[m], lat0), k=K_LOC)
        lvl_loc[m] = np.quantile(trw.elev.values[ii_], Q_LOC, axis=1)
        rad_loc[m] = dd_.max(axis=1)
    rel_loc = sh_v.elev.values.astype(float) - lvl_loc
    rows_l = []
    for d in sorted(sh_v.date.unique()):
        m = (sh_v.date == d).values
        hw_d, hl_d = m & hard & (y_v == 1), m & hard & (y_v == 0)
        rows_l.append(dict(date=d, radius_for_15_points_m=round(float(np.nanmedian(rad_loc[m])), 0),
                           hard_water_rel_median=round(float(np.nanmedian(rel_loc[hw_d])), 1) if hw_d.any() else np.nan,
                           hard_land_rel_median=round(float(np.nanmedian(rel_loc[hl_d])), 1) if hl_d.any() else np.nan,
                           hard_water_le_0=round(float((rel_loc[hw_d] <= 0).mean()), 2) if hw_d.any() else np.nan,
                           hard_land_le_0=round(float((rel_loc[hl_d] <= 0).mean()), 2) if hl_d.any() else np.nan))
    print(f"\n8d. LOCAL level = q{int(Q_LOC * 100)} of the elevation of the {K_LOC} nearest confident-water TRAIN points of the same date.")
    print("    Compare with 8a: land should be accepted ('le_0' = at or below the local level) much less than before on the high-level dates.")
    print(pd.DataFrame(rows_l).to_string(index=False))

    # ---- 8e. terrain rules, global vs local level, and their paired difference to the rule ----
    preds_h = dict(preds_v)
    preds_h["rule (terrain-free reference)"] = base_pred
    rows_e = {"rule (terrain-free reference)": eval_row(base_pred)}
    for nm_, pr_ in [("global level", rel), ("local level", rel_loc)]:
        for tol_ in (0.0, 5.0):
            okx = np.nan_to_num(pr_, nan=1e9) <= tol_
            k1 = f"terrain on hard pixels | {nm_} | tol {tol_:.0f} m"
            k2 = f"terrain + mndwi > 0 on hard pixels | {nm_} | tol {tol_:.0f} m"
            preds_h[k1] = np.where(hard_f & okx, 1.0, base_pred)
            preds_h[k2] = np.where(hard_f & okx & (mn > 0), 1.0, base_pred)
            rows_e[k1] = eval_row(preds_h[k1]); rows_e[k2] = eval_row(preds_h[k2])
    print("\n8e. Terrain rules (no training). Rule = the two-sensor rule; every other row adds terrain on hard pixels only:")
    print(pd.DataFrame(rows_e).T.to_string())
    h1_table(sh_v, cells_v, W_v, y_v, preds_h, "rule (terrain-free reference)",
             [k for k in rows_e if k != "rule (terrain-free reference)"], "VAL terrain rules vs the rule")
    print("    Read the paired table, not the single F1 values. The tolerance is a physical choice (DEM error), not a tuning knob.")

# ======================================================================
# TEST  (closed until RUN_FINAL_TEST = True)
# ======================================================================
print("\n" + "=" * 78)
if not RUN_FINAL_TEST:
    print(f"TEST SET (dates {TEST_DATES}) NOT USED.")
    print("When the model is chosen: set CHOSEN_MODEL, then RUN_FINAL_TEST = True and run again.")
    print("The test is run ONCE. After it, the model and the settings must not be changed because of it.")
    print("=" * 78)
    raise SystemExit
test_counts = ev_test.groupby("date").size().reindex(TEST_DATES, fill_value=0)
too_few = [d for d in TEST_DATES if test_counts[d] < MIN_TEST_POINTS_PER_DATE]
if too_few:
    print(f"TEST STAYS CLOSED. Not every test date is labeled yet (fewer than {MIN_TEST_POINTS_PER_DATE} points): {too_few}.")
    print("Opening the test with dates missing would use up the test set. Label them, set RUN_FINAL_TEST = False,")
    print("and open the test only when all test dates are done.")
    print("=" * 78)
    raise SystemExit
print(f"TEST SET, dates {TEST_DATES}: FINAL numbers. Model {CHOSEN_MODEL}, rule chosen on validation: {best_rule_val}")
print("=" * 78)
ft = get_frame(ev_test)
if ft is None:
    print("No test labels yet. Label the test dates first."); raise SystemExit
sh_t, cells_t, W_t, y_t = ft
print(f"Shore-zone frame: {len(sh_t)} points in {len(cells_t)} cells (date x stratum), unsure labels excluded")
preds_t = get_preds(sh_t)
res_t = results_table(sh_t, cells_t, W_t, y_t, preds_t, "TEST (rows other than the chosen model are context, not a selection)")
h1_t = h1_table(sh_t, cells_t, W_t, y_t, preds_t, best_rule_val, ["RF: " + CHOSEN_MODEL], "TEST")
lodo(sh_t, cells_t, W_t, y_t, CHOSEN_MODEL, "TEST", res_t.set_index("method").loc["RF: " + CHOSEN_MODEL, "F1_water"])
unsure_case(ev_test, CHOSEN_MODEL, "TEST")
kappa_report(ev_test, "TEST")
res_t.drop(columns="F1").to_csv(os.path.join(RESULTS_DIR, f"test_results_{DATASET_VERSION}.csv"), index=False)
h1_t.to_csv(os.path.join(RESULTS_DIR, f"test_h1_{DATASET_VERSION}.csv"), index=False)
print("\nSaved to:", RESULTS_DIR)
print("NOTE: area and its CI need a prediction MAP (all pixels). That is the next step; this file tests the classifier on points.")