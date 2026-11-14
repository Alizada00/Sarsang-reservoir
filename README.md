# Sarsang Reservoir Water Mapping

Google Earth Engine and Python workflow for preparing multi-sensor water-mapping samples, collecting manual reference labels, and evaluating Random Forest models around the Sarsang Reservoir, Azerbaijan.

The workflow combines Sentinel-2 surface reflectance, Sentinel-1 VV/VH backscatter, JRC Global Surface Water masks, and SRTM elevation/terrain features. Weak-label training samples and manually labelled evaluation samples are separated using a frozen spatial block split.

## Repository structure

- `src/sarsang_dataset.js` — Earth Engine dataset construction and manual-label interface.
- `src/sarsang_model.py` — Random Forest training, validation, diagnostics, and gated test evaluation.
- `requirements.txt` — Python dependencies.

## Workflow

1. Optionally run `MODE = 'timeseries'` year by year to inspect water-area time series.
2. Use the frozen spatial block split configured through `CONFIG.BLOCKS_ASSET`. Do not recreate it when continuing the existing study.
3. Run `MODE = 'build'` to export `sarsang_dataset_v9.csv` to the configured Google Drive folder.
4. Run `MODE = 'label'` for each required date. Label points as water (1), land (0), or unsure (-1), then save console CSV files as `val_*.csv`.
5. Run the Python workflow with `RUN_FINAL_TEST = False` for model development and validation.
6. Open the final test only after model selection is complete and all required test dates have enough manual labels. Set `RUN_FINAL_TEST = True` for final evaluation.

## Version consistency

The Earth Engine script uses `CONFIG.VERSION = 'v9'`; the Python script uses `DATASET_VERSION = "v9"` and expects `sarsang_dataset_v9.csv`. Keep these values synchronized if the dataset schema or sampling design changes.

Avoid changing dates, thresholds, block roles, sampling counts, or feature definitions after manual labelling has started unless the study design is deliberately revised and dependent data are regenerated.

## Configuration and execution

The public copy intentionally leaves project-specific Earth Engine asset IDs blank. Set your own accessible asset paths in the relevant `CONFIG` fields before running asset-dependent modes.

The Python script does not mount Google Drive or authenticate to a personal account automatically. Set `SARSANG_DIR` to the directory containing the dataset and manual-label files:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SARSANG_DIR="/path/to/sarsang_phd"
python src/sarsang_model.py
```

On Windows PowerShell:

```powershell
$env:SARSANG_DIR = "C:\\path\\to\\sarsang_phd"
python src/sarsang_model.py
```

Expected inputs: `sarsang_dataset_v9.csv`, `val_<date>_<annotator>.csv`, and optionally `label_corrections.csv`. Manual-label files should contain `annotator`, `date`, `lon`, `lat`, and `label`.

## Evaluation design and interpretation

- Models are fitted on weakly labelled points from training blocks only.
- Manual labels for validation dates are used for model selection, feature-set comparison, rule comparison, and diagnostics.
- Manual labels for test dates remain closed during model selection. The script checks that each test date meets the configured minimum label count.
- Validation and test are separated by date; manual labels are spatially separated from training blocks by the configured buffer.
- Accuracy and water F1 are calculated for the sampled shore-zone evaluation frame using area-weighted, stratified estimators. The current Python script evaluates labelled points; it does not produce a full-reservoir prediction map or map-wide area estimate.
- Compact feature sets include `mndwi` and smoothed `VV`, inputs to the weak-label rule. Interpret these models as learned/smoothed versions of the rule, not fully independent detectors.
- `2020-12-27` is assigned to validation because its test results had previously been viewed. It is not an unseen test date.

## Data and credential safety

This repository excludes private datasets, manual-label CSV files, exported results, Drive mount commands, Earth Engine authentication tokens, browser-session data, service-account keys, and project-specific Earth Engine asset IDs. Keep credentials and private research data outside the repository.

## Limitations

The Earth Engine script has not been verified by a complete execution in the Earth Engine Code Editor. Local syntax checks do not validate server-side operations, data availability, export tasks, or asset permissions. Dataset CSVs, manual labels, correction files, trained models, and result tables are not included. The later hold-out dates are not automatically evaluated by the current Python script. No numerical model-performance result is claimed here.

## Citation

If this code supports a publication, cite the associated manuscript or project record once available. Describe the sampling design, manual-label protocol, weak-label construction, and feature-set limitations when reporting results. Follow the AI-assistance disclosure requirements of the relevant journal or institution.
