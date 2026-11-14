# Code review notes

## Files

- `src/sarsang_dataset.js` — Earth Engine dataset construction and manual-label interface; version v9.
- `src/sarsang_model.py` — Random Forest training, manual-label matching, validation, diagnostics, and gated test evaluation; version v16, configured for dataset v9.

## Static checks

- Python source parses successfully.
- JavaScript source passes a local `node --check` syntax check.
- Dataset version is synchronized (`v9` in both scripts).
- The Python workflow fits models from `set = train` and evaluates against manual labels from `set = label`.
- Validation and test date lists are disjoint in the current configuration.

These checks do not establish successful Earth Engine execution or validate numerical results on the actual dataset and label files.

## Verification before interpreting results

1. Run the required Earth Engine modes and confirm exports complete.
2. Confirm the frozen block and manual-label assets correspond to the current study design.
3. Check the exported CSV schema against the Python script.
4. Run the Python workflow on the actual dataset and labels; inspect row counts, unmatched labels, missing dates, and diagnostics.
5. Keep the final test closed until model selection is complete. Record the final test run and preserve the exact source/configuration used.
6. Review warning suppression; suppressing all warnings can hide numerical or data-quality signals.
7. Configure your own Earth Engine asset IDs in a private working copy.

## Methodological caveats

- `2020-12-27` is a validation date, not an unseen test date, because its test result had previously been viewed.
- Compact feature sets include inputs used by the weak-label rule. Their predictions should not be presented as independent of that rule.
- The current Python script evaluates sampled points in the shore-zone frame. It does not produce a full-scene prediction map or map-wide area estimate.
- Hold-out dates for the later independent round are declared in Earth Engine configuration but are not processed by the current Python evaluation script.
