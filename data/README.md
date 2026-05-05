# Geological inputs

Keep downloaded source logs under `raw/` and normalized files under `processed/`.
These directories and model checkpoints are excluded from version control.

The audit loader accepts a borehole CSV and hypothesis JSONL. Required CSV fields
are `world_id`, `split`, `borehole_id`, `role`, `x`, `depth_1`, and `depth_2`.
Splits are `train`, `calibration`, or `test`. Observation roles are `visible` or
`verification`. Complete geological regions stay in one split, and visible and
verification boreholes must have distinct IDs and coordinates. A single role
cannot repeat a transect coordinate under another ID. Exact whole-world
observation copies are rejected across splits independently of IDs and row
order. Source region/site assignments remain necessary to establish geographic
independence. Retain the site origin alongside relative transect coordinates.
Coordinates and boundary depths use metres after normalization.

Hypothesis records contain `case_id`, `world_id`, and four `parameters` describing
two intercepts and two slopes. Source interval logs require an explicit column
mapping and geological units defining the two interpreted boundaries. Preserve
the release title, source-record URL, file checksum, unit conversion, and boundary
interpretation in the input provenance manifest. USGS releases differ in schema.
Select the source release before configuring the importer.

## Acquiring interval logs

Select a public USGS release or ScienceBase record with drill-core intervals.
Download its CSV export and retain the dataset title, source-record URL and
release metadata. Raw field names differ between releases. Edit
`configs/interval_columns.example.json` to map the selected file's columns.

The interval importer needs region/world ID, borehole ID, transect coordinate,
interval top/bottom depths and lithology. Choose the two geological units whose
first tops define the modeled boundaries. It checks interval ordering, converts
metres/feet, and writes a SHA-256 provenance record beside the normalized CSV.

```bash
python -m strataaudit.usgs_import --intervals data/raw/core.csv --columns configs/interval_columns.example.json --source-url "$SOURCE_RECORD_URL" --dataset-title "$DATASET_TITLE" --middle-unit shale --lower-unit basement --output-dir data/processed/core
```

Set the URL/title to the selected release and replace the unit names with the
actual geological interpretation. At least three regions and four boreholes
per region are needed: two visible and two withheld boreholes at minimum.
Regions receive disjoint training/calibration/test assignments. Supply a
hypothesis JSONL for those normalized worlds separately.

## Hypothesis records

```json
{"case_id": "region-1-h1", "world_id": "region-1", "parameters": [6.0, 16.0, 0.1, -0.1]}
```

This is a schema example. Parameters are `[depth_1, depth_2, slope_1, slope_2]`
for `depth(x)=intercept+slope*x`. Use a consistent coordinate origin between
observations and hypotheses. The audit uses a transect model with two planar boundaries.

## Small local inputs

```bash
python scripts/generate_sample_data.py --worlds 30 --output-dir outputs/fixtures
python -m strataaudit.gempy_worlds --worlds 15 --output-dir outputs/gempy
```

The first command creates analytic planar worlds. The second requires the
GemPy extra and performs interpolation/custom-grid sampling. Both write
explicit synthetic provenance and normalized inputs. GemPy also retains its
surface-point/orientation CSVs and boundary sampling resolution.
Its five visible sites and sixteen withheld sites per world use disjoint grids.

Keep source data, exported adapters and model weights in their configured directories.
