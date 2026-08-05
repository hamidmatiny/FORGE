# Running ingest against real nuScenes-mini

`forge ingest` and its tests run entirely against a synthetic fixture
(`tests/fixtures/nuscenes_mini_synthetic/`) that mirrors the real
nuScenes-devkit JSON layout. Nothing here downloads or commits the real
dataset — it's ~4 GB and licensed for **non-commercial use only**.

## 1. Get the data

Register and download `v1.0-mini` from https://www.nuscenes.org/download,
then extract it so you have:

```
<your-path>/nuscenes-mini/
├── v1.0-mini/
│   ├── scene.json
│   ├── sample.json
│   ├── sample_data.json
│   ├── sensor.json
│   ├── calibrated_sensor.json
│   ├── ego_pose.json
│   └── ...
├── samples/
└── sweeps/
```

## 2. Point FORGE at it

```bash
uv run forge ingest --input-dir <your-path>/nuscenes-mini --local
```

Or via the DVC pipeline (tracks the resulting lake for versioning):

```bash
# edit params.yaml: set ingest.input_dir to <your-path>/nuscenes-mini
dvc repro ingest
dvc push   # to the local remote configured in .dvc/config
```

`data/lake/` is DVC-tracked, not git-tracked — the `dvc.yaml` `ingest` stage
is the reproducible record of how it was produced, not the data itself.

## 3. Verify

```bash
uv run python -c "
from forge.schemas import FramesTable
frames = FramesTable.read_parquet('data/lake/frames.parquet')
print(f'{len(frames)} frames, {len({f.scene_id for f in frames})} scenes')
"
```

Ground-truth annotations in nuScenes-mini are used only for evaluation
(Phase 7) — never as pipeline input, per the dataset's non-commercial license.
