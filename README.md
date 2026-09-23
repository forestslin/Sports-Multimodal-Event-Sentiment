# Reproducibility package for PeerJ manuscript CS-142751

Manuscript: **Evaluating time-bounded visual and automatic speech recognition commentary fusion for soccer action spotting**.

This package reproduces the revised SoccerNet-v2 action-spotting experiment. It uses public SoccerNet-v2 point-event labels and ResNet PCA-512 descriptors together with timestamped Whisper-v1 English-normalized commentary from SoccerNet-Echoes. Source broadcast videos are not redistributed.

The SoccerNet-v2 labels and visual descriptors are third-party resources distributed by the [SoccerNet project](https://www.soccer-net.org/) under its access conditions. The third-party [SoccerNet-Echoes dataset](https://huggingface.co/datasets/SoccerNet/SN-echoes) is shared by the SoccerNet team, curated by SimulaMet under the AI-Storyteller project, and lists dataset DOI [10.57967/hf/2539](https://doi.org/10.57967/hf/2539). This study used `whisper_v1_en/1.0.0/soccer_net_echoes_hf_dataset-train.arrow` from that dataset. The code and derived analysis outputs in this repository were generated for the present study; they do not represent a re-release of the third-party source materials. The frozen code snapshot and author-generated derived results are archived at [Zenodo, version DOI 10.5281/zenodo.22917320](https://doi.org/10.5281/zenodo.22917320). The archive contains repository commit `805f1b538d0b8fc9de539695b1265b2d51c86517`, generated prediction exports, checkpoints, and a checksum manifest; it excludes third-party event-label files, visual feature archives, videos, and transcript Arrow files.

## Analysis scope

- Task: 17-class point-event action spotting.
- Prediction unit: the right edge of one completed two-second observation bin.
- Inputs: current and preceding 10 seconds of visual descriptors; ASR segments ending no later than the prediction time within a 20-second lookback.
- Split: intersection of public modalities while preserving official SoccerNet assignments (176 train games; 54 validation games).
- Primary metrics: class-macro event mAP within symmetric ±5-, ±10-, and ±20-second matching radii and matched-event timestamp MAE within ±10 seconds. Official SoccerNet metrics are reported separately.
- Runs: seven primary model configurations × five prespecified random seeds × 20 fixed epochs, plus text and post-processing controls.

## Software

Formal runs used Python 3.13 on CPU with the versions in `requirements.txt`. A CUDA 13.0 build was verified on an RTX 3060 Laptop GPU, but it was slower for these small networks because loading and evaluation dominated; GPU and CPU outputs were not mixed.

## Files

- `src/tgre_experiment.py`: data construction, models, training, NMS, and event-level evaluation.
- `src/run_experiments.py`: repeated-seed experiment driver.
- `src/summarize_results.py`: mean, SD, descriptive confidence intervals, and paired seed differences.
- `src/official_crosscheck.py`: prediction export and independent evaluation with the official SoccerNet package.
- `src/robustness_and_timing.py`: missing-text sensitivity and CPU inference timing.
- `src/game_bootstrap.py`: paired 1,000-replicate validation-game cluster bootstrap.
- `src/game_bootstrap_sensitivity.py`: empty-class sensitivity analysis for the paired game bootstrap.
- `src/nms_sensitivity.py`: 4-, 8-, and 12-second NMS sensitivity.
- `src/audit_selection.py` and `src/audit_text_coverage.py`: intersection and ASR-coverage audits.
- `src/finalize_revision_evidence.py`: manuscript-facing synthesis of frozen outputs.
- `src/make_figures.py`: submission figures and figure source-data tables.
- `artifacts/modality_coverage.json`: exact retained game identifiers and availability counts.
- `artifacts/experiment_results.json`: final seed-level records.
- `artifacts/final_revision_evidence.json`: final aggregate statistics and controls.
- `artifacts/game_cluster_bootstrap_sensitivity.json`: fixed-17-class and available-class bootstrap intervals.
- `artifacts/robustness_and_timing.json`: missing-text and timing measurements.
- `artifacts/official_crosscheck`: representative late-fusion and visual results from the official SoccerNet evaluator.
- `figures/source_data`: CSV data used to draw quantitative figures.

## Data placement

After obtaining SoccerNet access, place `Labels-v2.json` under:

```text
data/SN-Labels/<league>/<season>/<game>/Labels-v2.json
```

Place the two half-level descriptors in an adjacent `f` directory (one directory above this repository):

```text
../f/<league>/<season>/<game>/1_ResNET_TF2_PCA512.npy
../f/<league>/<season>/<game>/2_ResNET_TF2_PCA512.npy
```

Place the SoccerNet-Echoes Arrow file at:

```text
data/SN-echoes/whisper_v1_en/1.0.0/soccer_net_echoes_hf_dataset-train.arrow
```

The scripts under `src/download_*.py` record the data retrieval logic and versions. Access to SoccerNet resources remains subject to the dataset terms.

## Reproduce the final analysis

From the repository root, create an environment and install the recorded dependencies:

```text
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Then run the analysis from the repository root:

```text
.venv/Scripts/python.exe src/run_experiments.py --models visual text late rtgre rtgre_flow tcn_visual tcn_fusion --seeds 2026 2027 2028 2029 2030 --epochs 20 --device cpu
.venv/Scripts/python.exe src/summarize_results.py
.venv/Scripts/python.exe src/audit_selection.py
.venv/Scripts/python.exe src/audit_text_coverage.py
.venv/Scripts/python.exe src/game_bootstrap.py
.venv/Scripts/python.exe src/game_bootstrap_sensitivity.py
.venv/Scripts/python.exe src/nms_sensitivity.py
.venv/Scripts/python.exe src/robustness_and_timing.py
.venv/Scripts/python.exe src/official_crosscheck.py --model late --seed 2028 --epochs 20
.venv/Scripts/python.exe src/finalize_revision_evidence.py
.venv/Scripts/python.exe src/make_figures.py
```

The representative official-evaluator cross-check does not replace the five-seed primary analysis.

## Statistical interpretation

Values are mean±SD across five independently initialized training runs. Five-seed t intervals describe training-initialization uncertainty. A paired game-cluster bootstrap describes uncertainty across the 54 validation games conditional on the trained checkpoints. No null-hypothesis significance label is attached. Published full-split results are not treated as directly comparable to this availability intersection.
