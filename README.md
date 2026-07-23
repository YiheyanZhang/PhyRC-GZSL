# PhyRC-GZSL

Anonymous implementation and reproducibility package for **PhyRC-GZSL: Physics-Relational Prototypes and Risk-Calibrated Joint Decoding for Hyperspectral Generalized Zero-Shot Learning**.

PhyRC-GZSL performs strictly inductive hyperspectral generalized zero-shot learning. It transfers class-level physical relations into unseen visual prototypes, estimates seen/pseudo-unseen evidence with class-wise cross-fitting, and applies risk-calibrated joint decoding without using true unseen samples for training or model selection.

## Repository layout

```text
phyrc_gzsl/
├── configs/       Dataset and protocol configuration
├── data/processed Class-level attributes used by the experiments
├── models/        Spectral backbone definitions
├── phyrc/         Calibration, attention, and joint decoding
├── baseline/      Controlled baseline adapters and upstream licenses
├── checkpoints/   Paper-facing JSON results (weights excluded)
└── tests/         Unit and protocol checks
```

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Place each downloaded hyperspectral dataset under `phyrc_gzsl/data/raw/` using the paths declared in `phyrc_gzsl/configs/*.yaml`. Raw datasets and trained weights are intentionally not redistributed; see `phyrc_gzsl/data/README.md` for the exact layout.

## Usage

Train a seen-only backbone:

```bash
python -m phyrc_gzsl.train_spectral_mae --config phyrc_gzsl/configs/paviau_p1.yaml
```

Evaluate PhyRC-GZSL with a trained backbone:

```bash
python -m phyrc_gzsl.evaluate_phyrc \
  --config phyrc_gzsl/configs/paviau_p1.yaml \
  --attributes phyrc_gzsl/data/processed/PaviaU_structured_attributes.json \
  --unseen-classes 1 2 3 4 5 6 7 8 9 \
  --backbone-pattern phyrc_gzsl/checkpoints/paviau_p1_backbone_s{unseen}.pt \
  --output phyrc_gzsl/checkpoints/paviau_phyrc_single_unseen.json
```

Generate the fixed Indian Pines experiment manifest without training:

```bash
python -m phyrc_gzsl.run_indian_pines_submission
```

Run the source-only checks:

```bash
python -m pytest phyrc_gzsl/tests -q
python test_release_layout.py
```

Paper-facing aggregate JSON files are retained under `phyrc_gzsl/checkpoints/`; model weights are excluded. Baseline adapters and their upstream license files remain under `phyrc_gzsl/baseline/`.

## Anonymous-release notes

This snapshot intentionally contains no author names, affiliations, acknowledgements, Git history from the development workspace, raw datasets, or trained weights. Add citation and license metadata only when preparing the public post-review release.
