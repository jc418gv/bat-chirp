# Upstream Projects

This repository is a local workflow layer built around upstream open-source projects. Credit for those detector, hardware, and research foundations belongs to their original authors and maintainers.

## Primary Upstream Dependencies

- BatDetect2: https://github.com/macaodha/batdetect2
  This pipeline depends on BatDetect2 for bat detection inference and associated model outputs.
- AudioMoth and Open Acoustic Devices: https://www.openacousticdevices.info/
  This workflow is designed around AudioMoth-style recordings and filenames produced by that ecosystem.

## Licensing Boundary

This repository does not claim ownership of upstream code, models, documentation, or branding.

Nothing in this repository changes, extends, replaces, or relicenses those upstream projects. If you install, bundle, copy, or redistribute them, you must follow their original licenses, terms, and citation guidance.

Any personal notes, local reference clones, or research comparisons used while developing this workflow remain separate from the licensing of this repository and from the licensing of the upstream projects themselves.

## Implementation Boundary

The BatDetect2 project remains responsible for the detector itself, feature extraction, model inference, and its native JSON output format.

This repository adds local workflow code around that upstream tool:

- command-line orchestration that invokes an external BatDetect2 installation
- AudioMoth-specific filename parsing and overnight session selection
- CSV summaries built from BatDetect2 JSON output files
- review clip export, spectrogram rendering, and HTML browsing pages
- local post-processing for review convenience, such as candidate train expansion

This repository is therefore a wrapper and post-processing layer around BatDetect2 outputs, not a vendored copy of the BatDetect2 source tree.

## Model Use

The upstream BatDetect2 detector and classifier used by this workflow are not treated here as trusted North American species outputs. Species-level predictions remain exploratory until validated against local review data.