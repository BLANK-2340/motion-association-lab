# Motion Association Lab

An original, dependency-free CPU tracking workbench. Investigate when a motion
prior preserves identity and when merged observations make it worse. Includes
generated image sequences, a component detector, exact association, explicit
diagnostics, reproducible evidence and an interactive replay.

[Public workbench](https://blank-2340.github.io/motion-association-lab/)
is an interactive viewer of **saved CPU results**, not live inference. Select a
sequence, scrub/play frames, inspect both methods and toggle ground-truth outlines.
Source publication alone does not establish demo availability; deployment and
live interaction acceptance are recorded separately in the delivery report.

## Setup and demo

Python 3.10+; no packages, downloads, credentials or external data required.
Node 22+ runs optional viewer checks, with no npm dependencies.

```bash
python -W error tracking_lab.py --self-test
python tracking_lab.py
python -m http.server 8000
```

Open `http://localhost:8000/docs/`. The original CLI also supports
`python tracking_lab.py --html tracks.html`, creating a new saved-result viewer
without overwriting an existing file.

## Architecture and scientific question

Does constant-velocity prediction preserve identities better than last-position
association through crossings and short detection gaps?

1. Original 100 × 64 grayscale images contain two identical 5 × 5 squares.
2. Threshold at 0.5 and find 4-connected components. Observe their centroids.
3. Predict with the last observed velocity, or use last position as an ablation.
4. Exact bitmask dynamic programming first maximizes one-to-one matches inside
   an inclusive 10-pixel gate, then minimizes squared distance. Ties use indices.
5. Retain tracks for three missed frames. Emit only observed detections, not
   extrapolated positions. Reject scenes exceeding eight active tracks atomically.
6. Evaluation matches against ground truth inside a 6-pixel gate and counts
   misses, false positives and changes in each truth object's last matched ID.

Ground truth and future frames never enter the tracker. Matching is exponential
in this tiny bounded scene, not a scalable Hungarian implementation. Invisible
objects remain in ground truth and count as misses. Diagnostics are not official
MOTChallenge, IDF1 or HOTA metrics.

## Measured results and failure cases

Original three seeds yield identical diagnostics, not diverse motion evidence:

| Sequence | Matches, both variants | Last-position ID switches | Velocity ID switches |
|---|---:|---:|---:|
| Clean | 80/80 | 0 | 0 |
| Crossing / merged components | 78/80 | 2 | 2 |
| Three-frame occlusion | 77/80 | 2 | 0 |

A separate fixed **24-cell sensitivity sweep** gives 21 total switches for
last-position versus 24 for velocity. Velocity improves 3 cells, ties 13 and
worsens 8. This is a designed diagnostic, not a population estimate or universal
gain. See [protocol, evidence and limitations](RESULTS.md).

## Checks and exact evidence regeneration

```bash
python -W error tracking_lab.py --self-test
python -W error -m unittest discover -s tests -v
python -W error export_workbench.py --check
node --check docs/app.js
node --test tests/viewer.test.cjs
```

`python export_workbench.py` deliberately regenerates `docs/evidence.json`.
`--check` independently recomputes and compares every byte, binding the core and
exporter SHA-256 identities. CI runs these checks. Python tests cover an exhaustive
small assignment oracle, ties, lifecycle, atomic validation, known metric counts,
generated truth, export consistency and sweep coverage. Viewer tests reject
corrupted evidence and independently check conservation of emitted counts.

## Limitations, sources and license

- Original synthetic square sequences only, no real-video or learned detector.
- Same-color objects, component merging, abrupt motion and long gaps remain hard.
- The public replay shows the binary foreground mask, not grayscale background
  noise; labels and metrics come from saved Python computation.
- No GPU, runtime benchmark, state-of-the-art, statistical-significance or
  real-world generalization claim. Mobile browser validation is not yet recorded.
- All implementation, generated images and site assets are original. No external
  code/data/assets are incorporated. No paper reproduction is claimed. MIT license.

Read [persistent progress and next milestone](PROGRESS.md). The initial module
docstring records historical November 16 status; later progress supersedes it.
