# Fixed protocol and measured results

Recorded November 17, 2026. Question: when does a last-observed velocity help or
hurt identity continuity relative to position-only association?

## Original reference

Seeds 101, 202 and 303; 40 frames and two ground-truth objects per frame. Three
scenarios: clean (14-pixel vertical separation), crossing (2-pixel separation),
and crossing with object 1 removed from observed pixels on frames 18–20. Speeds
are 2 pixels/frame in opposite directions. Seed changes background noise below
the detector threshold and vertical translation only. All three seeds yield the
same metrics; they are not independent, diverse motion/noise evidence.

Each seed gives clean 80/80 matches, crossing 78/80, occlusion 77/80 for both
variants. Switch counts are respectively 0/0, 2/2 and 2/0 (last-position/velocity).
There are no false-positive emitted observations in these original cases.

## Controlled sweep, fixed before its execution

Cartesian product of speed {1,2}, vertical separation {0,2,8}, disappearance
length {0,1,3,5}: **24 cells**, not repeated random trials. Every cell has 40
frames and 80 truth observations. For speed s, centers are (10+s*t,26) and
(88-s*t,26+separation). Object 1 is absent from the raster for
18 <= t < 18+gap. No random background; all pixels are 0 or 0.9.

Keep the original detector threshold 0.5, association gate 10, memory 3 and
evaluation gate 6. No threshold selection on these results. Total identity
switches: **21 last-position, 24 velocity**. Velocity improves 3 cells, equals
13 and worsens 8. All eight worse cells have speed 1 and separation 0 or 2,
across all four gap lengths. Each cell's complete counts, recall and mean
centroid error are in `docs/evidence.json` under `sweep`.

Interpretation: the velocity prior can bridge short missed observations, but a
merged component changes observed centroids and therefore estimated velocity.
More elaborate motion prediction alone does not solve observation ambiguity.
Speed changes also change the relative timing of the fixed gap and crossing;
this is not an isolated causal estimate of speed. Coincident truth centers have
index-based evaluation ties. Switch counts depend on this explicitly fixed rule.

## Metric definitions

Framewise maximum-cardinality/minimum-squared-distance matching within 6 pixels.
Misses = truth count minus matches; false positives = emitted count minus matches.
Recall = matches / truth count. Mean centroid error is Euclidean distance over
matched pairs. A switch occurs when a truth object's matched track ID differs
from its last match, even across missed frames. No official benchmark equivalence
is claimed. Hidden objects still count as truth, and unobserved predictions are
not emitted or counted as detections.

## Reproduce and inspect

`python export_workbench.py --check` recomputes all evidence and demands identical
canonical JSON, including core/exporter SHA-256. Each original case records all
40 frames' foreground coordinates, ground truth, hidden IDs, component centroids
and both variants' emitted tracks. Metrics cover whole sequences, not selected
frames. The sweep stores cell parameters and full metrics. The hashes bind
content but are not authenticated provenance. No external data or assets.

Neither the original seeds nor the sweep establish statistical significance,
generalization to real scenes, neural-model quality or competitive performance.
Real-video data, learned detections, latency and GPU validation remain untested.
