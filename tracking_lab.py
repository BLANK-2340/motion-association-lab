"""Motion Association Lab, original dependency-free CPU research foundation.

Repository: https://github.com/BLANK-2340/motion-association-lab
Created public November 16, 2026 under BLANK-2340. Second new November project.

Run (Python 3.10+):
    python -W error tracking_lab.py --self-test
    python tracking_lab.py
    python tracking_lab.py --html tracks.html

Question: does constant-velocity prediction preserve identity better than
last-position association through crossings and short detection gaps?

Architecture: original raster sequences -> threshold/4-connected components ->
centroids -> exact maximum-cardinality, minimum-distance gated assignment ->
track lifecycle. Neither truth identities nor future frames enter the tracker.
Only current observed tracks are emitted, never extrapolated detections.
Assignment uses bitmask dynamic programming, bounded to eight objects. This is
an educational small-scene solver, not a scalable Hungarian implementation.

Protocol: three fixed seeds (101, 202, 303), 40 frames, two identical 5x5 squares
per frame; clean separated paths, crossing with merged components, and crossing
with a three-frame occlusion of object 1. Detector threshold 0.5, gate 10 pixels,
memory 3 missed frames, evaluation gate 6 pixels, fixed before evaluation.
Truth remains present during occlusion, so invisible objects count as misses.
Identity switches compare each ground-truth object's last matched track ID,
including across gaps. These are explicitly defined toy-scene diagnostics,
not official MOTChallenge/IDF1/HOTA benchmark metrics. No training or tuning.

HTML is an interactive frame viewer of saved CPU results, NOT live inference.
Measured November 16: every seed produced identical metrics. Clean: 80/80
matches, no switches for either method. Merged crossing: 78/80 matches and two
switches for both. Occlusion: 77/80 matches, two switches for last-position and
zero for velocity. Seeds change background noise and vertical translation only;
they do not constitute a diverse motion/noise study or establish significance.
Recorded validation: seven initial unit groups passed. Final checks also require
capacity/atomicity, end-to-end regression, subprocess repeatability, exclusive
HTML output, committed-archive rerun and exact remote source verification.
No public demo, README, separate license file or CI yet; not showcase-ready.
Next session: public browser demonstration, architecture/results/limitations,
license, CI, multi-seed evidence and live interaction acceptance. Do not rebuild
this foundation. Other original portfolio repositories remain unchanged.

All code and synthetic images are original. No external data/code/assets.
MIT License, Copyright (c) 2026 Armaan Singh
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to
do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import unittest


def points(values):
    result = tuple(tuple(p) for p in values)
    if len(result) > 8 or any(len(p) != 2 or not all(math.isfinite(v) for v in p)
                              for p in result):
        raise ValueError("at most eight finite 2D points required")
    return result


def assign(left, right, gate):
    """Lexicographic optimum: maximize matches, minimize squared distance, indices.

    Gate is inclusive. Exact ties prefer smaller detection indices. No matching
    pair is forced when outside the gate; returned pairs are one-to-one.
    """
    left, right = points(left), points(right)
    if not math.isfinite(gate) or gate <= 0:
        raise ValueError("positive finite gate required")
    costs = [[sum((a-b)**2 for a, b in zip(p, q)) for q in right] for p in left]

    @lru_cache(None)
    def solve(i, mask):
        if i == len(left):
            return (0, 0.0, ())
        count, cost, pairs = solve(i+1, mask)
        options = [(count, cost, pairs)]
        for j, distance in enumerate(costs[i]):
            if not mask & (1 << j) and distance <= gate*gate:
                count, cost, pairs = solve(i+1, mask | (1 << j))
                options.append((count-1, cost+distance, ((i, j),)+pairs))
        return min(options)

    return solve(0, 0)[2]


def detect(image, threshold=.5):
    """Centroids of all 4-connected foreground components, sorted spatially."""
    if (not image or not image[0] or not math.isfinite(threshold)
            or not 0 <= threshold <= 1):
        raise ValueError("nonempty image and valid threshold required")
    width, height = len(image[0]), len(image)
    if width > 256 or height > 256 or any(len(row) != width for row in image):
        raise ValueError("rectangular image at most 256x256 required")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for row in image for v in row):
        raise ValueError("pixels must be finite in [0,1]")
    seen, centers = set(), []
    for y in range(height):
        for x in range(width):
            if (x, y) in seen or image[y][x] <= threshold:
                continue
            stack, component = [(x, y)], []
            seen.add((x, y))
            while stack:
                px, py = stack.pop()
                component.append((px, py))
                for qx, qy in ((px-1, py), (px+1, py), (px, py-1), (px, py+1)):
                    if (0 <= qx < width and 0 <= qy < height and
                            (qx, qy) not in seen and image[qy][qx] > threshold):
                        seen.add((qx, qy))
                        stack.append((qx, qy))
            centers.append(tuple(sum(p[k] for p in component)/len(component) for k in (0, 1)))
    return sorted(centers)


@dataclass(frozen=True)
class Track:
    identity: int
    position: tuple
    velocity: tuple
    observed_at: int


class Tracker:
    def __init__(self, motion=True, gate=10., max_missed=3):
        if type(motion) is not bool or not math.isfinite(gate) or gate <= 0:
            raise ValueError("boolean motion and positive finite gate required")
        if type(max_missed) is not int or not 0 <= max_missed <= 20:
            raise ValueError("integer memory in [0,20] required")
        self.motion, self.gate, self.max_missed = motion, gate, max_missed
        self.tracks, self.frame, self.next_id = (), -1, 1

    def update(self, detections):
        detections = points(detections)  # Validate before any state mutation.
        frame = self.frame+1
        active = [t for t in self.tracks if frame-t.observed_at <= self.max_missed+1]
        predicted = [tuple(p+(frame-t.observed_at)*v if self.motion else p
                           for p, v in zip(t.position, t.velocity)) for t in active]
        pairs = assign(predicted, detections, self.gate)
        next_tracks, emitted, matched = [], [], set()
        by_track = dict(pairs)
        for i, old in enumerate(active):
            if i in by_track:
                j = by_track[i]
                position = detections[j]
                velocity = tuple((p-q)/(frame-old.observed_at)
                                 for p, q in zip(position, old.position))
                track = Track(old.identity, position, velocity, frame)
                next_tracks.append(track)
                emitted.append(track)
                matched.add(j)
            elif frame-old.observed_at <= self.max_missed:
                next_tracks.append(old)
        next_id = self.next_id
        for j, position in enumerate(detections):
            if j not in matched:
                track = Track(next_id, position, (0., 0.), frame)
                next_id += 1
                next_tracks.append(track)
                emitted.append(track)
        if len(next_tracks) > 8:
            raise ValueError("scene exceeds eight active tracks including retained misses")
        self.next_id = next_id
        self.tracks, self.frame = tuple(next_tracks), frame
        return tuple(sorted(emitted, key=lambda t: t.identity))


def sequence(seed=101, scenario="crossing"):
    if type(seed) is not int or scenario not in ("clean", "crossing", "occlusion"):
        raise ValueError("integer seed and known scenario required")
    rng = random.Random(seed)
    y0 = 26+rng.randrange(3)
    frames = []
    for frame in range(40):
        truth = [(1, (10+2*frame, y0)),
                 (2, (88-2*frame, y0+(14 if scenario == "clean" else 2)))]
        image = [[rng.random()*.1 for _ in range(100)] for _ in range(64)]
        hidden = []
        for identity, (x, y) in truth:
            if scenario == "occlusion" and identity == 1 and 18 <= frame <= 20:
                hidden.append(identity)
                continue
            for py in range(y-2, y+3):
                for px in range(x-2, x+3):
                    image[py][px] = .9
        frames.append({"image": image, "truth": truth, "hidden": hidden})
    return frames


def evaluate(truth_frames, predicted_frames, gate=6.):
    """Framewise spatial matching; persistent truth->track identity diagnostics."""
    if len(truth_frames) != len(predicted_frames):
        raise ValueError("frame counts must agree")
    last, counts, distances = {}, {"truth": 0, "matches": 0, "misses": 0,
                                   "false_positives": 0, "identity_switches": 0}, []
    for truth, predictions in zip(truth_frames, predicted_frames):
        for observations in (truth, predictions):
            if len({identity for identity, _ in observations}) != len(observations):
                raise ValueError("duplicate identity within frame")
        pairs = assign([p for _, p in truth], [p for _, p in predictions], gate)
        counts["truth"] += len(truth)
        counts["matches"] += len(pairs)
        counts["misses"] += len(truth)-len(pairs)
        counts["false_positives"] += len(predictions)-len(pairs)
        for i, j in pairs:
            identity, p = truth[i]
            track, q = predictions[j]
            counts["identity_switches"] += int(identity in last and last[identity] != track)
            last[identity] = track
            distances.append(math.dist(p, q))
    return {**counts, "recall": counts["matches"]/counts["truth"] if counts["truth"] else None,
            "mean_centroid_error": sum(distances)/len(distances) if distances else None}


def experiment():
    results = []
    for seed in (101, 202, 303):
        for scenario in ("clean", "crossing", "occlusion"):
            frames = sequence(seed, scenario)
            detections = [detect(f["image"]) for f in frames]
            truth = [f["truth"] for f in frames]
            for motion in (False, True):
                tracker = Tracker(motion=motion)
                predictions = [[(t.identity, t.position) for t in tracker.update(d)] for d in detections]
                results.append({"seed": seed, "scenario": scenario,
                                "variant": "velocity" if motion else "last_position",
                                "metrics": evaluate(truth, predictions)})
    return {"schema_version": 1, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "original synthetic images; not official MOT benchmark metrics", "results": results}


def html_view(path):
    """Write exclusively, never overwrite; viewer contains CPU-computed snapshots."""
    frames = sequence(101, "occlusion")
    trackers = [Tracker(motion=False), Tracker(motion=True)]
    snapshots = []
    for f in frames:
        detected = detect(f["image"])
        snapshots.append({"truth": f["truth"], "hidden": f["hidden"], "detections": detected,
                          "tracks": [[(t.identity, t.position) for t in tracker.update(detected)]
                                     for tracker in trackers]})
    payload = json.dumps(snapshots, allow_nan=False)
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motion Association Lab | saved CPU sequence</title>
<style>body{background:#101a22;color:#e6f0f8;font:17px system-ui;max-width:950px;margin:32px auto;padding:18px}svg{width:100%;background:#192b38;border:1px solid #638296}input{width:100%}text{font:3px monospace}a{color:#9decff}</style>
<h1>Where does identity go?</h1><p>Interactive viewer of <b>saved CPU results</b>, not live tracking inference.
Original two-square crossing with a three-frame occlusion. Outline: ground truth; filled dots: observed track IDs.</p>
<label>Frame <output id="number">0</output><input id="frame" type="range" min="0" max="39" value="0"></label>
<h2>Last-position association</h2><svg id="last" viewBox="0 0 100 64" role="img" aria-label="Last-position tracks"></svg>
<h2>Constant-velocity association</h2><svg id="motion" viewBox="0 0 100 64" role="img" aria-label="Velocity tracks"></svg>
<p id="detail"></p><p>Identical appearances and merged detections remain ambiguous. This is a bounded synthetic diagnostic, not a real-video model or official tracking benchmark.</p>
<script>const frames=PAYLOAD;const slider=document.querySelector('#frame');
function draw(){const k=+slider.value,f=frames[k];document.querySelector('#number').textContent=k;
['last','motion'].forEach((id,v)=>{document.getElementById(id).innerHTML=
f.truth.map(([i,[x,y]])=>`<rect x="${x-2.5}" y="${y-2.5}" width="5" height="5" fill="none" stroke="white"/><text x="${x-3}" y="${y-4}" fill="white">GT${i}</text>`).join('')+
f.tracks[v].map(([i,[x,y]])=>`<circle cx="${x}" cy="${y}" r="1.3" fill="${['#7ae6cc','#ffbf72','#a9a3ff','#ff849a'][i%4]}"/><text x="${x+2}" y="${y+4}" fill="#f5edc6">T${i}</text>`).join('');});
document.querySelector('#detail').textContent=`Frame ${k}: ${f.detections.length} components; hidden truth IDs: ${f.hidden.join(', ')||'none'}`;}
slider.addEventListener('input',draw);draw();</script></html>'''.replace("PAYLOAD", payload)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(html)


class Tests(unittest.TestCase):
    def test_assignment_against_exhaustive_oracle(self):
        rng = random.Random(42)
        for n, m in itertools.product(range(4), repeat=2):
            for _ in range(8):
                a = [(rng.randrange(10), rng.randrange(10)) for _ in range(n)]
                b = [(rng.randrange(10), rng.randrange(10)) for _ in range(m)]
                candidates = []
                for choices in itertools.product(range(-1, m), repeat=n):
                    used = [j for j in choices if j >= 0]
                    if len(used) != len(set(used)):
                        continue
                    pairs = tuple((i, j) for i, j in enumerate(choices) if j >= 0)
                    squared = [math.dist(a[i], b[j])**2 for i, j in pairs]
                    if all(d <= 25.00000001 for d in squared):
                        candidates.append((-len(pairs), sum(sum((p-q)**2 for p, q in zip(a[i], b[j]))
                                                          for i, j in pairs), pairs))
                self.assertEqual(assign(a, b, 5), min(candidates)[2])

    def test_gating_ties_and_cardinality(self):
        self.assertEqual(assign([(0, 0), (0, 0)], [(1, 0), (1, 0)], 1), ((0, 0), (1, 1)))
        self.assertEqual(assign([(0, 0)], [(2, 0)], 1), ())
        self.assertEqual(assign([(0, 0), (2, 0)], [(1, 0), (-2, 0)], 2), ((0, 1), (1, 0)))

    def test_component_centroids_connectivity_and_threshold(self):
        image = [[1, 1, 0], [0, 0, 1], [.5, 0, 1]]
        self.assertEqual(detect(image), [(.5, 0.), (2., 1.5)])
        self.assertEqual(image[0], [1, 1, 0])

    def test_velocity_gap_and_expiration(self):
        tracker = Tracker(max_missed=2, gate=3)
        first = tracker.update([(0, 0)])[0]
        tracker.update([(2, 0)])
        self.assertEqual(tracker.update([]), ())
        self.assertEqual(tracker.update([]), ())
        track = tracker.update([(8, 0)])[0]
        self.assertEqual(track.identity, first.identity)
        self.assertEqual(track.velocity, (2, 0))
        for _ in range(3):
            tracker.update([])
        self.assertNotEqual(tracker.update([(16, 0)])[0].identity, first.identity)
        self.assertEqual(first.position, (0, 0))

    def test_metrics_known_counts_including_identity_across_gap(self):
        truth = [[(1, (0, 0))], [(1, (1, 0))], [(1, (2, 0))]]
        pred = [[(7, (0, 0)), (8, (50, 50))], [], [(9, (2, 0))]]
        self.assertEqual(evaluate(truth, pred), {"truth": 3, "matches": 2, "misses": 1,
                         "false_positives": 1, "identity_switches": 1,
                         "recall": 2/3, "mean_centroid_error": 0})
        self.assertIsNone(evaluate([], [])["recall"])

    def test_invalid_input_atomic_update(self):
        tracker = Tracker()
        tracker.update([(1, 1)])
        before = (tracker.tracks, tracker.frame, tracker.next_id)
        for invalid in ([(math.nan, 0)], [(0,)], [(0, 0)]*9):
            with self.assertRaises(ValueError):
                tracker.update(invalid)
            self.assertEqual(before, (tracker.tracks, tracker.frame, tracker.next_id))
        for image in ([], [[]], [[1], [0, 1]], [[math.inf]], [[-.1]]):
            with self.assertRaises(ValueError):
                detect(image)
        with self.assertRaises(ValueError):
            assign([], [], 0)
        with self.assertRaises(ValueError):
            evaluate([[]], [])

    def test_sequence_truth_and_occlusion(self):
        f = sequence(101, "occlusion")
        self.assertEqual(len(f), 40)
        self.assertEqual(f, sequence(101, "occlusion"))
        self.assertEqual([i for i, x in enumerate(f) if x["hidden"]], [18, 19, 20])
        self.assertEqual(len(f[19]["truth"]), 2)
        self.assertEqual(len(detect(f[19]["image"])), 1)
        self.assertTrue(all(len(detect(x["image"])) == 2 for x in sequence(101, "clean")))

    def test_capacity_failure_preserves_all_state(self):
        tracker = Tracker(gate=1)
        tracker.update([(i*10, 0) for i in range(8)])
        before = (tracker.tracks, tracker.frame, tracker.next_id)
        with self.assertRaises(ValueError):
            tracker.update([(i*10, 10) for i in range(8)])
        self.assertEqual(before, (tracker.tracks, tracker.frame, tracker.next_id))

    def test_end_to_end_regression_and_no_future_frames(self):
        report = experiment()
        self.assertEqual(len(report["results"]), 18)
        for row in report["results"]:
            metrics = row["metrics"]
            expected = {"clean": (80, 0), "crossing": (78, 2),
                        "occlusion": (77, 0 if row["variant"] == "velocity" else 2)}
            self.assertEqual((metrics["matches"], metrics["identity_switches"]),
                             expected[row["scenario"]])
            self.assertEqual(metrics["false_positives"], 0)
        first, second = Tracker(), Tracker()
        prefix = [[(0, 0)], [(2, 0)], [], [(6, 0)]]
        a = [first.update(d) for d in prefix]
        b = [second.update(d) for d in prefix]
        second.update([(100, 100)])
        self.assertEqual(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--html", metavar="NEW_PATH")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__], exit=True)
    elif args.html:
        html_view(args.html)
        print("Wrote saved CPU sequence viewer:", args.html)
    else:
        print(json.dumps(experiment(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
