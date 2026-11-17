"""Recompute saved tracking evidence and a fixed controlled sensitivity sweep."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from tracking_lab import Tracker, detect, evaluate, sequence

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'docs/evidence.json'


def capture(frames):
    trackers = [Tracker(motion=False), Tracker(motion=True)]
    snapshots = []
    for frame in frames:
        detections = detect(frame['image'])
        tracks = [[(t.identity, t.position) for t in tracker.update(detections)] for tracker in trackers]
        snapshots.append({'truth': frame['truth'], 'hidden': frame['hidden'],
                          'foreground': [[x, y] for y, row in enumerate(frame['image'])
                                         for x, p in enumerate(row) if p > .5],
                          'detections': detections, 'tracks': tracks})
    return {'frames': snapshots,
            'metrics': [evaluate([s['truth'] for s in snapshots],
                                 [s['tracks'][v] for s in snapshots]) for v in (0, 1)]}


def controlled(speed, separation, gap):
    """40 frames, mid-sequence disappearance, fixed gate/memory, no tuning."""
    frames = []
    for time in range(40):
        truth = [(1, (10+speed*time, 26)), (2, (88-speed*time, 26+separation))]
        hidden = [1] if 18 <= time < 18+gap else []
        image = [[0.]*100 for _ in range(64)]
        for identity, (x, y) in truth:
            if identity not in hidden:
                for py in range(y-2, y+3):
                    for px in range(x-2, x+3):
                        image[py][px] = .9
        frames.append({'image': image, 'truth': truth, 'hidden': hidden})
    return frames


def build():
    cases = []
    for seed, scenario in itertools.product((101, 202, 303), ('clean', 'crossing', 'occlusion')):
        cases.append({'name': f'{scenario} / seed {seed}', 'scenario': scenario,
                      'seed': seed, **capture(sequence(seed, scenario))})
    sweep = []
    for speed, separation, gap in itertools.product((1, 2), (0, 2, 8), (0, 1, 3, 5)):
        sweep.append({'speed': speed, 'separation': separation, 'gap': gap,
                      'metrics': capture(controlled(speed, separation, gap))['metrics']})
    return {'schema': 1, 'scope': 'saved CPU results, not live inference',
            'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                              for p in ('tracking_lab.py', 'export_workbench.py')},
            'protocol': {'frames': 40, 'width': 100, 'height': 64, 'detector_threshold': .5,
                         'association_gate': 10, 'memory': 3, 'evaluation_gate': 6,
                         'variants': ['last_position', 'velocity']},
            'cases': cases, 'sweep': sweep}


def canonical():
    return json.dumps(build(), sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = canonical()
    if args.check:
        if TARGET.read_text(encoding='utf-8') != content:
            raise SystemExit('saved evidence differs from source recomputation')
        print('saved CPU evidence and 24-cell sweep reproduce exactly')
    else:
        TARGET.parent.mkdir(exist_ok=True)
        TARGET.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
