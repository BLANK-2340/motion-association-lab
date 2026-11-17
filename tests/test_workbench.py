import json
import unittest
from export_workbench import TARGET, build, canonical, capture, controlled
from tracking_lab import detect


class WorkbenchTests(unittest.TestCase):
    def test_frozen_export(self):
        self.assertEqual(TARGET.read_text(), canonical())

    def test_controlled_pixels_and_missing_frames(self):
        frames=controlled(2, 8, 5)
        self.assertEqual([t for t,f in enumerate(frames) if f['hidden']], list(range(18,23)))
        self.assertEqual(frames[39]['truth'], [(1,(88,26)),(2,(10,34))])
        self.assertEqual(len(detect(frames[0]['image'])), 2)
        self.assertEqual(len(detect(frames[19]['image'])), 1)

    def test_independent_capture_counts(self):
        result=capture(controlled(1,8,0))
        for m in result['metrics']:
            self.assertEqual((m['matches'],m['misses'],m['identity_switches']), (80,0,0))
        for f in result['frames']:
            self.assertEqual(len(f['foreground']),50)
            self.assertEqual(len(f['tracks'][0]),2)

    def test_sweep_unique_protocol_and_evidence_counts(self):
        data=build()
        cells={(r['speed'],r['separation'],r['gap']) for r in data['sweep']}
        self.assertEqual(len(cells),24)
        for c in data['cases']:
            self.assertEqual(len(c['frames']),40)
            for v in (0,1):
                emitted=sum(len(f['tracks'][v]) for f in c['frames'])
                m=c['metrics'][v]
                self.assertEqual(emitted,m['matches']+m['false_positives'])
                self.assertEqual(m['matches']+m['misses'],80)


if __name__=='__main__':
    unittest.main()
