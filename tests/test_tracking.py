import unittest
import numpy as np
from tracking import iou, LocalTracker, GlobalMatcher


class TrackingTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0,0,10,10),(0,0,10,10)),1.)
        self.assertEqual(iou((0,0,10,10),(20,20,30,30)),0.)

    def test_local_track(self):
        t=LocalTracker()
        self.assertEqual(t.update([(0,0,30,50)]),[1])
        self.assertEqual(t.update([(1,1,31,51)]),[1])

    def test_cross_camera(self):
        g=GlobalMatcher(threshold=.8)
        f=np.array([1.,0.,0.])
        a,_,_=g.match('cam1',1,f,0.)
        b,score,cross=g.match('cam2',1,f,1.)
        self.assertEqual(a,b)
        self.assertTrue(cross)
        self.assertGreater(score,.99)
        c,_,_=g.match('cam2',2,np.array([0.,1.,0.]),1.)
        self.assertNotEqual(c,a)

if __name__=='__main__':unittest.main()
