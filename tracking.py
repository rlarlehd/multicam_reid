"""Small deterministic IoU tracker and a cross-camera cosine matching gallery."""
from dataclasses import dataclass
import numpy as np


def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1])
    x2,y2=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    ar1=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    ar2=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return intersection/(ar1+ar2-intersection+1e-12)


class LocalTracker:
    def __init__(self, threshold=.3, max_age=15):
        self.threshold,self.max_age=threshold,max_age
        self.next_id=1
        self.tracks={}
        self.frame=0

    def update(self, boxes):
        self.frame+=1
        candidates=[]
        for tid, (old_box,last) in self.tracks.items():
            if self.frame-last <= self.max_age:
                for j,box in enumerate(boxes):
                    score=iou(old_box,box)
                    if score >= self.threshold:
                        candidates.append((score,tid,j))
        used_tracks,used_boxes=set(),set()
        assigned={}
        for _,tid,j in sorted(candidates,reverse=True):
            if tid not in used_tracks and j not in used_boxes:
                assigned[j]=tid
                used_tracks.add(tid)
                used_boxes.add(j)
        for j in range(len(boxes)):
            if j not in assigned:
                assigned[j]=self.next_id
                self.next_id+=1
            self.tracks[assigned[j]]=(boxes[j],self.frame)
        self.tracks={tid:entry for tid,entry in self.tracks.items() if self.frame-entry[1] <= self.max_age}
        return [assigned[j] for j in range(len(boxes))]


@dataclass
class Identity:
    feature: np.ndarray
    camera: str
    last_time: float
    count: int = 1


class GlobalMatcher:
    """Local track -> global ID. Allow linking ONLY to identities last seen elsewhere."""
    def __init__(self, threshold=.75, max_gap=120., momentum=.85):
        self.threshold,self.max_gap,self.momentum=threshold,max_gap,momentum
        self.next_id=1
        self.gallery={}
        self.local_to_global={}

    def match(self, camera, track_id, feature, timestamp):
        key=(camera,track_id)
        feat=np.asarray(feature,dtype=np.float32)
        feat=feat/(np.linalg.norm(feat)+1e-12)
        if key in self.local_to_global:
            gid=self.local_to_global[key]
            self._update(gid,feat,camera,timestamp)
            return gid,1.0,False
        candidates=[]
        for gid,identity in self.gallery.items():
            delta=timestamp-identity.last_time
            if identity.camera != camera and 0 <= delta <= self.max_gap:
                candidates.append((float(np.dot(feat,identity.feature)),gid))
        if candidates:
            score,gid=max(candidates)
            if score>=self.threshold:
                self.local_to_global[key]=gid
                self._update(gid,feat,camera,timestamp)
                return gid,score,True
        gid=self.next_id
        self.next_id+=1
        self.gallery[gid]=Identity(feat,camera,timestamp)
        self.local_to_global[key]=gid
        return gid,float(max(candidates)[0]) if candidates else 0.0,False

    def _update(self,gid,feat,camera,timestamp):
        identity=self.gallery[gid]
        blended=self.momentum*identity.feature+(1-self.momentum)*feat
        identity.feature=blended/(np.linalg.norm(blended)+1e-12)
        identity.camera=camera
        identity.last_time=timestamp
        identity.count+=1
