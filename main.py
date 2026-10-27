"""Offline multi-camera person detection + SOLIDER ReID smoke test."""
import argparse
import csv
from pathlib import Path
import cv2
from ultralytics import YOLO
from reid import SOLIDEREmbedder
from tracking import LocalTracker,GlobalMatcher


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--videos',nargs='+',required=True,help='Two or more videos, one per camera')
    p.add_argument('--repo',required=True,help='Local official SOLIDER-REID repository')
    p.add_argument('--config',required=True,help='SOLIDER-REID model YAML used for the checkpoint')
    p.add_argument('--weights',required=True,help='Fine-tuned SOLIDER-REID checkpoint (.pth)')
    p.add_argument('--detector',default='yolov8n.pt')
    p.add_argument('--device',default='cuda')
    p.add_argument('--out',default='outputs')
    p.add_argument('--conf',type=float,default=.35)
    p.add_argument('--similarity',type=float,default=.75)
    p.add_argument('--max-gap',type=float,default=120.)
    p.add_argument('--fps-sample',type=float,default=2.,help='Frames sampled per second per camera')
    p.add_argument('--max-seconds',type=float,default=60.,help='Process up to this many seconds')
    p.add_argument('--batch-size',type=int,default=16)
    return p.parse_args()


def main():
    a=parse_args()
    if len(a.videos)<2: raise ValueError('At least two cameras required')
    if a.fps_sample<=0 or a.max_seconds<=0: raise ValueError('Sampling FPS and duration must be positive')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    embedder=SOLIDEREmbedder(a.repo,a.config,a.weights,a.device,a.batch_size)
    detector=YOLO(a.detector)
    matcher=GlobalMatcher(a.similarity,a.max_gap)
    captures=[]
    writers={}
    try:
        for i,path in enumerate(a.videos):
            cap=cv2.VideoCapture(path)
            if not cap.isOpened(): raise RuntimeError(f'Cannot open video: {path}')
            fps=cap.get(cv2.CAP_PROP_FPS)
            if fps<=0: raise RuntimeError(f'Invalid input FPS: {path}')
            width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            camera=f'cam{i+1}'
            captures.append(dict(name=camera,cap=cap,fps=fps,next_index=0,tracker=LocalTracker(),done=False,source=path))
            writer=cv2.VideoWriter(str(out/f'{camera}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),min(fps,a.fps_sample),(width,height))
            if not writer.isOpened(): raise RuntimeError('VideoWriter initialization failed')
            writers[camera]=writer
        with (out/'matches.csv').open('w',newline='',encoding='utf-8') as f:
            csvwriter=csv.writer(f)
            csvwriter.writerow(['camera','source','timestamp_s','local_track_id','global_id','matched_other_camera','similarity','x1','y1','x2','y2'])
            while any(not c['done'] for c in captures):
                # Process interleaved in approximate video-time order; videos assumed synchronized at t=0.
                available=[c for c in captures if not c['done']]
                c=min(available,key=lambda x:x['next_index']/x['fps'])
                frame_index=c['next_index']
                t=frame_index/c['fps']
                if t>=a.max_seconds:
                    c['done']=True;continue
                cap=c['cap'];cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
                ok,frame=cap.read()
                if not ok: c['done']=True;continue
                c['next_index']=int(round((round(t*a.fps_sample)+1)*c['fps']/a.fps_sample))
                if c['next_index']<=frame_index:c['next_index']=frame_index+1
                det=detector.predict(frame,classes=[0],conf=a.conf,verbose=False,device=embedder.device.type)[0]
                boxes=[];crops=[]
                h,w=frame.shape[:2]
                for xyxy in det.boxes.xyxy.cpu().numpy():
                    x1,y1,x2,y2=map(int,xyxy)
                    x1,x2=max(0,x1),min(w,x2);y1,y2=max(0,y1),min(h,y2)
                    if x2-x1<12 or y2-y1<24:continue
                    boxes.append((x1,y1,x2,y2));crops.append(frame[y1:y2,x1:x2])
                tids=c['tracker'].update(boxes)
                if crops:
                    features=embedder.encode(crops)
                    for box,tid,feat in zip(boxes,tids,features):
                        gid,score,cross=matcher.match(c['name'],tid,feat,t)
                        x1,y1,x2,y2=box
                        color=((37*gid)%200+30,(91*gid)%200+30,(147*gid)%200+30)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                        cv2.putText(frame,f'G{gid} / T{tid}',(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
                        csvwriter.writerow([c['name'],c['source'],f'{t:.3f}',tid,gid,int(cross),f'{score:.4f}',*box])
                cv2.putText(frame,f"{c['name']} t={t:.1f}s",(15,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                writers[c['name']].write(frame)
        print(f'Done. Results in {out.resolve()}')
    finally:
        for c in captures:c['cap'].release()
        for w in writers.values():w.release()


if __name__=='__main__':main()
