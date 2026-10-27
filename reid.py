"""SOLIDER-REID official model adapter (fine-tuned ReID checkpoint required)."""
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class SOLIDEREmbedder:
    def __init__(self, repo, config, weights, device='cuda', batch_size=16):
        repo, config, weights = Path(repo).resolve(), Path(config).resolve(), Path(weights).resolve()
        if not (repo / 'model' / 'make_model.py').exists():
            raise FileNotFoundError(f'Official SOLIDER-REID repository not found: {repo}')
        for p in (config, weights):
            if not p.is_file():
                raise FileNotFoundError(p)
        sys.path.insert(0, str(repo))
        from config import cfg as default_cfg
        from model import make_model
        cfg = default_cfg.clone()
        cfg.defrost()
        cfg.merge_from_file(str(config))
        cfg.MODEL.PRETRAIN_PATH = ''  # Do not initialize pretrained backbone twice
        cfg.MODEL.PRETRAIN_CHOICE = 'finetune'
        cfg.TEST.NECK_FEAT = 'after'
        cfg.MODEL.SIE_CAMERA = False
        cfg.MODEL.SIE_VIEW = False
        cfg.freeze()
        self.device = torch.device(device if device != 'cuda' or torch.cuda.is_available() else 'cpu')
        if device.startswith('cuda') and self.device.type == 'cpu':
            print('[warning] CUDA unavailable; switching to CPU')
        # Only load checkpoints from a trusted source: torch.load uses pickle.
        ckpt = torch.load(weights, map_location='cpu', weights_only=True)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if not isinstance(ckpt, dict):
            raise ValueError('Expected a state_dict checkpoint, not a raw backbone checkpoint')
        ckpt = {k.removeprefix('module.'): v for k, v in ckpt.items()}
        # Ignore classification heads; infer classes to initialize official module.
        num_classes = ckpt['classifier.weight'].shape[0] if 'classifier.weight' in ckpt else 1
        self.model = make_model(cfg, num_class=num_classes, camera_num=0, view_num=0,
                                semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
        own = self.model.state_dict()
        essential = [k for k in own if k.startswith(('base.', 'bottleneck.'))]
        matched = [k for k in essential if k in ckpt and ckpt[k].shape == own[k].shape]
        if not essential or len(matched) / len(essential) < 0.95:
            raise RuntimeError(f'Incompatible SOLIDER checkpoint: matched {len(matched)}/{len(essential)} essential tensors. Check config/backbone and use a fine-tuned ReID model.')
        compatible = {k:v for k,v in ckpt.items() if k in own and v.shape == own[k].shape}
        self.model.load_state_dict(compatible, strict=False)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self.width, self.height = int(cfg.INPUT.SIZE_TEST[1]), int(cfg.INPUT.SIZE_TEST[0])
        self.mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3,1,1)
        self.std = torch.tensor(cfg.INPUT.PIXEL_STD).view(3,1,1)
        print(f'[SOLIDER] loaded {len(matched)}/{len(essential)} essential tensors, device={self.device}')

    @torch.inference_mode()
    def encode(self, crops):
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)
        result=[]
        for start in range(0, len(crops), self.batch_size):
            batch=[]
            for crop in crops[start:start+self.batch_size]:
                if crop.size == 0:
                    raise ValueError('Empty person crop')
                rgb=cv2.cvtColor(cv2.resize(crop,(self.width,self.height)), cv2.COLOR_BGR2RGB)
                x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float().div_(255)
                batch.append((x-self.mean)/self.std)
            output=self.model(torch.stack(batch).to(self.device))
            if isinstance(output, (tuple,list)):
                output=output[0]
            if output.ndim != 2:
                raise RuntimeError(f'Unexpected embedding shape: {tuple(output.shape)}')
            result.append(F.normalize(output.float(), p=2, dim=1).cpu().numpy())
        return np.concatenate(result,axis=0)
