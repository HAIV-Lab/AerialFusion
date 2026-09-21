import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .Dynamic_Fusion_Net.Fusion_Net.modules.generator import Generator

device = torch.device(os.environ.get("AERIALFUSION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

class Dynamic_Fusion():
    def __init__(self):
        super(Dynamic_Fusion, self).__init__()

    def input_setup_all(self, data_vi, data_ir):
        padding = 0
        sub_ir_sequence = []
        sub_vi_sequence = []
        _ir = data_ir
        _vi = data_vi
        input_ir = (_ir - 127.5) / 127.5
        input_ir = np.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_ir.shape
        input_ir = input_ir.reshape([w, h, 1])
        input_vi = (_vi - 127.5) / 127.5
        input_vi = np.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        w, h = input_vi.shape
        input_vi = input_vi.reshape([w, h, 1])
        sub_ir_sequence.append(input_ir)
        sub_vi_sequence.append(input_vi)
        train_data_ir = np.asarray(sub_ir_sequence)
        train_data_vi = np.asarray(sub_vi_sequence)
        return train_data_ir, train_data_vi

    def fusion_color(self, vi, ir, vi_color, SG_vi = None, SG_ir = None):
        g = Generator().to(device)
        path = Path(__file__).parent
        path = path / 'Dynamic_Fusion_Net/checkpoint/epoch_72/model-72.pt'
        weights = torch.load(path, map_location=device)
        g.load_state_dict(weights, strict=False)                                                                        # 载入权重
        g.eval()

        with torch.no_grad():
            train_data_ir, train_data_vi = self.input_setup_all(vi, ir)
            train_data_ir = train_data_ir.transpose([0, 3, 1, 2])
            train_data_vi = train_data_vi.transpose([0, 3, 1, 2])

            train_data_ir = torch.tensor(train_data_ir).float().to(device)
            train_data_vi = torch.tensor(train_data_vi).float().to(device)

            if SG_vi is not None and SG_ir is not None:
                SG_ir = torch.tensor(SG_ir).unsqueeze(0).unsqueeze(0).float().to(device)
                SG_vi = torch.tensor(SG_vi).unsqueeze(0).unsqueeze(0).float().to(device)

            result = g(train_data_ir, train_data_vi, SG_ir = None, SG_vi = None)                                       # 单通道融合图
            result = np.squeeze(result.cpu().numpy() * 127.5 + 127.5).astype(np.uint8)                                 # 单通道

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result = clahe.apply(result)              # 单通道

            img_vi_rgb = cv2.cvtColor(vi_color, cv2.COLOR_BGR2RGB)
            img_vi_ycrcb = cv2.cvtColor(img_vi_rgb, cv2.COLOR_RGB2YCR_CB)
            img_copy = img_vi_ycrcb
            img_copy[:, :, 0] = result
            img_f_bgr = cv2.cvtColor(img_copy, cv2.COLOR_YCR_CB2BGR)

            return img_f_bgr
