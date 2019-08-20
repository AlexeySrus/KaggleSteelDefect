import argparse
import time
from sys import platform

if platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')

from models import *
from utils.datasets import *
from utils.utils import *


class YoloDetector:
    def __init__(self, config, device='cpu'):
        self.device = device
        self.img_size = config['img_size']
        self.model = Darknet(config['cfg'], self.img_size)
        self.model.load_state_dict(
            torch.load(config['weights'], map_location=device)['model']
        )

        self.model.fuse()
        self.model.to(device).eval()

        self.confidence = config['conf_thres']
        self.numbers_of_thresholds = config['nms_thres']

        self.classes = load_classes(parse_data_cfg(config['data_cfg'])['names'])

    def detect(self, inp_img, prepare=True):
        """
        Detection by yolo
        Args:
            inp_img: input image in RGB format as np.uint8 array

        Returns:
            yolo detections boxes, classes, confidences
        """
        if prepare:
            d = letterbox(inp_img, new_shape=self.img_size)
            img = d[0]
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0

            img = torch.from_numpy(
                img.transpose((2, 0, 1))
            ).unsqueeze(0).to(self.device)
        else:
            img = inp_img

        pred, _ = self.model(img)
        det = non_max_suppression(
            pred, self.confidence, self.numbers_of_thresholds
        )[0]

        if det is not None and len(det) > 0:
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], inp_img.shape[:2]
            ).round()

            return (
                det[:, :-1].detach().to('cpu').numpy(),
                [
                    self.classes[int(d)]
                    for d in det.detach().to('cpu').numpy()[:, -1]
                ],
                det[:, 4].detach().to('cpu').numpy()
            )
        return None
