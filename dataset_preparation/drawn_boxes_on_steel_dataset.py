import os
import numpy as np
import argparse
import cv2
from tqdm import tqdm

from segmentation.unet.dataset.dataset_generator import SteelDatasetGenerator

boxes_colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0)
]


def boxes_by_mask(mask):
    contours, _ = cv2.findContours(
        mask.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return [cv2.boundingRect(contour) for contour in contours]


def parse_args():
    parser = argparse.ArgumentParser(description='Generate YOLO dataset')

    parser.add_argument('--images-folder', required=True, type=str,
                        help='Path to folder with images.')
    parser.add_argument('--table-csv', required=True, type=str,
                        help='Path to csv table dataset file.')
    parser.add_argument('--result-folder', required=True, type=str,
                        help='Path to result folder.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = SteelDatasetGenerator(
        args.images_folder,
        args.table_csv,
        False,
        0.0
    )

    if not os.path.isdir(args.result_folder):
        os.makedirs(args.result_folder)

    for i in tqdm(range(len(dataset))):
        image_tensor, masks_tensor = dataset[i]

        save_image_path = os.path.join(
            args.result_folder,
            '{}_image.jpg'.format(i + 1)
        )

        image = (image_tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for cls, ch in enumerate(masks_tensor):
            channel_mask = (ch.numpy() * 255.0).astype(np.uint8)

            for rect in boxes_by_mask(channel_mask):
                x, y, w, h = rect
                image = cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    boxes_colors[cls],
                    2
                )

        cv2.imwrite(save_image_path, image)
