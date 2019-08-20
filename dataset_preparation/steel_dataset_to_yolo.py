import os
import numpy as np
import argparse
import cv2
from tqdm import tqdm

from segmentation.unet.dataset.dataset_generator import SteelDatasetGenerator


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
    parser.add_argument('--yolo-dataset', required=True, type=str,
                        help='Full path to result YOLO dataset.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = SteelDatasetGenerator(
        args.images_folder,
        args.table_csv,
        False,
        0.0
    )

    if not os.path.isdir(os.path.join(args.yolo_dataset, 'images/')):
        os.makedirs(os.path.join(args.yolo_dataset, 'images/'))

    if not os.path.isdir(os.path.join(args.yolo_dataset, 'labels/')):
        os.makedirs(os.path.join(args.yolo_dataset, 'labels/'))

    with open(os.path.join(args.yolo_dataset, 'dataset.txt'), 'w') as f:
        for i in tqdm(range(len(dataset))):
            image_tensor, masks_tensor = dataset[i]

            save_image_path = os.path.join(
                os.path.join(args.yolo_dataset, 'images/'),
                '{}_image.jpg'.format(i + 1)
            )

            save_label_path = os.path.join(
                os.path.join(args.yolo_dataset, 'labels/'),
                '{}_image.txt'.format(i + 1)
            )

            cv2.imwrite(
                save_image_path,
                (image_tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)
            )

            f.write(save_image_path + '\n')

            with open(save_label_path, 'w') as label_f:
                for cls, ch in enumerate(masks_tensor):
                    channel_mask = (ch.numpy() * 255.0).astype(np.uint8)

                    for rect in boxes_by_mask(channel_mask):
                        label_f.write(
                            '{cls} {x} {y} {w} {h}\n'.format(
                                cls=cls,
                                x=rect[0] / channel_mask.shape[1],
                                y=rect[1] / channel_mask.shape[0],
                                w=rect[2] / channel_mask.shape[1],
                                h=rect[3] / channel_mask.shape[0],
                            )
                        )
