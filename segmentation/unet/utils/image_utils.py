import numpy as np
import cv2


def resize_coeff(x, new_x):
    """
    Evaluate resize coefficient from image shape
    Args:
        x: original value
        new_x: expect value

    Returns:
        Resize coefficient
    """
    return new_x / x


def resize_image(img, resize_shape=(128, 128), interpolation=cv2.INTER_AREA):
    """
    Resize single image
    Args:
        img: input image
        resize_shape: resize shape in format (height, width)
        interpolation: interpolation method

    Returns:
        Resized image
    """
    return cv2.resize(img, None, fx=resize_coeff(img.shape[1], resize_shape[1]),
                     fy=resize_coeff(img.shape[0], resize_shape[0]),
                     interpolation=interpolation)


def crop_image_by_center(x, shape):
    x = x.transpose(2, 0, 1)

    target_shape = shape[-2:]
    input_tensor_shape = x.shape[-2:]

    crop_by_y = (input_tensor_shape[0] - target_shape[0]) // 2
    crop_by_x = (input_tensor_shape[1] - target_shape[1]) // 2

    indexes_by_y = (
        crop_by_y, input_tensor_shape[0] - crop_by_y
    )

    indexes_by_x = (
        crop_by_x, input_tensor_shape[1] - crop_by_x
    )

    x = x[:, indexes_by_y[0]:indexes_by_y[1],
           indexes_by_x[0]:indexes_by_x[1]]

    return x.transpose(1, 2, 0)


def upper_bin(img, threshold):
    res = img.copy()
    res[img > threshold] = 255
    res[img <= threshold] = 0
    return res


def ring_by_np(size):
    res = np.zeros(shape=(size, size), dtype=np.uint8)
    m = size // 2
    for i in range(size):
        for j in range(size):
            if (i - m) ** 2 + (j - m) ** 2 <= m ** 2:
                res[i][j] = 255
    return res


def increase_sharpen(img):
    img_blured = cv2.GaussianBlur(img, (5, 5), 0)
    img_m = cv2.addWeighted(img, 1.5, img_blured, -0.5, 0)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_s = cv2.filter2D(img_m, -1, kernel, borderType=cv2.CV_8U)
    return img_s


def image_to_quadrate(img, shape):
    crop_shape = list(img.shape[:2])
    if crop_shape[0] > crop_shape[1]:
        crop_shape[0] -= (crop_shape[0] - crop_shape[1])
    else:
        crop_shape[1] -= (crop_shape[1] - crop_shape[0])

    t_img = crop_image_by_center(img, crop_shape)
    t_img = resize_image(t_img, shape)

    return t_img


def pad_image_to_qadrate(img):
    size = max(img.shape[0], img.shape[1])

    x_pad = (size - img.shape[1]) // 2
    y_pad = (size - img.shape[0]) // 2

    result = (np.random.rand(size, size, 3) * 255).astype(np.uint8)

    result[
        y_pad:y_pad + img.shape[0],
        x_pad:x_pad + img.shape[1]:
    ] = img.copy()

    return result, x_pad, y_pad


def mask2rle(mask):
    """
    Complete predict-length encoding for mask (binary image). It is used to create submission file.
    :param mask: Mask (binary image).
    :return: List of form [index count index count ...]
    """
    pixels = mask.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle2str(runs):
    """
    Convert RLE list to string.
    :param runs: List of codes.
    :return: String of rle.
    """
    return ' '.join(str(x) for x in runs)


def split_on_tiles_h(img, num_tiles=None):
    """ Split image into tiles of size (height x height)
    :param num_tiles: Number of tiles to split by width. If None, the possible lower number will be used.
    """
    img = np.array(img)
    h, w = img.shape
    if num_tiles is None:
        num_tiles = int(np.ceil(np.array(img).shape[1] / h))
    assert num_tiles * h >= w
    pad_width = num_tiles * h - w
    padded_img = np.pad(img, ((0, 0), (0, pad_width)))
    tiles = np.hsplit(padded_img, num_tiles)
    return tiles


def combine_tiles(tiles, src_h, src_w):
    """ Combine tiles into one image and unpad. """
    if tiles.ndim == 3:
        return np.dstack(tiles)[:src_h, :src_w]
    elif tiles.ndim == 4:
        return np.dstack(tiles)[:, :src_h, :src_w]
    else:
        raise ValueError('Unsupported tiles ndim: {}'.format(tiles.ndim))
