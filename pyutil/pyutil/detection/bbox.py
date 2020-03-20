from pathlib import Path

import numpy as np
from typing import List, Union, Iterable


def clip(
        box: List[Union[int, float]],
        w: Union[int, float] = 1.0,
        h: Union[int, float] = 1.0
) -> List[Union[int, float]]:
    """
    Clips box coordinates to be within image dim. If box coordinates is in absolute format, set w and h to image
    width and height
    :param box: List of box coordinates (either relative or absolute)
    :param w: Image width (1 if relative coordinates)
    :param h: Image height (1 if relative coordinates)
    :return: New list of clipped box coordinates
    """
    return [
        max(0, min(w, box[0])),
        max(0, min(h, box[1])),
        max(0, min(w, box[2])),
        max(0, min(h, box[3])),
    ]


def pad_box(
        box: List[Union[int, float]],
        ratio: float = 0.1
) -> List[Union[int, float]]:
    """
    Pad box with a % increase in all dimensions.
    :param box: Box coordinates
    :param ratio: How much of side length to expand in all direction. 0.1 gives a new side length of 120% of original
    :return: New box coordinates
    """
    xmin, ymin, xmax, ymax = box
    w, h = xmax - xmin, ymax - ymin

    xmin -= w * ratio
    ymin -= h * ratio
    xmax += w * ratio
    ymax += h * ratio

    return [xmin, ymin, xmax, ymax]


def pad_box_to_aspect_ratio(
        box: List[Union[int, float]],
        image_height: Union[int, float] = 1.0,
        image_width: Union[int, float] = 1.0,
        aspect_ratio: float = 1.0,
) -> List[Union[int, float]]:
    """
    Expand box to fit new desired aspect ration without loosing original area of image.
    If it is not enough
    :param box:
    :param image_height:
    :param image_width:
    :param aspect_ratio:
    :return:
    """
    def pad_func(low_edge, high_edge, new_side_length, img_dim):
        pad = new_side_length - (high_edge - low_edge)

        available_space = img_dim - (high_edge - low_edge)

        if available_space > pad:
            # Enough space to pad
            if low_edge >= pad / 2 and img_dim - high_edge >= pad / 2:
                # Pad equal amount on both sides
                low_pad, high_pad = pad / 2, pad / 2
            elif low_edge < img_dim - high_edge:
                # Pad to image edge on low side. Rest on high side
                low_pad = low_edge
                high_pad = pad - low_pad
            else:
                # Pad to image edge on high side. Rest on low side
                high_pad = img_dim - high_edge
                low_pad = pad - high_pad

            return low_edge - low_pad, high_edge + high_pad

        # Not enough space to pad to aspect ratio. Grab as much space as possible
        return 0, img_dim

    xmin, ymin, xmax, ymax = box
    w, h = xmax - xmin, ymax - ymin
    assert w > 0 and h > 0, f'Box must have side lengths > 0. {box}'

    if w / h < aspect_ratio:
        xmin, xmax = pad_func(xmin, xmax, h * aspect_ratio, image_width)
    elif w / h > aspect_ratio:
        ymin, ymax = pad_func(ymin, ymax, w / aspect_ratio, image_height)

    return [xmin, ymin, xmax, ymax]


def box_to_absolute_coordinates(
        box: List[Union[int, float]],
        image_height: int,
        image_width: int
) -> List[Union[int]]:
    """
    Conver box coordinates to absolute coordinates
    :param box: Bounding box in relative coordinates
    :param image_height: Image height in pixels
    :param image_width: Image width in pixels
    :return: Box in absolute coordinates
    """
    box = np.array(box)
    assert np.all(box <= 1.0), 'Box coordinates is not relative'

    return (box * np.array([image_width, image_height, image_width, image_height])).astype(int).tolist()


def box_to_relative_coordinates(
        box: List[Union[int, float]],
        image_height: int,
        image_width: int
) -> List[Union[float]]:
    """
    Conver box coordinates to relative coordinates
    :param box: Bounding box in absolute coordinates
    :param image_height: Image height in pixels
    :param image_width: Image width in pixels
    :return: Box in relative coordinates
    """
    box = np.array(box)
    assert np.all(box[2:] > 1.0), 'Box coordinates is not absolute'

    return (box / np.array([image_width, image_height, image_width, image_height])).tolist()


def save_detections(
        detections: Iterable,
        output_path: Union[str, Path]
):
    """
    Dumps detections to npy file
    :param detections: Detection data.
    :param output_path: Where to store detections
    """
    output_path = Path(output_path)

    assert output_path.parent.exists(), f'{output_path.parent} does not exist.'

    np.save(str(output_path), detections, allow_pickle=True)


def load_detections(
        input_path: Union[str, Path]
):
    """
    Loads detections from npy file
    :param input_path: File path
    :return: Detections
    """
    input_path = Path(input_path)

    assert input_path.exists(), f'{input_path} does not exist'

    return np.load(str(input_path), allow_pickle=True).tolist()


def calc_iou(
        box1: Union[list, np.ndarray],
        box2: Union[list, np.ndarray]
) -> float:
    """
    Calculates Intersection Over Union between two boxes
    :param box1: Coordinates for box 1
    :param box2: Coordinates for box 2
    :return: Intersection Over Union
    """
    # TODO: verify coordinate mode is the same
    assert len(box1) == len(box2) == 4

    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)
    interArea = max((xB - xA + 1), 0) * max((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou
