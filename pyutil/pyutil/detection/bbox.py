from pathlib import Path

import numpy as np
import json
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

    np.save(str(output_path), detections)


def load_detections(
        input_path: Union[str, Path]
) -> Iterable:
    """
    Loads detections from npy file
    :param input_path: File path
    :return: Detections
    """
    input_path = Path(input_path)

    assert input_path.exists(), f'{input_path} does not exist'

    return np.load(str(input_path))