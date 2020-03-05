import cv2
import numpy as np

from collections import defaultdict
from typing import Union, List, Tuple


def get_random_color_map(
        a_min: int = 0,
        a_max: int = 255,
        n: int = 3
) -> defaultdict:
    """
    Returns a defaultdict that defaults to a random numpy-array of n values between a_min and a_max
    :param a_min: Min array value
    :param a_max: Max array value
    :param n: Number of random values
    :return: Defaultdict with values
    """
    return defaultdict(lambda: np.random.randint(a_min, a_max, n))


def draw_bounding_box(
        image: np.ndarray,
        detection: dict,
        show_label: bool = True,
        show_confidence: bool = True,
        color: Union[np.ndarray, Tuple, List] = (255, 255, 255)
) -> np.ndarray:
    """
    Will overlay a bounding box on provided image.
    :param image: Image to overlay on
    :param detection: Detection dict to overlay. must have 'box', 'label' and 'confidence' entries
    :param show_label: If True, writes box label in image
    :param show_confidence: If True, write box confidence in image
    :param color: Color to be used
    :return: Image
    """
    thickness = max(image.shape) // 350
    h, w = image.shape[:2]

    scale = np.array([1, 1, 1, 1]) if np.all(np.array(detection['box']) < 1.0) else np.array([w, h, w, h])
    xmin, ymin, xmax, ymax = (np.array(detection['box']) * scale).astype(int)

    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)

    out_txt = ''
    if show_label:
        out_txt = detection['label']
    if show_confidence:
        out_txt += f'({detection["confidence"]:.2f})'

    if out_txt != '':
        image = cv2.putText(
            image,
            out_txt,
            (xmin, ymin - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color=color,
            thickness=thickness
        )
    return image


def draw_keypoints(
        image: np.ndarray,
        keypoints: list,
        limbs: Union[list, None] = None,
        color: Union[np.ndarray, Tuple, List] = (255, 255, 255),
        limb_color: Union[np.ndarray, Tuple, List] = (255, 255, 255),
) -> np.ndarray:
    """
    Will overlay a set of keypoints on image. If limbs are provided, lines will be drawn to mark them
    Function sees negative values as to be skipped
    :param image: Image to overlay on
    :param keypoints: List of keypoints to use. Must be a list with tuples, arrays or np.ndarrays
    :param limbs: List of limbs to use. Must be a list of index pairs for all limbs. Indexes must match elements in
    keypoints.
    :param color: Color to be used for points
    :param limb_color: Color to be used for limbs
    :return: Image
    """
    pose = np.array(keypoints).astype(int)
    thickness = max(image.shape) // 350

    if limbs is not None:
        for src, dst in limbs:
            x1, y1, x2, y2 = pose[src][0], pose[src][1], pose[dst][0], pose[dst][1]
            # Dont draw line if one of the points is negative
            if np.any(pose[(src, dst), :] < 0):
                continue
            image = cv2.line(image, (x1, y1), (x2, y2), limb_color, thickness=thickness)

    for x, y in pose:
        if x < 0 or y < 0:
            continue
        image = cv2.circle(image, (x, y), thickness, color, thickness=-1)

    return image
