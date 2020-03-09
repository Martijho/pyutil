import cv2
from pathlib import Path

from typing import Union, Tuple


def get_video_stats(
        video_file: Union[str, Path]
) -> Tuple[int, Tuple[int, int]]:
    """
    Returns information about video.
    :param video_file: path to video file
    :return: (#frames, (image height, image width)
    """
    assert Path(video_file).exists(), f'{video_file} does not exist'

    cap = cv2.VideoCapture(str(video_file))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return n_frames, (h, w)