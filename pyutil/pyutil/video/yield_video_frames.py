import cv2
from tqdm import tqdm

from typing import Union


def yield_video_frames(
        video_file: Union[str, "Path"],
        color_mode: str = 'bgr',
        image_shape: Union[tuple, None] = None,
        use_pbar: bool = False
):
    """
    Reads video file using opencv and yields frame by frame
    :param video_file: String or path of video file
    :param color_mode: rgb, bgr or gray
    :param image_shape: If given, reshape to this tuple.
    :param use_pbar: If True, output tqdm progressbar
    """

    cap = cv2.VideoCapture(str(video_file))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert cap.isOpened(), 'Video stream is not open'

    pbar = tqdm(total=frame_count) if use_pbar else None

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if image_shape is not None:
            frame = cv2.resize(frame, image_shape)

        if use_pbar:
            pbar.update(1)
        if color_mode.lower() == 'bgr':
            yield frame_count, frame
        elif color_mode.lower() == 'rgb':
            yield frame_count, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif color_mode.lower() in {'gray', 'grayscale'}:
            yield frame_count, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError('Invalid color mode. Only rgb, bgr and gray is supported')
        
        frame_count += 1
    if use_pbar:
        pbar.close()
