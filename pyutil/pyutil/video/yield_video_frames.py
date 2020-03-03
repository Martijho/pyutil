import cv2

from typing import Union


def yield_video_frames(
        video_file: Union[str, "Path"],
        color_mode: str = 'bgr',
        image_shape: Union[tuple, None] = None
):
    """
    Reads video file using opencv and yields frame by frame
    :param video_file: String or path of video file
    :param color_mode: rgb or bgr
    :param image_shape: If given, reshape to this tuple.
    """

    cap = cv2.VideoCapture(str(video_file))
    frame_count = cv2.CAP_PROP_FRAME_COUNT

    assert cap.isOpened(), 'Video stream is not open'

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if image_shape is not None:
            frame = cv2.resize(frame, image_shape)

        if color_mode.lower() == 'bgr':
            yield frame_count, frame
        elif color_mode.lower() == 'rgb':
            yield frame_count, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_count += 1

