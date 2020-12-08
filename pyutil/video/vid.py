from typing import Tuple, Union

import cv2
import numpy as np
from pathlib import Path

from pyutil.image.img import Img


class Vid:
    def __init__(
            self,
            path: Union[str, Path],
            color_mode: str = 'bgr',
            size: Tuple[int, int] = None,
            as_img_objects: bool = True
            # TODO: implement initialization from directory of frames
    ):
        """
        Wrapper from video-file reading.
        Object is iterable and calls to next(img) returns sequential frames
        :param path: Path to video file
        :param color_mode: Desired color mode of returned images
        :param size: Desired size of returned images
        :param as_img_objects: Returned image is Img object if True, numpy array if False
        """
        if not Path(path).exists():
            raise FileNotFoundError(f'{path} not found')
        if color_mode not in {'rgb', 'bgr'}:
            raise ValueError(f'Unexpected color mode. {color_mode} not in (rgb, bgr)')

        self.path = str(path)
        self._color_mode = color_mode
        self.as_img_objects = as_img_objects

        self._cap = cv2.VideoCapture(self.path)
        self._num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))

        self.output_size = size if size is not None else(self._h, self._w)

    def __next__(self) -> Union[Img, np.ndarray]:
        if self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                image = Img(frame, numpy_color_mode='bgr')
                if image.hw != self.output_size:
                    image.resize(*self.output_size)
                if self._color_mode == 'rgb':
                    _ = image.rgb
                elif self._color_mode == 'bgr':
                    _ = image.bgr
                else:
                    raise NotImplementedError('Only supports rgb and bgr')
                if self.as_img_objects:
                    return image
                return image.get()

        raise StopIteration

    def __iter__(self) -> "Vid":
        self._cap = cv2.VideoCapture(self.path)
        return self

    def __len__(self) -> int:
        return self._num_frames

    @property
    def fps(self):
        return self._fps

    @property
    def wh(self):
        return self._w, self._h

    @property
    def hw(self):
        return self._h, self._w

    def show(self, fs=False, fps=None):
        """
        Show video using cv2.
        Function uses preset colormode and image size.

        use 'q' to stop video

        :param fs: Show in fullscreen
        :param fps: overwrite video FPS for visualization
        """
        fps = self.fps if fps is None else fps
        wait_time = int(1000/fps)
        name = ''
        if fs:
            cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for image in self:
            cv2.imshow(name, image.get())
            if ord('q') == cv2.waitKey(wait_time):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    v = Vid('/home/martin/data/sevensix/gold_standard/casper_ruud_fh.mp4', size=(100, 100))

    for i in v:
        print(i.wh)