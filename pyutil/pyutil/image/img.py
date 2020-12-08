import re
import warnings
from typing import Union, Tuple

import cv2
import magic
import numpy as np
from pathlib import Path


class Img:
    def __init__(
            self,
            data: Union[str, Path, np.ndarray],
            lazy: bool = False,
            numpy_color_mode: str = 'rgb'
    ):
        """
        QOL class for handling images.
        :param data: Path to image file or numpy-matrix
        :param lazy: Skip loading image file at initialization. Only reads meta data
        :param numpy_color_mode: If data is a numpy-matrix, set this as the color mode rgb|bgr
        """
        if type(data) == np.ndarray:
            self.path = 'NOT DEFINED'
            self.image = np.array(data)
            self.h, w = self.image.shape[:2]
            self.color_mode = numpy_color_mode
            self.lazy = lazy
        else:
            if not Path(data).exists():
                raise FileNotFoundError(f'{data} not found')

            self.path = str(data)
            self.lazy = lazy

            if not lazy:
                self._load()
            else:
                self.image = None
                self.color_mode = None
                t = magic.from_file(self.path).split('baseline')[-1]
                self.w, self.h = re.search('(\d+) ?x ?(\d+)', t).groups()
                self.w, self.h = int(self.w), int(self.h)

    def _load(self):
        """
        Loads image into memory using cv2.
        """
        self.image = cv2.imread(self.path)
        self.h, self.w = self.image.shape[:2]
        self.color_mode = 'bgr'

    def _cond_warn_and_load(self):
        if self.image is None:
            warnings.warn('Forced image loading. Init Img-object with lazy=False')
            self._load()

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Gets image numpy-shape. If lazy Img object, the image is loaded first
        :return: Image shape
        """
        self._cond_warn_and_load()
        return self.image.shape

    @property
    def hw(self) -> Tuple[int, int]:
        """
        Gets image height and width
        :return: (image height, image width)
        """
        return self.h, self.w

    @property
    def wh(self) -> Tuple[int, int]:
        """
        Gets image width and height
        :return: (image width, image height)
        """
        return self.w, self.h

    @property
    def color(self) -> str:
        """
        Gets image color mode as a string
        :return: color mode
        """
        return self.color_mode

    @property
    def rgb(self) -> np.ndarray:
        self._cond_warn_and_load()
        if self.color_mode == 'rgb':
            pass
        elif self.color_mode == 'bgr':
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.color_mode = 'rgb'
        else:
            raise NotImplementedError('Cannot change colormode to rgb from anything other than rgb|bgr')
        return self.image

    @property
    def bgr(self) -> np.ndarray:
        self._cond_warn_and_load()
        if self.color_mode == 'bgr':
            pass
        elif self.color_mode == 'rgb':
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            self.color_mode = 'bgr'
        else:
            raise NotImplementedError(
                f'Cannot change colormode to bgr from anything other than rgb|bgr ({self.color_mode})'
            )
        return self.image

    def resize(self, *args, **kwargs):
        """
        Resizes image to given shape inplace.
        Shapes can be provided in the following formats:
            - As named arguments (h and w, H and W, height and width)
            - As 2 unnamed arguments (height, width)
            - As 1 unnamed argument (scale)

        Integers will be treated as desired pixel size, while floats will be used to scale one or both dimensions

        :param args: 0, 1 or 2 unnamed arguments
        :param kwargs:  0, 1 or 2 named arguments
        """
        if len(args) > 0:
            assert len(kwargs) == 0, 'Only named or unnamed arguments, not both'
        if len(kwargs) > 0:
            assert len(args) == 0, 'Only named or unnamed arguments, not both'
        assert len(kwargs) <= 2 or len(args) <= 2, 'Ambiguous new image shape'

        h, w = None, None

        for kw in ('h', 'H', 'height'):
            h = kwargs[kw] if kw in kwargs else h
        for kw in ('w', 'W', 'width'):
            w = kwargs[kw] if kw in kwargs else w

        if h is None or w is None:
            if len(args) == 2:
                h, w = args
            elif len(args) == 1:
                h, w = args[0], args[0]

        pixel_h = int(self.h * h) if type(h) == float else h
        pixel_w = int(self.w * w) if type(w) == float else w

        pixel_h = self.h if pixel_h == 1 or pixel_h is None else pixel_h
        pixel_w = self.w if pixel_w == 1 or pixel_w is None else pixel_w

        self.image = cv2.resize(self.image, (pixel_w, pixel_h))

    def show(
            self,
            fs: bool = False
    ):
        """
        Shows image using cv2
        :param fs: Show image in fullscreen
        """
        image = self.bgr
        name = ''
        if fs:
            cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(name, image)
        cv2.waitKey()
        cv2.destroyAllWindows()
