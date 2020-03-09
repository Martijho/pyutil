from typing import Union

import tensorflow as tf
from pathlib import Path
import numpy as np

from object_detection.utils import label_map_util


def load_model(
        model_dir: Union[str, Path]
) -> "Detector":
    model_dir = Path(model_dir)

    assert model_dir.exists(), 'Model directory does not exist'
    assert model_dir.is_dir(), 'Path is not a directory'

    pb = model_dir / 'frozen_inference_graph.pb'
    pbtxts = list(model_dir.glob('*.pbtxt'))
    labelmap = str(pbtxts[0]) if len(pbtxts) > 0 else None

    return Detector(str(pb), labelmap)


class Detector:
    def __init__(
            self,
            pb_file: str,
            labelmap_file: Union[str, None] = None
    ):
        self._session = None
        self._detection_graph = None
        self.labels = None

        self._load_pb(pb_file)
        self._load_pbtxt(labelmap_file)

    def _load_pb(self, pb_file):
        self._detection_graph = tf.Graph()
        with self._detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(pb_file, 'rb') as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')

            self.input_shape_tensor = tf.shape('image_tensor:0', name='input_shape_tensor')

    def _load_pbtxt(self, pbtxt_file):
        if pbtxt_file is not None:
            self.labels = {
                int(pred_nr): label for label, pred_nr in label_map_util.get_label_map_dict(pbtxt_file).items()
            }

    def __enter__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._session = tf.Session(graph=self._detection_graph, config=config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()
        self._session = None

    def run(
            self,
            image: np.ndarray,
            output_raw: bool = False,
            coordinate_mode: str = 'absolute'
    ):
        if coordinate_mode.lower() == 'absolute':
            h, w = image.shape[:2]
        elif coordinate_mode.lower() == 'relative':
            h, w = 1, 1
        else:
            raise ValueError('Coordinate mode must be relative or absolute')

        image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
        output_tensors = [
            self._detection_graph.get_tensor_by_name(name) for name in [
                'detection_boxes:0',
                'detection_scores:0',
                'detection_classes:0',
                'num_detections:0',
                'input_shape_tensor:0'
            ]
        ]

        image = np.expand_dims(image, axis=0)

        raw_output = self._session.run(output_tensors, feed_dict={image_tensor: image})
        if output_raw:
            return raw_output

        boxes, confs, label_ids, n, input_shape = raw_output
        n = int(n[0])
        confs = confs[0]
        boxes = [[b[1] * w, b[0] * h, b[3] * w, b[2] * h] for b in boxes[0]]
        labels = [
            self.labels[lid] if self.labels is not None else str(lid)
            for lid in label_ids[0]
        ]

        detections = [
            {'box': box, 'confidence': conf, 'label': label}
            for box, conf, label in zip(boxes[:n], confs[:n], labels[:n])
        ]

        return detections
