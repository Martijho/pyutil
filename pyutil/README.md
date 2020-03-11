# PyUtil

Current util functions

Requires tensorflow and [object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection) for some functionality

### pyutil.video
##### yield_video_frames
Takes a path to a video file and will yield frame number and video frames in tuples.
Example
```python
from pyutil.video import yield_video_frames
import cv2
for frame_nr, frame in yield_video_frames(
    'video_path.mp4',
    color_mode='bgr', 
    image_shape=(640, 480), 
    use_pbar=True
): 
    cv2.imshow('video', frame)
    cv2.waitKey(30)
cv2.destroyAllWindows()
```
##### get_video_stats
Takes a path to a video file and returns a tuple with`(number of image frames, (image height, image width), video FPS)`
##### class: VideoCreator
A small wrapper for cv2.VideoWriter. Mostly to help remember/simplify how its used. 
Use the object in a with-block or remember to call on .start() and .stop() methods.
example: 
```python 
from pyutil.video import VideoCreator
some_images = [...]
with VideoCreator('output_video.mp4', fps=30, image_shape=(500, 500), color_mode='gray') as vc:
for image in images:
    vc.add_frame(image)
```


### pyutil.image
##### get_random_color_map
Returns a defaultdict that provides unique rgb values for each label. Can be used to generate consistent colors for a set
of labels.
##### draw_bounding_box
Function that overlays a detection on an image. Current detection format only supports detections as a dict with the keys
"box", "label" and "confidence".
Example
```python
from pyutil.image import get_random_color_map, draw_bounding_box
from matplotlib import pyplot as plt

c = get_random_color_map()
image = ... # Some image
detections ... # List of detection-dicts

for det in detections: 
    image = draw_bounding_box(
        image, 
        det, 
        color=c[det['label']], 
        show_label=True, 
        show_confidence=True
    )
plt.imshow(image)
plt.show()
```
##### draw_keypoints
Function that overlays keypoints on an image. One of the optional parameter is "limbs" where the connections between points
can be provided. Both keypoints and limbs bust be iterables that yields two values. For keypoints these values are x and y coordinates
for a landmark, and for limbs these points are two indices for which points in keypoints that are to be connected. 
Example
```python
from pyutil.image import get_random_color_map, draw_keypoints
from matplotlib import pyplot as plt

c = get_random_color_map()
image = ... # Some image
keypoints = ... # List of keypoints
connections = ... # List of which keypoints to be connected

image = draw_keypoints(
    image, 
    keypoints, 
    limbs=connections, 
    color=c['points'], 
    limb_color=c['connection'], 
    limb_only=False
)
plt.imshow(image)
plt.show()
```
### pyutil.detection
##### clip
Clips coordinates between some boundries. Default assumption is that coordinates are relative and to be clipped between 0 and 1. 
If coordinates are absolute, provide the image width and height in the `w` and `h` optional parameters.
##### pad_box
Pads a bounding box with some percentage on all sides. Default padding is 0.1 which increase the height and width with 20% and the 
total box area with 44%. This function keeps the box aspect ratio
##### pad_box_to_aspect_ratio
Pads a bounding box to be a new aspect ratio, but while covering the original area. The function keeps the shortest side the same
and tries to pad the other side equally on both sides if possible. If not it pads as much it can on one side and the rest on the 
opposing side. 
##### save_detections
Takes a List of detection dicts and a path and dumps the detections as a .npy file. 
##### load_detections
Loads the detections from some given .npy file 
##### load_model
Loads a tensorflow object detection model from a model directory and returns a Detector object. The directory provided
must contain a `frozen_inference_graph.pb` file, and optionally a label map as a `.pbtxt` file. 
##### Class: Detector
###### NB: This class requires a tensorflow installation and a full setup of the [tensorflow object_detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 
A class wrapper for a tensorflow object detection model. The object loads a .pb and a .pbtxt file on initialization. 
Use objects of this class in a with-block and get inferences through the .run() method. 
Example of how inference can be done with this class and the load_model function
```python
from pyutil.detector import load_model

model_directory = '...' # Path to a tensorflow object detection model
image = ... # Some image

with load_model(model_directory) as detector: 
    detections = detector.run(
        image, 
        output_raw = False, 
        coordinate_mode = 'relative'
    )
```