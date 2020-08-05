# PyUtil

Current util functions

Requires tensorflow and [object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection) for some functionality

## Content
* [pyutil.video](README.md#pyutil.video)
    * [pyutil.video.yield_video_frames](README.md#yield_video_frames)
    * [pyutil.video.get_video_stats](README.md#get_video_stats)
    * [pyutil.video.VideoCreator](README.md#classvideocreator)
* [pyutil.image](README.md#image)
    * [pyutil.image.get_random_color_map](README.md#get_random_color_map)
    * [pyutil.image.draw_keypoints](README.md#draw_keypoints)
    * [pyutil.image.draw_bounding_box](README.md#draw_bounding_box)
* [pyutil.detection](README.md#detection)
    * [pyutil.detection.clip](README.md#clip)
    * [pyutil.detection.pad_box](README.md#pad_box)
    * [pyutil.detection.pad_box_to_aspect_ratio](README.md#pad_box_to_aspect_ratio)
    * [pyutil.detection.save_detections](README.md#save_detections)
    * [pyutil.detection.load_detections](README.md#load_detections)
    * [pyutil.detection.detector.load_model](README.md#load_model)
    * [pyutil.detection.detector.Detector](README.md#classdetector)
* [pyutil.keras](README.md#keras)
    * [pyutil.keras.run_lr_sweep](README.md#run_lr_sweep)
    * [pyutil.keras.callbacks.LRTensorBoard](README.md#classlrtensorboard)
    * [pyutil.keras.callbacks.CallbackWrapper](README.md#classcallbackWrapper)

### pyutil.video
##### yield_video_frames
Takes a path to a video file and will yield frame number and video frames in tuples.
Example yields every frame up to frame nr 100 from the video-file 'video_path.mp4'. Each image is 640 by 480 BGR
```python
from pyutil.video import yield_video_frames
import cv2
for frame_nr, frame in yield_video_frames(
    'video_path.mp4',
    color_mode='bgr', 
    image_shape=(640, 480), 
    use_pbar=True, 
    frame_window=(0, 100)
): 
    cv2.imshow('video', frame)
    cv2.waitKey(30)
cv2.destroyAllWindows()
```
##### get_video_stats
Takes a path to a video file and returns a tuple with`(number of image frames, (image height, image width), video FPS)`
##### Class:VideoCreator
A small wrapper for cv2.VideoWriter. Mostly to help remember/simplify how its used. 
Use the object in a with-block or remember to call on .start() and .stop() methods.
example: 
```python
from pyutil.video import VideoCreator
some_images = [...]
with VideoCreator('output_video.mp4', fps=30, image_shape=(500, 500), color_mode='gray') as vc:
    for image in some_images:
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
detections = [...] # List of detection-dicts

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
keypoints = [...] # List of keypoints
connections = [...] # List of which keypoints to be connected

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
##### Class:Detector
###### NB: This class requires a tensorflow installation and a full setup of the [tensorflow object_detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 
A class wrapper for a tensorflow object detection model. The object loads a .pb and a .pbtxt file on initialization. 
Use objects of this class in a with-block and get inferences through the .run() method. 
Example of how inference can be done with this class and the load_model function
```python
from pyutil.detection.detector import load_model

model_directory = '...' # Path to a tensorflow object detection model
image = ... # Some image

with load_model(model_directory) as detector: 
    detections = detector.run(
        image, 
        output_raw = False, 
        coordinate_mode = 'relative'
    )
```

### pyutil.keras
##### Class:LRTensorBoard
###### NB: This class assumes existing install of keras or tensorflow version > 2.0.0  

Extends keras callback. Replaces the native TensorBoard-callback. 
Performes the same logging as TensorBoard, but includes learning rate in tensorboard-graphs
##### Class:CallbackWrapper
###### NB: This class assumes existing install of keras or tensorflow version > 2.0.0  
Wrapper object for calls that are to be made as part of a keras training pipeline.

To use this functionality, provide this callback-wrapper with callable functions with the 
same arguments as the function they are ment to replace.
The wrapper will replace all methods in the callback that matches the named argument. 

Input-arguments to callable function must match those provided to the matching keras-callback function.
Example of how to wrap a function in a CallbackWrapper
```python
from pyutil.keras.callbacks import CallbackWrapper

def wrapped_on_epoch_end_function(epoch, logs=None):
    print(f'Epoch {epoch} ended')

model.fit(
    ...,
    callbacks=[
        CallbackWrapper(
            on_epoch_end=wrapped_on_epoch_end_function, 
            on_training_begin=lambda logs=None: print('Queue Rocky-theme')
        )
    ]
)
```

####### Learning rate sweep
##### run_lr_sweep
This function runs a learning rate sweep on the provided model and creates
loss-lr plots for all metrics. 
```
run_lr_sweep(
    model,
    training_generator,
    lr_start=1e-6,
    lr_stop=1e2,
    steps=1000,
    mode='exponential',
    cutoff=1e3
)
```