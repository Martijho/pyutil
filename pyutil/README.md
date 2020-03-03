# PyUtil

Current util functions

##### yield_video_frames
Takes a path to a video file and will yield frame number and video frames in tuples.
Example
```python
from pyutil import yield_video_frames

for frame_nr, frame in yield_video_frames('video_path.mp4'): 
    # Do some stuff
```

 