# Road Lane Line Detection System

This project implements a road lane line detection system using computer vision techniques with OpenCV and NumPy in Python.

## Features

- Processes video frames to detect lane lines on the road.
- Applies grayscale conversion, Gaussian blur, and Canny edge detection.
- Masks the region of interest to focus on the road area.
- Uses perspective transform to get a bird's eye view of the lanes.
- Detects lane pixels and fits a second-order polynomial to each lane line.
- Calculates vehicle offset from lane center and provides directional feedback.
- Displays the lane detection results on the video frames.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Usage

1. Place your test video file in the project directory or update the video path in `road_lane_detection.py`.
2. Run the script:

```bash
python road_lane_detection.py
```

3. The processed video with lane detection will be displayed in a window.
4. Press 'q' to quit the video display.

## Files

- `road_lane_detection.py`: Main script containing the lane detection implementation.
- `test_video.mp4`: Sample test video used for lane detection.
- `test_video_access.py`: (If applicable) Additional test scripts.
- `README.md`: Project documentation.

## License

This project is open source and available under the MIT License.
