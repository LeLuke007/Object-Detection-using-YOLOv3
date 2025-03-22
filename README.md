# Object Detection using YOLOv3

This project demonstrates object detection using the YOLOv3 (You Only Look Once) algorithm with OpenCV and Python. The model is capable of detecting various objects in images and drawing bounding boxes around them.

## Prerequisites

- Python 3.x
- OpenCV
- NumPy

You can install the required packages using pip:

```sh
pip install opencv-python numpy
```

## Download YOLOv3 Model

Download the YOLOv3 model weights from the following link and place it in models directory as `models/yolov3.weights` :

[YOLOv3 Weights](https://github.com/patrick013/Object-Detection---Yolov3/raw/refs/heads/master/model/yolov3.weights)

## Usage

1. Place the images you want to process in the `images/` directory.
2. Run the `main.py` script:
3. The script will process each image, detect objects, and display the images with bounding boxes. The labels and bounding box coordinates will be saved in the `labels/` directory.

## Example

If an image contains a person, the output might look like this:

```
Person : 50 100 200 300
```

This indicates that a person was detected with a bounding box from (50, 100) to (200, 300).
