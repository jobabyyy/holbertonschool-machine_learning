#!/usr/bin/env python3
"""Yolo Task 1"""

import numpy as np
from tensorflow import keras as K


class Yolo:
    """
    Yolo class uses algorithm Yolo v3 to complete
    object detection in images and videos.
    Objects are classified within a frame.
    The purpose of this class is to allow for user-friendly
    usage of YOLOv3 by encapsulating model loading,
    class info, and parameter config.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Adding expected parameters
           Args:
                model_path: path to pretained Yolo model
                classes_path: list of class names
                class_t: box score threshold for filtering
                nms_t: IOU threshold for non max suppression
                anchors: anchor for box info
        """
        self.model = K.models.load_model(model_path)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        with open(classes_path) as file:
            class_names = file.read()
        self.class_names = class_names.replace("\n", "|").split("|")[:-1]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors


def process_outputs(self, outputs, image_size):
    """
        Process the outputs from the Darknet model for a single image.
        :param outputs: list of numpy.ndarrays containing the predictions
        from the Darknet model for a single image
        :param image_size: numpy.ndarray containing the image’s original
        size [image_height, image_width]
        :return: tuple of (boxes, box_confidences, box_class_probs)
    """
    # Initialize lists to store boxes, box confidences
    # and box class probabilities
    boxes = []
    box_confidences = []
    box_class_probs = []
    # Loop through each output
    for i, output in enumerate(outputs):
        # Get the shape of the output
        grid_height, grid_width, anchor_boxes, _ = output.shape
        # Initialize an array of zeros to
        # store the box information
        box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
        # Get the box tx, ty, tw and th values from the output
        box_tx = output[..., 0]
        box_ty = output[..., 1]
        box_tw = output[..., 2]
        box_th = output[..., 3]
        # Get the pw and ph values from the anchors
        pw = self.anchors[i, :, 0]
        ph = self.anchors[i, :, 1]
        # Calculate the box x and y values using sigmoid and np.arange
        box_x = sigmoid(box_tx) + np.arange(
            grid_width).reshape(1, grid_width, 1)
        box_y = sigmoid(box_ty) + np.arange(
            grid_height).reshape(grid_height, 1, 1)
        # Calculate the box w and h values using pw, ph and np.exp
        box_w = pw * np.exp(box_tw)
        box_h = ph * np.exp(box_th)
        # Normalize the box x and y values by dividing
        # by grid_width and grid_height respectively
        box_x /= grid_width
        box_y /= grid_height
        # Normalize the box w and h values by dividing
        # by the model input shape
        box_w /= self.model.input.shape[1]
        box_h /= self.model.input.shape[2]
        # Calculate the x1, y1, x2 and y2 values using
        # the calculated x,y,w,h values and image_size
        x1 = (box_x - box_w / 2) * image_size[1]
        y1 = (box_y - box_h / 2) * image_size[0]
        x2 = (box_x + box_w / 2) * image_size[1]
        y2 = (box_y + box_h / 2) * image_size[0]
        # Store the calculated x1,y1,x2,y2 values in the box array
        box[..., 0] = x1
        box[..., 1] = y1
        box[..., 2] = x2
        box[..., 3] = y2
        # Append the calculated box to the boxes list
        boxes.append(box)
        # Calculate the confidence of each box using sigmoid
        box_conf = sigmoid(output[..., 4])
        # Append the reshaped confidence to the list of confidences
        box_confidences.append(
            box_confidence.reshape(
                grid_height, grid_width, anchor_boxes, 1))
        # Calculate class probabilities of each box using sigmoid
        box_class_prob = sigmoid(output[..., 5:])
        # Append the calculated class probabilities to
        # the list of class probabilities
        box_class_probs.append(box_class_prob)

    return boxes, box_confidences, box_class_probs


@staticmethod
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
