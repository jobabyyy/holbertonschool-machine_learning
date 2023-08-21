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
        """process output"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            grid_height, grid_width, anchor_boxes, _ = output.shape
            num_classes = output.shape[-1] - 5

            box = output[..., :4]  # Extracting box coordinates and dimensions
            box_conf = output[..., 4:5]  # Extracting box confidence
            box_class_probs = output[..., 5:]  # Extracting class probabilities

            # Applying sigmoid function to box_conf and box_class_probs
            box_conf = self.sigmoid(box_conf)
            box_class_probs = self.sigmoid(box_class_probs)

            grid_x = np.arange(grid_width)
            grid_y = np.arange(grid_height)

            # Creating grid coordinates for x and y
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)

            # Reshaping for calculations
            grid_x = grid_x.reshape((-1, 1))
            grid_y = grid_y.reshape((-1, 1))

            # Box adjustments using sigmoid and anchors
            box[:, :, :, 0] = (box[:, :, :, 0] + grid_x) / grid_width
            box[:, :, :, 1] = (box[:, :, :, 1] + grid_y) / grid_height
            box[:, :, :, 2] = np.exp(box[:, :, :, 2]) * self.anchors[:, 0]
            box[:, :, :, 3] = np.exp(box[:, :, :, 3]) * self.anchors[:, 1]

            # Converting box coordinates to image coordinates
            box[:, :, :, 0:2] *= image_size[::-1]
            box[:, :, :, 2:4] *= image_size[::-1]

            # Stacking results
            boxes.append(box)
            box_confidences.append(box_conf)
            box_class_probs.append(box_class_probs)

        return boxes, box_confidences, box_class_probs

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
