#!/usr/bin/env python3
"""Yolo Task 1"""

import numpy as np
from tensorflow import keras as K


class Yolo:
    """
    Yolo class uses the Yolo v3 algorithm to complete
    object detection in images and videos.
    Objects are classified within a frame.
    The purpose of this class is to allow for user-friendly
    usage of YOLOv3 by encapsulating model loading,
    class info, and parameter config.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo instance.

        Args:
            model_path (str): Path to pretrained Yolo model.
            classes_path (str): List of class names.
            class_t (float): Box score threshold for filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (np.array): Anchor boxes.
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
        """Processes the model outputs to obtain box coordinates,
        confidences, and class probabilities.
        Args:
            outputs (list): List of numpy.ndarrays containing the darknet
            model predictions from multiple feature maps.
            image_size (tuple): Tuple containing image size as (height, width).
        Returns:
            boxes (list): List of box coordinates.
            box_confidences (list): List of box confidences.
            box_class_probs (list): List of box class probabilities.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]

            # Extract box parameters
            box_tx, box_ty, box_tw, box_th = (output[..., 0],
                                              output[..., 1],
                                              output[..., 2],
                                              output[..., 3])

            # Apply sigmoid function to t_x, t_y, and box_confidence
            box_tx_sigmoid = self.sigmoid(box_tx)
            box_ty_sigmoid = self.sigmoid(box_ty)

            # Generate a grid of the same shape as the predictions
            grid_x = np.arange(grid_width).reshape(1, grid_width, 1)
            grid_y = np.arange(grid_height).reshape(1, grid_height, 1)

            box_x = (box_tx_sigmoid + grid_x) / grid_width
            box_y = (box_ty_sigmoid + grid_y) / grid_height

            box_w = (self.anchors[i, :, 0] * np.exp(box_tw)) / input_width
            box_h = (self.anchors[i, :, 1] * np.exp(box_th)) / input_height

            # Convert box coordinates relative to the size of the image
            x1 = (box_x - box_w / 2) * image_size[1]
            y1 = (box_y - box_h / 2) * image_size[0]
            x2 = (box_x + box_w / 2) * image_size[1]
            y2 = (box_y + box_h / 2) * image_size[0]

            # Append box coordinates
            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
