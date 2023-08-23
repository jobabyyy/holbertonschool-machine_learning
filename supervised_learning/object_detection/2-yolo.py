#!/usr/bin/env python3
"""Pt.2: Class Yolo continuation...
Filter boxes applied."""

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
            outputs: List of numpy.ndarrays containing the darknet
            model predictions from multiple feature maps.
            image_size: Tuple containing image size as (height, width).
        Returns:
            boxes: List of box coordinates.
            box_confidences: List of box confidences.
            box_class_probs: List of box class probabilities.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes based on their confidence and class probabilities.

        Arguments:
        boxes: list of numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes, 4)
        containing the processed boundary boxes for each output, respectively
        box_confidences: numpy.ndarrays of shape (grid_height,
        grid_width, anchor_boxes
        box_class_probs:numpy.ndarrays of shape (grid_height, grid_width,
        anchor_boxes, classes) containing the processed box class
        probabilities for each output, respectively

        Returns:
        filtered_boxes: a numpy.ndarray containing all of the
        filtered bounding boxes
        box_classes: a numpy.ndarray of shape containing the class number
        that each box in filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape containing the box scores
        for each box in filtered_boxes, respectively
        """
        # Compute box scores by x box confidences w/ box class probabilities
        box_scores = []
        for i in range(len(boxes)):
            box_scores.append(box_confidences[i] * box_class_probs[i])

        # Find the index of the highest score for each box
        box_classes = [np.argmax(box_score, axis=-1)
                       for box_score in box_scores]
        # Find the value of the highest score for each box
        box_class_scores = [np.max(box_score, axis=-1)
                            for box_score in box_scores]

        # Create a mask to filter out boxes with low scores
        mask = [box_class_score >= self.class_t
                for box_class_score in box_class_scores]

        # Apply mask to boxes & scores to keep only the highscoring boxes
        filtered_boxes = [box[mask[i]] for i, box in enumerate(boxes)]
        filtered_box_classes = [box_class[mask[i]] for i,
                                box_class in enumerate(box_classes)]
        filtered_box_scores = [box_class_score[mask[i]] for i,
                               box_class_score in enumerate(box_class_scores)]

        # Concatenate all the filtered boxes from all outputs into one list
        filtered_boxes = np.concatenate(filtered_boxes)
        filtered_box_classes = np.concatenate(filtered_box_classes)
        filtered_box_scores = np.concatenate(filtered_box_scores)

        return (filtered_boxes, filtered_box_classes, filtered_box_scores)
