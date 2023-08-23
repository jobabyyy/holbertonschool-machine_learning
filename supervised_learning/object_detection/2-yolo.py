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
        Filters YOLO boxes based on object confidence and class probability.

        Args:
            boxes: List of numpy.ndarrays containing processed boundary boxes.
            box_confidences: numpy.ndarrays w/ processed box confidences.
            box_class_probs: numpy.ndarrays w/ processed box
            class probabilities
        Returns:
            tuple: A tuple of filtered bounding boxes,
            predicted classes, box scores.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_probs in zip(boxes, box_confidences,
                                                box_class_probs):
            # Flatten the arrays for easier manipulation
            box = box.reshape(-1, 4)
            confidence = confidence.reshape(-1)
            class_probs = class_probs.reshape(-1, len(self.class_names))

            # Get indices of boxes w/ confidence greater than threshold
            confidence_mask = confidence >= self.class_t
            boxes_above_confidence = box[confidence_mask]
            class_probs_above_confidence = class_probs[confidence_mask]
            confidence_above_confidence = confidence[confidence_mask]

            # Multiply confidence with class probabilities to get box scores
            box_scores_above_confidence = (
                confidence_above_confidence *
                class_probs_above_confidence.max(axis=1)
            )

            # Get indices of boxes that contribute to max class probability
            class_indices = class_probs_above_confidence.argmax(axis=1)

            filtered_boxes.append(boxes_above_confidence)
            box_classes.append(class_indices)
            box_scores.append(box_scores_above_confidence)

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
