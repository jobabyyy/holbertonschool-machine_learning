#!/usr/bin/env python3

"""
Continued...
Class Yolo that uses Yolo v3
algorithm to perform object
detection
"""

import tensorflow.keras as K
import numpy as np


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
                nnms_t: IOU threshold for non max suppression
                anchors: anchor for box info
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path) as file:
            class_names = file.read()
        self.class_names = class_names.replace("\n", "|").split("|")[:-1]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Attributes used:
                        outputs: list if numpy.ndarray containing
                        the predictions from the Darknet model
                        for a single image
                                        - grid height
                                        - grid width
                                        - anchor boxes
                                        - classes
                        images size: list is a numpy.ndarray containing
                        the image's orginal size [image heigh, & width}
        returns: tuple of 
                 - boxes
                 - box confidences
                 - box class probs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            grid_height, grid_width, anchor_boxes = output.shape[:3]

            processed_boxes = np.zeros_like(output[:, :, ::4])
            processed_box_confidences = output[:, :, :, 4:5]
            processed_box_class_probs = output[:, :, :, 5:]

            for row in range(grid_height):
                for col in range(grid_width):
                    for box in range(anchor_boxes):
                        # Extract the box coordinates and adjust them based on grid, anchor, and image size
                        t_x, t_y, t_w, t_h = output[row, col, box, :4]
                        box_x = (col + self.sigmoid(t_x)) / grid_width
                        box_y = (row + self.sigmoid(t_y)) / grid_height
                        box_w = (np.exp(t_w) * self.anchors[box][0]) / image_size[1]
                        box_h = (np.exp(t_h) * self.anchors[box][1]) / image_size[0]

                        x1 = (box_x - box_w / 2) * image_size[1]
                        y1 = (box_y - box_h / 2) * image_size[0]
                        x2 = (box_x + box_w / 2) * image_size[1]
                        y2 = (box_y + box_h / 2) * image_size[0]

                    processed_boxes[row, col, box, :] = np.array([x1, y1, x2, y2])

            boxes.append(processed_boxes)
            box_confidences.append(processed_box_confidences)
            box_class_probs.append(processed_box_class_probs)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))


        return (boxes, box_confidences, box_class_probs)
