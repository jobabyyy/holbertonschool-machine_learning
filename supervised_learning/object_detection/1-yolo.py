#!/usr/bin/env python3
"""
Class Yolo that uses Yolo v3
algorithm to perform object
detection
This class aims to create a base
structure that will serve as the
foundation for the following tasks.
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
                nms_t: IOU threshold for non max suppression
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
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            # Box coordinates adjustment
            box_tx = output[..., 0:1]
            box_ty = output[..., 1:2]
            box_tw = output[..., 2:3]
            box_th = output[..., 3:4]
            # Get the anchors
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            # Calculate the real coordinates
            bx = self.sigmoid(box_tx) + np.arange(
                              grid_width).reshape(1, grid_width, 1)
            by = self.sigmoid(box_ty) + np.arange(
                              grid_height).reshape(grid_height, 1, 1)
            bw = pw * np.exp(box_tw)
            bh = ph * np.exp(box_th)
            # Normalize the coordinates
            bx /= grid_width
            by /= grid_height
            bw /= self.model.input.shape[1]
            bh /= self.model.input.shape[2]
            # Calculate the coordinates relative to the image size
            x1 = (bx - bw / 2) * image_size[1]
            y1 = (by - bh / 2) * image_size[0]
            x2 = (bx + bw / 2) * image_size[1]
            y2 = (by + bh / 2) * image_size[0]
            # Update the box with the new coordinates
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)
            # Get the confidences and class probabilities
            box_confidence = self.sigmoid(output[..., 4])
            box_confidences.append(
                box_confidence.reshape(
                    grid_height, grid_width, anchor_boxes, 1))
            box_class_prob = self.sigmoid(output[..., 5:])
            box_class_probs.append(box_class_prob)
        return boxes, box_confidences, box_class_probs

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
