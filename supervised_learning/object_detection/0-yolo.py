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