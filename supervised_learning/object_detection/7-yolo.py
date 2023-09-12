#!/usr/bin/env python3
"""Final Part: Class Yolo continuation...
"""

import numpy as np
from tensorflow import keras as K
import numpy as np
import cv2
import os


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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on confidence scores and class probabilities.

        Args:
            boxes (list): List of numpy.ndarrays containing box
            coordinates for each output.
        box_confidences (list):
            List of numpy.ndarrays containing
            box confidences for each output.
        box_class_probs (list):
            List of numpy.ndarrays containing box
            class probabilities for each output.
        Returns:
        filtered_boxes (numpy.ndarray):
            Array containing filtered box
            coordinates after applying confidence and class filters.
        box_classes (numpy.ndarray):
            Array containing class labels
            corresponding to the filtered boxes.
        box_scores (numpy.ndarray):
            Array containing confidence scores
            corresponding to the filtered boxes.
        """
        filtered_boxes = None
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            cur_box_score = box_confidences[i] * box_class_probs[i]
            cur_box_class = np.argmax(cur_box_score, axis=-1)
            cur_box_score = np.max(cur_box_score, axis=-1)
            mask = cur_box_score >= self.class_t

            if filtered_boxes is None:
                filtered_boxes = boxes[i][mask]
            else:
                filtered_boxes = np.concatenate((filtered_boxes,
                                                 boxes[i][mask]),
                                                axis=0)

            box_classes.append(cur_box_class[mask])
            box_scores.append(cur_box_score[mask])

        # Convert the lists to arrays
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        # Remove low-confidence boxes
        rm = box_scores < self.class_t
        filtered_boxes = filtered_boxes[~rm]
        box_classes = box_classes[~rm]
        box_scores = box_scores[~rm]

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression to filtered bounding boxes.
        Args:
        filtered_boxes:
            a numpy.ndarray of shape (?, 4)
            ontaining all of the filtered bounding boxes
        box_classes:
            a numpy.ndarray of shape (?,) containing the
            class number for the class that filtered_boxes
        p   redicts, respectively
        box_scores:
            a numpy.ndarray of shape (?) containing the box
            scores for each box in filtered_boxes, respectively
        Returns:
            Tuple of (box_predictions, predicted_box_classes
            predicted_box_scores):
        box_predictions:
            a numpy.ndarray of shape (?, 4)
            containing all of the predicted bounding boxes ordered
            by class and box score
        predicted_box_classes:
            a numpy.ndarray of shape (?,) containing the class number
            for box_predictions ordered by class and box score
        predicted_box_scores:
            a numpy.ndarray of shape (?) containing the box scores for
            box_predictions ordered by class and box score
        """
        idx = np.lexsort((-box_scores, box_classes))
        sorted_box_pred = filtered_boxes[idx]
        sorted_box_class = box_classes[idx]
        sorted_box_scores = box_scores[idx]
        _, counts = np.unique(sorted_box_class, return_counts=True)

        kept_indices = []
        n = 0
        for count in counts:
            max_score_idx = np.argmax(sorted_box_scores[n:n + count])
            kept_indices.append(n + max_score_idx)
            n += count

        return (sorted_box_pred[kept_indices],
                sorted_box_class[kept_indices],
                sorted_box_scores[kept_indices])

    @staticmethod
    def load_images(folder_path):
        """Loads images from a folder.

        Args:
            folder_path (str): The path to the folder holding the images.

        Returns:
            tuple: A tuple containing:
                - A list of images as numpy.ndarrays.
                - A list of paths to the individual images.
        """
        image_files = [f for f in os.listdir(folder_path) if
                       os.path.isfile(os.path.join(folder_path, f)) and
                       f.endswith(('.png', '.jpg', '.jpeg'))]
        image_paths = [os.path.join(folder_path,
                       image_file) for image_file in image_files]

        images = [cv2.cvtColor(cv2.imread(image_path),
                  cv2.COLOR_BGR2RGB) for image_path in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """preprocessing"""
        # Assuming that the model's input dimensions are class attributes
        input_h, input_w = self.model.input.shape[1], self.model.input.shape[2]

        # Placeholder for processed images
        pimages_list = []
        shapes_list = []

        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]

        for image in images:
            # Capture and store the original shape of each image
            orig_shape = image.shape[:2]
            shapes_list.append(orig_shape)

            # Resize and normalize the image
            processed_img = cv2.resize(image, (input_w, input_h),
                                       interpolation=cv2.INTER_CUBIC)
            processed_img = processed_img / 255.0
            pimages_list.append(processed_img)

        # Convert the lists to numpy arrays for return
        pimages = np.stack(pimages_list, axis=0)
        image_shapes = np.array(shapes_list)

        return pimages, image_shapes


def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
    """
    Displays the image with boundary boxes,
    class names, and box scores.
    :param image: a numpy.ndarray containing an
    unprocessed image
    :param boxes: a numpy.ndarray containing the boundary
    boxes for the image
    :param box_classes: a numpy.ndarray containing the
    class indices for each box
    :param box_scores: a numpy.ndarray containing the
    box scores for each box
    :param file_name: the file path where the original
    image is stored
    """

    # Iterate through the boxes
    for box, cls, score in zip(boxes, box_classes, box_scores):
        # Extracting coordinates of the box
        x, y, w, h = box

        # Drawing the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Creating the label for each box (class + score)
        label = '{}: {:.2f}'.format(cls, score)

        # Getting the width and height of the text box
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, 1)

        # Drawing a filled rectangle (background of the text)
        image = cv2.rectangle(image, (x, y - 20), (x + tw,
                              y), (255, 0, 0), -1)

        # Putting the label text on the image
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Show the image in a window named as file_name
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        # Check if 's' key is pressed
        if key == ord('s'):
            if not os.path.exists("detections"):
                os.makedirs("detections")
            cv2.imwrite(os.path.join("detections",
                                     file_name.split("/")[-1]), image)

        cv2.destroyAllWindows()


def predict(self, folder_path):
    image_paths = [os.path.join(folder_path, fname) for fname in
                   os.listdir(folder_path) if fname.endswith(('.png',
                                                              '.jpg',
                                                              '.jpeg'))]

    predictions = []

    for image_path in image_paths:
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        pimage, _ = self.preprocess_images([image])

        # performs the prediction and returns boxes, classes, and scores.
        boxes, box_classes,
        box_scores = self.your_prediction_method(pimage)

        predictions.append((boxes, box_classes, box_scores))

        file_name = os.path.basename(image_path)
        self.show_boxes(image, boxes, box_classes, box_scores, file_name)

    return predictions, image_paths