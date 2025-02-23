import cv2 as open_cv
import numpy as np
import logging
from drawing_utils import draw_contours
from colors import COLOR_GREEN, COLOR_WHITE, COLOR_BLUE

class MotionDetector:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1

    def __init__(self, video, coordinates, start_frame):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []
        self.capture = open_cv.VideoCapture(video)
        self.capture.set(open_cv.CAP_PROP_POS_FRAMES, start_frame)
        self.spots_status = ['available'] * len(coordinates)

    def detect_motion(self):
        ret, frame = self.capture.read()
        if not ret:
            return None, None

        if not self.contours:
            self._initialize_contours()

        blurred = open_cv.GaussianBlur(frame.copy(), (5, 5), 3)
        grayed = open_cv.cvtColor(blurred, open_cv.COLOR_BGR2GRAY)
        new_frame = frame.copy()

        for index, p in enumerate(self.coordinates_data):
            status = self.__apply(grayed, index, p)
            self.spots_status[index] = 'available' if status else 'occupied'

            coordinates = self._coordinates(p)
            color = COLOR_GREEN if status else COLOR_BLUE
            draw_contours(new_frame, coordinates, str(p["id"] + 1), COLOR_WHITE, color)

        return new_frame, self.spots_status

    def _initialize_contours(self):
        for p in self.coordinates_data:
            coordinates = self._coordinates(p)
            rect = open_cv.boundingRect(coordinates)

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] = coordinates[:, 0] - rect[0]
            new_coordinates[:, 1] = coordinates[:, 1] - rect[1]

            self.contours.append(coordinates)
            self.bounds.append(rect)

            mask = open_cv.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),
                [new_coordinates],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=open_cv.LINE_8)

            mask = mask == 255
            self.mask.append(mask)

    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)
        rect = self.bounds[index]
        roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        laplacian = open_cv.Laplacian(roi_gray, open_cv.CV_64F)
        
        coordinates[:, 0] = coordinates[:, 0] - rect[0]
        coordinates[:, 1] = coordinates[:, 1] - rect[1]

        status = np.mean(np.abs(laplacian * self.mask[index])) < MotionDetector.LAPLACIAN
        return status

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])

class CaptureReadError(Exception):
    pass