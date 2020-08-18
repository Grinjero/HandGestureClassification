import numpy as np
import time
import os
import cv2
from collections import deque

import torch

def median_filtering(queue):
    return np.median(queue, axis=0)


class Queue:
    def __init__(self, max_size, element_shape, is_tensor=False):
        self.queue = deque(maxlen=max_size)
        self.max_size = max_size

        # for i in range(max_size):
        #     filler = np.zeros(element_shape, dtype=float)
        #     if is_tensor:
        #         self.queue.append(torch.Tensor(filler))
        #     else:
        #         self.queue.append(filler)

    def __iter__(self):
        for element in self.queue:
            yield element

    # Adding elements to queue
    def enqueue(self, data):
        self.queue.append(data)
        assert len(self.queue) <= self.max_size
        return True

    # Removing the last element from the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        return ("Queue Empty!")

    def to_array(self):
        return np.array(self.queue)

    def to_tuple(self):
        return list(self.queue)
    # Average
    def mean_filtering(self):
        return self.to_array().mean(axis=0)

    # Median
    def median_filtering(self):
        return np.median(self.to_array(), axis=0)

    # Exponential average
    def exponential_average_filtering(self):
        weights = np.exp(np.linspace(-1., 0., self.max_size))
        weights /= weights.sum()
        average = weights.reshape(1, self.max_size).dot(self.to_array())
        return average.reshape(average.shape[1], )


class FPSMeasurer:

    def __init__(self):
        self._avg_duration = 0
        self._fps_start = 0
        self._avg_fps = 0
        self._fps_1sec = 0

        self.op_start_time = time.monotonic_ns()

    def operation_complete(self):
        time_now = time.monotonic_ns()

        op_duration = time_now - self.op_start_time
        self.update_avg_duration(op_duration)
        self.update_avg_fps()

    def update_avg_duration(self, op_duration):
        self._avg_duration = 0.98 * self._avg_duration + 0.02 * op_duration

    def update_avg_fps(self):
        time_now = time.monotonic_ns()
        if (time_now - self._fps_start) > 1000000000:
            self._fps_start = time_now
            self._avg_fps = 0.7 * self._avg_fps + 0.3 * self._fps_1sec
            self._fps_1sec = 0

        self._fps_1sec += 1

    def avg_duration(self):
        return self._avg_duration

    def fps(self):
        return self._avg_fps

class ImageStreamer:
    def __init__(self, image_directory, fps=30):
        self.image_index = 0
        self.image_list = os.listdir(image_directory)
        self.image_list = [os.path.join(image_directory, image) for image in self.image_list]
        self.video_size = len(self.image_list)
        self.fps = fps

    def isOpened(self):
        return self.image_index < self.video_size

    def read(self):
        if self.image_index >= self.video_size:
            return False, None

        image = cv2.imread(self.image_list[self.image_index])
        self.image_index += 1

        return True, image

    def release(self):
        # just so it has the safe interface as cv2.VideoCapture
        pass

class Dummy(object):
    pass
