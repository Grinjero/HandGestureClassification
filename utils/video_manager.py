"""
From SlowFast https://github.com/facebookresearch/SlowFast
"""

import atexit
from collections import deque
import threading
import cv2


class SyncVideoManager:
    """
    Captures last num_frames * subsampling_rate frames from the source, used for inference.
    """
    def __init__(self,
                 source,
                 spatial_transforms,
                 sequence_length=32,
                 num_frames_skip=2):
        """
        :param source:
        :param num_frames=16:
        :param spatial_transforms:
            spatial transform that will be applied to frames for classification
        :param num_frames_skip=2:
            distance in frames between two classifications
        """
        self.source = source
        self.spatial_transforms = spatial_transforms

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.source))

        self.queue_size = sequence_length
        self.frame_queue = deque(maxlen=self.queue_size)
        self.frame_to_display = None
        self.clip_ready = False

        self.num_frames_skip = num_frames_skip
        # counts the number of frames since last clip read
        self.frame_counter = 0

    def capture_stream(self):
        """
        :return: was stream successfully read
        """

        was_read, frame = self.cap.read()
        if was_read:
            self.frame_to_display = frame
            self.frame_queue.append(self.spatial_transforms(frame))
            if self.frame_counter >= self.num_frames_skip:
                self.clip_ready = True

            self.frame_counter += 1

        return was_read

    def read_clip(self):
        if not self.clip_ready:
            return None

        self.frame_counter = 0
        self.clip_ready = False
        return list(self.frame_queue)

    def read_frame(self):
        return self.frame_to_display


class AsyncVideoManager:
    """
    Captures last num_frames * subsampling_rate frames from the source, used for inference.
    """

    def __init__(self,
                 source,
                 spatial_transforms,
                 sequence_length=32,
                 num_frames_skip=2):
        """
        :param source:
        :param num_frames=16:
        :param spatial_transforms:
            spatial transform that will be applied to frames for classification
        :param num_frames_skip=2:
            distance in frames between two classifications
        """
        self.source = source
        self.spatial_transforms = spatial_transforms

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.source))

        self.queue_size = sequence_length
        # if python documentation is to believe this is multithread safe
        self.frame_classification_queue = deque(maxlen=self.queue_size)
        self.frame_display_queue = deque(maxlen=self.queue_size)
        self.queue_lock = threading.Lock()
        self.stopped = False

        self.num_frames_skip = num_frames_skip
        # counts the number of frames since last clip read
        self.frame_counter = 0

        self.clip_ready = False
        self.frame_ready = False
        atexit.register(self.clean)

    def capture_stream(self):
        was_read = True
        while was_read and not self.stopped:
            was_read, frame = self.cap.read()
            if was_read:
                with self.queue_lock:
                    self.frame_classification_queue.append(self.spatial_transforms(frame))
                    self.frame_display_queue.appendleft(frame)
                    if self.frame_counter >= self.num_frames_skip:
                        self.clip_ready = True
                    self.frame_ready = True

                self.frame_counter += 1

        self.stop()
        print("Capturing stopped")

    def read_clip(self):
        with self.queue_lock:
            if not self.clip_ready:
                return not self.stopped, None

            self.frame_counter = 0
            self.clip_ready = False
            return not self.stopped, list(self.frame_classification_queue)

    def read_frame(self):
        with self.queue_lock:
            if not self.frame_ready:
                return not self.stopped, None

            self.frame_ready = False
            return not self.stopped, self.frame_display_queue[0]

    def start(self):
        print("Capturing started")
        self.capture_thread = threading.Thread(target=self.capture_stream, args=(), name="VidRead-Thread", daemon=True)
        self.capture_thread.start()

    def stop(self):
        self.stopped = True

    def clean(self):
        print("Cleaning capturing")
        self.stopped = True
        self.capture_thread.join()
        self.cap.release()

