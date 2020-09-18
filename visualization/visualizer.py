import time
import atexit
import threading

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def get_class_color(color_map, class_id, num_classes):
    """
    Get color for a class id.
    Args:
        class_id (int): class id.
    """
    rgb = color_map(class_id / num_classes)[:3]
    bgr = int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)
    return bgr


class SyncVideoVisualizer:
    def __init__(self, image_visualizers, display_spatial_transforms=None, output_file=None):
        """
        :param image_visualizers: List of image visualizers that will draw on the input image (be careful that they
        don't draw over each other)
        :param display_spatial_transforms: Spatial transform or a Compose
        """
        self.image_visualizers = image_visualizers
        self.display_spatial_transforms = display_spatial_transforms

    def display(self, frame):
        """
        :return: False if user pressed q to shut down
        """
        if self.display_spatial_transforms:
            frame = self.display_spatial_transforms(frame)

        self._draw_visualizers(frame)

        cv2.imshow("GestureClassification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return frame

    def _draw_visualizers(self, frame):
        for image_visualizer in self.image_visualizers:
            image_visualizer.draw_frame(frame)
        return frame

    def update_state(self, state_dict):
        for image_visualizer in self.image_visualizers:
            image_visualizer.update_state(state_dict)


class AsyncVideoVisualizer:
    def __init__(self, video_capturer, image_visualizers):
        """
        :param video_capturer:
        :param image_visualizers: List of image visualizers that will draw on the input image (be careful that they
        don't draw over each other)
        """
        self.video_capturer = video_capturer
        self.image_visualizers = image_visualizers

        self.stop = False
        atexit.register(self.clean)

    def visualize_stream(self):
        more_to_come = True
        while more_to_come and not self.stop:
            more_to_come, frame = self.video_capturer.read_frame()

            if frame is None:
                # give the model or frame capturer some time
                time.sleep(0.01)
                continue

            frame = self._draw_visualizers(frame)
            cv2.imshow("GestureClassification", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Visualization stopped")
        self.clean()

    def _draw_visualizers(self, frame):
        for image_visualizer in self.image_visualizers:
            image_visualizer.draw_frame(frame)
        return frame

    def start(self):
        print("Visualization started")
        self.display_thread = threading.Thread(target=self.visualize_stream, args=(), name="VidRead-Display", daemon=True)
        self.display_thread.start()

    def update_state(self, state_dict):
        for image_visualizer in self.image_visualizers:
            image_visualizer.update_state(state_dict)

    def clean(self):
        print("Cleaning visualization")
        self.stop = True
        self.video_capturer.stop()
        cv2.destroyAllWindows()


class ImageVisualizer:
    def draw_frame(self, frame):
        pass

    def update_state(self, state_dict):
        pass


class FPSVisualizer(ImageVisualizer):
    def __init__(self, color=(255, 0, 0), alpha=0.6, y_position=10, fps_name="FPS"):
        self.color = color
        self.alpha = alpha
        self.y_position = y_position
        # self.color = color[0], color[1], color[2]
        self.current_fps = None
        self.fps_name = fps_name

    def update_state(self, state_dict):
        pass

    def update_fps(self, fps):
        self.current_fps = fps

    def draw_frame(self, frame):
        if self.current_fps is None:
            return frame

        h, w, c = frame.shape
        x = w - 5
        text = "{}: {:.3f}".format(self.fps_name, self.current_fps)
        (label_width, label_height), baseline = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                                                                thickness=2)
        x -= label_width
        cv2.putText(frame, text, (x, self.y_position), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=self.color, thickness=2)


class TopKVisualizer(ImageVisualizer):
    def __init__(self, class_map, top_k=1, colormap="rainbow", alpha=0.6, start_position=(10, 20)):
        super().__init__()
        self.num_classes = len(class_map.keys())
        self.class_map = class_map
        self.color_map = plt.get_cmap(colormap)
        self.top_k = top_k
        self.alpha = alpha
        self.start_position=start_position

        self.current_preds = None

    def update_state(self, state_dict):
        predictions = state_dict["predictions"]
        if predictions.ndim != 1:
            predictions = torch.squeeze(predictions)

        self.current_preds = predictions

    def draw_frame(self, frame):
        if self.current_preds is None:
            return frame

        top_scores, top_classes = torch.topk(self.current_preds, k=self.top_k)
        top_scores, top_classes = top_scores.tolist(), top_classes.tolist()

        x, y = self.start_position
        for i in range(self.top_k):
            color = get_class_color(self.color_map, top_classes[i], self.num_classes)
            class_name = self.class_map[top_classes[i]]
            text = "{} {:.6f}".format(class_name, top_scores[i])

            cv2.putText(frame, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=color, thickness=2)

            y += 30


class ClassifiedClassVisualizer(ImageVisualizer):
    def __init__(self, class_map, label_display_time=1.5, colormap="rainbow", y_position=30):
        """
        :param class_map: key -> class index, values -> class label
        :param label_display_time: how long the class label will be displayed on screen
        """

        self.class_map = class_map
        self.num_classes = len(class_map.keys())
        self.label_display_time = label_display_time
        self.color_map = plt.get_cmap(colormap)
        self.y_position = y_position
        self.current_class_ind = None
        self.current_class_time_start = None

    def set_current_class(self, class_ind):
        self.current_class_ind = class_ind
        self.current_class_time_start = time.clock()

    def deset_current_class(self):
        self.current_class_ind = None
        self.current_class_time_start = None

    def update_state(self, state_dict):
        if "activated_class" in state_dict and state_dict["activated_class"] is not None:
            current_class = state_dict["activated_class"]
            self.set_current_class(current_class)

            del state_dict["activated_class"]

    def draw_frame(self, frame):
        if self.current_class_ind is None:
            return

        duration_since_activation = time.clock() - self.current_class_time_start
        opacity = int(min((duration_since_activation / self.label_display_time) * 255, 255))
        h, w, c = frame.shape
        x = w
        label = self.class_map[self.current_class_ind]
        (label_width, label_height), baseline = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
        x -= label_width
        color = get_class_color(self.color_map, self.current_class_ind, self.num_classes)
        color = (color[0], color[1], color[2], opacity)
        cv2.putText(frame, label, (x, self.y_position), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=color, thickness=2)

        if duration_since_activation > self.label_display_time:
            self.deset_current_class()

