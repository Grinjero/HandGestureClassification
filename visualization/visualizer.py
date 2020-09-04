import atexit
import threading

import torch
import matplotlib.pyplot as plt
import cv2
import time

from datasets.dataset_utils import get_class_labels


class VideoVisualizer:
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
            frame = image_visualizer.draw_frame(frame)
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
    def __init__(self, color=(255, 0, 0), alpha=0.6):
        self.color = color
        self.alpha = alpha
        # self.color = color[0], color[1], color[2]
        self.current_fps = None

    def update_state(self, state_dict):
        self.current_fps = state_dict["fps"]

    def draw_frame(self, frame):
        if self.current_fps is None:
            return frame

        w, h, c = frame.shape
        x = int(0.9 * w)
        y = 30

        text = "FPS: {:.3f}".format(self.current_fps)
        return cv2.putText(frame, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=self.color, thickness=2)

class TopKVisualizer(ImageVisualizer):
    def __init__(self, num_classes, class_names_path, top_k=1, colormap="rainbow", alpha=0.6, start_position=(10, 20)):
        super().__init__()
        self.num_classes = num_classes
        self.class_map = get_class_labels(class_names_path)
        self.color_map = plt.get_cmap(colormap)
        self.top_k = top_k
        self.alpha = alpha
        self.start_position=start_position

        self.current_preds = None

    def _get_color(self, class_id):
        """
        Get color for a class id.
        Args:
            class_id (int): class id.
        """
        rgb = self.color_map(class_id / self.num_classes)[:3]
        bgr = int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)
        return bgr

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
            color = self._get_color(top_classes[i])
            class_name = self.class_map[top_classes[i]]
            text = "{} {:.6f}".format(class_name, top_scores[i])

            frame = cv2.putText(frame, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=2)

            y += 30

        return frame
