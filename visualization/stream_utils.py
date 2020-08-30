import cv2
from PIL import Image


def load_RGB_PIL_stream(cap):
    ret, frame = cap.read()

    return Image.fromarray(frame)


def display_dimensions(frame, position, image_dims, color, type="hwc"):
    if type == "hwc":
        disp_text = "H: {} W: {} C: {}".format(image_dims[0], image_dims[1], image_dims[2])
    else:
        disp_text = "H: {} W: {} C: {}".format(image_dims[1], image_dims[0], image_dims[2])
    cv2.putText(frame, disp_text, position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=color, thickness=2)


def display_framerate(frame, position, fps, color):
    cv2.putText(frame, "FPS: {:.2f}".format(fps), position, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=color, thickness=2)


def display_class_probability(frame, position, probability, label, out_type, color):
    cv2.putText(frame, "{} class: {}".format(out_type, label), position, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=color, thickness=2)
    x, y = position[0], position[1] + 25
    cv2.putText(frame, "{} probability: {:.5f}".format(out_type, probability), (x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=color, thickness=2)


class TaskInfo:
    def __init__(self):
        self.frames = None
        self.id = -1
        self.action_preds = None
        self.num_buffer_frames = 0
        self.img_height = -1
        self.img_width = -1
        self.crop_size = -1
        self.clip_vis_size = -1

    def add_frames(self, idx, frames):
        """
        Add the clip and corresponding id.
        Args:
            idx (int): the current index of the clip.
            frames (list[ndarray]): list of images in "BGR" format.
        """
        self.frames = frames
        self.id = idx

    def add_action_preds(self, preds):
        """
        Add the corresponding action predictions.
        """
        self.action_preds = preds

class _StopToken:
    pass