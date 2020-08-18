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
