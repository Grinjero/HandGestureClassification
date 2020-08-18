import numpy as np
import cv2
from PIL import Image

from spatial_transforms import CenterCrop, Scale, Compose, Normalize
from visualization.stream_utils import load_RGB_PIL_stream

def stream_camera_flow(spatial_transforms):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    optical_flow = cv2.optflow.createOptFlow_DualTVL1()

    previous_frame = None
    while (cap.isOpened()):
        ret, current_frame = cap.read()

        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        current_frame = spatial_transforms(current_frame)

        if previous_frame is not None:
            flow = optical_flow.calc(previous_frame, current_frame, None)
            flow_magnitude = np.sqrt(np.power(flow[:, :, 0], 2) + np.power(flow[:, :, 1], 2))

            concatenated = np.concatenate((flow[:, :, 0], flow[:, :, 1], flow_magnitude), axis=1)
            concatenated = cv2.resize(concatenated, (128 * 6, 128 * 2))

            text_loc_x = 5
            text_loc_y = 20
            cv2.putText(concatenated, "Flow0", (text_loc_x, text_loc_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0), thickness=1)
            text_loc_x += 128 * 2
            cv2.putText(concatenated, "Flow1", (text_loc_x, text_loc_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0), thickness=1)
            text_loc_x += 128 * 2
            cv2.putText(concatenated, "Magnitude", (text_loc_x, text_loc_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0), thickness=1)
            cv2.imshow('Flow horizontal and vertical', concatenated)

        previous_frame = current_frame

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def stream_camera_rgb(spatial_transforms):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while(cap.isOpened()):
        ret, frame = cap.read()

        frame = spatial_transforms(frame)

        image_size_str = "H: {} W: {}".format(frame.shape[0], frame.shape[1])
        cv2.putText(frame, image_size_str, (15, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(255, 0, 0), thickness=1)

        cv2.imshow('RGB', frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    spatial_transforms = Compose([
        Scale(112),
        CenterCrop(112),
    ])
    # TODO: otkrit zašto ima sivi border kad je scale isti kao center crop
    # shvatit zašto ne vata nikakve druge geste osim no gesture i doing something else
    # stream_camera_flow(spatial_transforms)
    stream_camera_rgb(spatial_transforms)

if __name__ == "__main__":
    main()
