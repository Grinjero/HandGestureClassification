import cv2
import argparse
import time
import os
from torch import Tensor

from spatial_transforms_cv2 import *
from online.online_classifier import OnlineClassifier
from online.online_utils import FPSMeasurer, ImageStreamer
from visualization.stream_utils import *

DISPLAY_SCALE = 480

def parse_args():
    parser = argparse.ArgumentParser()

    # model related
    parser.add_argument('--model', type=str, help='mobilenet', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the .pth file of the used model', required=True)
    parser.add_argument('--no_cuda', type=bool, help='Should GPU or CPU be used', default=False)

    parser.add_argument('--categories_path', type=str,
                        help='File containing class indices and their names, etc. annotation_Jester/categories.txt',
                        required=True)

    # classification related
    parser.add_argument('--sampling_delay', type=int, default=2,
                        help='Delay in frame sampling')
    parser.add_argument('--output_queue_size', type=int, default=16,
                        help='How many outputs are stored in a queue for filtering outputs')
    parser.add_argument('--classification_thresh', type=float, default=0.2,
                        help='Minimal difference between top 2 probabilities needed to classify as gesture')

    # image preprocessing related
    parser.add_argument('--smaller_dimension_size', type=int, default=120)
    parser.add_argument('--center_crop_size', type=int, default=112)

    # camera, video or image (if directory containing extracted frames) stream
    subparsers = parser.add_subparsers(dest="stream_type", help='Choose between image stream, video stream or camera stream', required=True)
    # camera
    camera_subparser = subparsers.add_parser('camera')
    camera_subparser.add_argument('--camera_index', type=int, default=0, help="Index of camera")
    #video stream
    video_subparser = subparsers.add_parser('video')
    video_subparser.add_argument('--video_path', type=str, help="Path to the video")
    video_subparser.add_argument('--output_fps', type=int, default=30, help="FPS of the output video")
    #image stream
    image_subparser = subparsers.add_parser('images')
    image_subparser.add_argument('--images_dir_path', type=str, help="Path to the directory containing video frames")
    image_subparser.add_argument('--fps', type=int, default=30, help="FPS of the original video")
    image_subparser.add_argument('--output_fps', type=int, default=30, help="FPS of the output video")

    return parser.parse_args()


def main():
    args = parse_args()

    input_image_size = (3, args.center_crop_size, args.center_crop_size)
    spatial_transforms = Compose([
        Scale(args.smaller_dimension_size),
        CenterCrop(args.center_crop_size),
        ToTensor(),
        Normalize([0, 0, 0], [1, 1, 1])
    ])

    classifier = OnlineClassifier(model_name=args.model,
                                  model_path=args.model_path,
                                  categories_path=args.categories_path,
                                  sampling_delay=args.sampling_delay,
                                  classification_thresh=args.classification_thresh,
                                  output_queue_size=args.output_queue_size,
                                  # It shall remain a magic number, all models use the same value
                                  num_input_images=16,
                                  input_image_size=input_image_size,
                                  spatial_transforms=spatial_transforms)
    num_ops = 0
    start_time = time.time()

    if args.stream_type == "camera":
        cap = cv2.VideoCapture(args.camera_index)
    elif args.stream_type == "images":
        cap = ImageStreamer(args.images_dir_path, args.fps)
    elif args.stream_type == "video":
        cap = cv2.VideoCapture(args.video_path)
    else:
        raise ValueError("Missing stream type argument (argparse should handle this anyway)")


    fps_measurer = FPSMeasurer()

    previous_filtered_class_prob = 0
    previous_unfiltered_class_prob = 0
    previous_filtered_label = None
    previous_unfiltered_label = None

    cv2.namedWindow("Stream")
    cv2.moveWindow("Stream", 200, 400)
    cv2.namedWindow("Augmented stream")
    cv2.moveWindow("Augmented stream", 1000, 400)

    original_frame_disp_scale = Scale(DISPLAY_SCALE)
    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret is False:
            return
        original_frame = frame
        augmented_image = spatial_transforms(frame)
        if isinstance(augmented_image, Tensor):
            augmented_image = augmented_image.numpy()
            augmented_image = np.transpose(augmented_image, (1, 2, 0))

        original_frame = original_frame_disp_scale(original_frame)
        display_dimensions(original_frame, (15, 25), original_frame.shape, (255, 0, 0))

        classification_result = classifier(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if classification_result is None:
            filtered_class_prob, unfiltered_class_prob, filtered_label, unfiltered_label = previous_filtered_class_prob, previous_unfiltered_class_prob, previous_filtered_label, previous_unfiltered_label
        else:
            filtered_class_prob, unfiltered_class_prob, filtered_label, unfiltered_label = classification_result

        previous_filtered_class_prob, previous_unfiltered_class_prob, previous_filtered_label, previous_unfiltered_label = filtered_class_prob, unfiltered_class_prob, filtered_label, unfiltered_label

        display_class_probability(original_frame, (15, 50), filtered_class_prob, filtered_label, "Filtered", (255, 0, 255))
        display_class_probability(original_frame, (15, 100), unfiltered_class_prob, unfiltered_label, "Raw", (255, 255, 0))

        fps_measurer.operation_complete()
        display_framerate(original_frame, (420, 25), fps_measurer.fps(), (0, 255, 255))

        cv2.imshow("Augmented stream", augmented_image)
        cv2.imshow("Stream", original_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        num_ops += 1

    duration = time.time() - start_time
    print("{} classifications in {:.3f} seconds".format(num_ops, duration))
    print("Avg FPS {:.3f}".format(num_ops/duration))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
