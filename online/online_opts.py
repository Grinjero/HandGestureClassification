import os
import yaml

def parse_dataset(args):
    dataset_config_path = args.dataset_config
    if not os.path.exists(dataset_config_path):
        raise ValueError("Given dataset config does not exist")

    with open(dataset_config_path, "r") as fd:
        dataset_config = yaml.load(fd)

    args.contrast_class_indices = dataset_config["CONTRAST_CLASS_INDICES"]
    args.categories_path = dataset_config["CATEGORIES_PATH"]
    args.num_classes = dataset_config["NUM_CLASSES"]
    args.n_classes = dataset_config["NUM_CLASSES"]


def parse_preprocessing(parser):
    parser.add_argument('--smaller_dimension_size', type=int, default=120)
    parser.add_argument('--center_crop_size', type=int, default=112)

def parse_online(parser):
    parser.add_argument('--model_path', type=str, help='Path to the .pth file of the used model', required=True)
    parser.add_argument('--skip_frames', type=int, default=4, help='How many frames to skip between classifications')

def parse_activator(parser):
    parser.add_argument('--output_queue_size', type=int, default=4, help='Size of filtering queue for output')
    parser.add_argument('--filter_method', choices=["median", "mean", "exp"], default="median")
    parser.add_argument('--contrast_patience', type=int, default=3)
    parser.add_argument('--average_gesture_duration', type=float, default=3, help='Average duration of gestures in seconds')
    parser.add_argument('--early_threshold', type=float, default=0.9)
    parser.add_argument('--late_threshold', type=float, default=0.15, help='Threshold for the dff')
    parser.add_argument('--cumulative_method', type=str, default="timed")

def parse_source(parser):
    parser.add_argument('--output_fps', type=int, default=-1, help="FPS of the output video")
    # camera, video or image (if directory containing extracted frames) stream
    subparsers = parser.add_subparsers(dest="source", help='Choose between image stream, video stream or camera stream', required=True)
    # camera
    camera_subparser = subparsers.add_parser('camera')
    camera_subparser.add_argument('--camera_index', type=int, default=0, help="Index of camera")
    #video stream
    video_subparser = subparsers.add_parser('video')
    video_subparser.add_argument('--video_path', type=str, help="Path to the video", required=True)
