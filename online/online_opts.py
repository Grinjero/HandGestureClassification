def parse_paths(parser):
    parser.add_argument('--model_path', type=str, help='Path to the .pth file of the used model', required=True)
    parser.add_argument('--categories_path', type=str,
                        help='File containing class indices and their names, etc. annotation_Jester/categories.txt',
                        required=True)

def parser_preprocessing(parser):
    parser.add_argument('--smaller_dimension_size', type=int, default=120)
    parser.add_argument('--center_crop_size', type=int, default=112)

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

    # #image stream
    # image_subparser = subparsers.add_parser('images')
    # image_subparser.add_argument('--images_dir_path', type=str, help="Path to the directory containing video frames")
    # image_subparser.add_argument('--fps', type=int, default=30, help="FPS of the original video")
    # image_subparser.add_argument('--output_fps', type=int, default=-1, help="FPS of the output video")