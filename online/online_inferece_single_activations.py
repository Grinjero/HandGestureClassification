import argparse

from classifier.action_classifier import ActionClassifier
from datasets.dataset_utils import get_class_labels
from spatial_transforms import *
from visualization.visualizer import SyncVideoVisualizer, TopKVisualizer, FPSVisualizer, ClassifiedClassVisualizer
from utils.video_manager import SyncVideoManager
from online.online_opts import *
from online.online_utils import FPSMeasurer
from classifier.action_activation import ActionActivator
from opts import parse_input, parse_model
from online.online_opts import parse_source, parse_paths, parse_preprocessing, parse_online

DISPLAY_SCALE = 600

def parse_args():
    parser = argparse.ArgumentParser()

    parse_paths(parser)
    parse_source(parser)
    parse_online(parser)
    parse_activator(parser)
    parse_preprocessing(parser)
    parse_model(parser)
    parse_input(parser)

    return parser.parse_args()


def main():
    opts = parse_args()

    spatial_transforms = Compose([
        CV2ToPIL("BGR"),
        Scale(opts.smaller_dimension_size),
        CenterCrop(opts.center_crop_size),
        ToTensor(1),
        Normalize([114.7748, 107.7354, 99.475], [1, 1, 1])
    ])

    sequence_length = opts.sample_duration * opts.downsample

    classifier = ActionClassifier(opts=opts)
    if opts.source == "camera":
        video_capturer = SyncVideoManager(source=opts.camera_index,
                                              sequence_length=sequence_length,
                                              num_frames_skip=opts.skip_frames,
                                              spatial_transforms=spatial_transforms)
    elif opts.source == "video":
        video_capturer = SyncVideoManager(source=opts.video_path,
                                              sequence_length=sequence_length,
                                              num_frames_skip=opts.skip_frames,
                                              spatial_transforms=spatial_transforms)
    else:
        raise ValueError("Invalid source")

    class_map = get_class_labels(opts.categories_path)
    topK_visualizer = TopKVisualizer(class_map,
                                     top_k=5)
    fps_visualizer = FPSVisualizer()
    class_visualizer = ClassifiedClassVisualizer(class_map)
    image_visualizers = [topK_visualizer, fps_visualizer, class_visualizer]
    video_visualizer = SyncVideoVisualizer(image_visualizers, ScaleCV2(DISPLAY_SCALE))
    activation_processing = ActionActivator(opts, (0, 2))

    fps_measurer = FPSMeasurer()
    while video_capturer.capture_stream():
        clip = video_capturer.read_clip()
        if clip:
            prediction = classifier(clip)
            activated_class = activation_processing(prediction)

            classifier_state = {
                "predictions": prediction,
                "fps": fps_measurer.fps(),
                "activated_class": activated_class
            }
            video_visualizer.update_state(classifier_state)

        frame = video_capturer.read_frame()
        if video_visualizer.display(frame) is False:
            # not really intuitive but the kill switch of the program is in the visualizer
            break

        fps_measurer.operation_complete()

if __name__ == "__main__":
    main()