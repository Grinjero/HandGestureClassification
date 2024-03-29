import argparse
import time

from datasets.dataset_utils import get_class_labels
from spatial_transforms import *
from classifier.action_classifier import ActionClassifier
from classifier.classifier_postprocessing import ClassifierPostprocessing
from visualization.visualizer import AsyncVideoVisualizer, TopKVisualizer, FPSVisualizer
from utils.video_manager import AsyncVideoManager

from online.online_utils import FPSMeasurer
from opts import parse_input, parse_model
from online.online_opts import parse_source, parse_preprocessing, parse_online, parse_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parse_preprocessing(parser)
    parse_model(parser)
    parse_input(parser)
    parse_source(parser)
    parse_online(parser)
    parser.add_argument('--dataset_config', type=str, default="annotation_Jester\Jester.yaml", help="Path to the dataset config")
    args = parser.parse_args()
    parse_dataset(args)
    return args


def main():
    opts = parse_args()
    class_map = get_class_labels(opts.categories_path)

    spatial_transforms = Compose([
        CV2ToPIL("BGR"),
        Scale(opts.smaller_dimension_size),
        CenterCrop(opts.center_crop_size),
        ToTensor(1),
        Normalize([114.7748, 107.7354, 99.475], [1, 1, 1])
    ])

    sequence_length = opts.sample_duration * opts.downsample

    classifier = ActionClassifier(opts=opts, )
    if opts.source == "camera":
        video_capturer = AsyncVideoManager(source=opts.camera_index,
                                           sequence_length=sequence_length,
                                           num_frames_skip=opts.skip_frames,
                                           spatial_transforms=spatial_transforms)
    elif opts.source == "video":
        video_capturer = AsyncVideoManager(source=opts.video_path,
                                           sequence_length=sequence_length,
                                           num_frames_skip=opts.skip_frames,
                                           spatial_transforms=spatial_transforms)
    else:
        raise ValueError("Invalid source")

    topK_visualizer = TopKVisualizer(class_map=class_map,
                                    top_k=5)
    fps_visualizer = FPSVisualizer()
    image_visualizers = [topK_visualizer, fps_visualizer]
    video_visualizer = AsyncVideoVisualizer(video_capturer, image_visualizers)

    fps_measurer = FPSMeasurer()

    video_capturer.start()
    video_visualizer.start()
    more_to_come = True
    while more_to_come:
        more_to_come, clip = video_capturer.read_clip()

        if clip is None or len(clip) < sequence_length:
            time.sleep(0.02)
            continue

        prediction = classifier(clip)
        # processed_prediction = postprocessing(prediction)

        fps_measurer.operation_complete()

        classifier_state = {
            "predictions": prediction,
            "fps": fps_measurer.fps()
        }
        video_visualizer.update_state(classifier_state)

    video_capturer.clean()
    video_visualizer.clean()


if __name__ == "__main__":
    main()
