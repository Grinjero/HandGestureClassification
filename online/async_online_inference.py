import argparse
import time

from spatial_transforms import *
from classifier.action_classifier import ActionClassifier
from classifier.classifier_postprocessing import ClassifierPostprocessing
from visualization.visualizer import VideoVisualizer, TopKVisualizer, FPSVisualizer
from utils.video_manager import ReadOnlyVideoManager

from online.online_utils import FPSMeasurer
from opts import parse_input_opts, parse_model_opts
from online.online_opts import parse_source, parse_paths, parser_preprocessing


def parse_args():
    parser = argparse.ArgumentParser()
    parse_paths(parser)
    parser_preprocessing(parser)
    parse_model_opts(parser)
    parse_input_opts(parser)
    parse_source(parser)

    return parser.parse_args()


def main():
    opts = parse_args()

    spatial_transforms = Compose([
        CV2ToPIL("BGR"),
        Scale(opts.smaller_dimension_size),
        CenterCrop(opts.center_crop_size),
        ToTensor(),
        Normalize([0, 0, 0], [1, 1, 1])
    ])

    classifier = ActionClassifier(opts=opts, spatial_transforms=spatial_transforms)
    if opts.source == "camera":
        video_capturer = ReadOnlyVideoManager(source=opts.camera_index, num_frames=opts.sample_duration,
                                              subsampling_rate=opts.downsample)
    elif opts.source == "video":
        video_capturer = ReadOnlyVideoManager(source=opts.video_path, num_frames=opts.sample_duration,
                                              subsampling_rate=opts.downsample)
    else:
        raise ValueError("Invalid source")

    topK_visualizer = TopKVisualizer(opts.n_classes,
                                      opts.categories_path,
                                      top_k=5)
    fps_visualizer = FPSVisualizer()
    image_visualizers = [topK_visualizer, fps_visualizer]
    video_visualizer = VideoVisualizer(video_capturer, image_visualizers)
    postprocessing = ClassifierPostprocessing(None)

    fps_measurer = FPSMeasurer()

    video_capturer.start()
    video_visualizer.start()
    more_to_come = True
    pred_counter = 0
    while more_to_come:
        more_to_come, clip = video_capturer.read_clip()

        if clip is None:
            time.sleep(0.02)
            continue

        prediction = classifier(clip)
        processed_prediction = postprocessing(prediction)
        print("Done with prediction " + str(pred_counter))

        fps_measurer.operation_complete()

        pred_counter += 1
        classifier_state = {
            "predictions": processed_prediction,
            "fps": fps_measurer.fps()
        }
        video_visualizer.update_state(classifier_state)

    video_capturer.clean()
    video_visualizer.clean()


if __name__ == "__main__":
    main()
