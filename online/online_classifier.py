import numpy as np
import torch
from torch.nn import functional

from model import generate_model
from datasets.dataset_utils import get_class_labels
from online.online_utils import Queue

class OnlineClassifier:
    def __init__(self,
                 opts,
                 spatial_transforms,
                 input_image_size,
                 num_input_images=16):
        """
        :param model_name:
        :param model_path:
        :param categories_path:
        :param sampling_delay:
        :param classification_thresh:
        :param output_queue_size:
        :param num_input_images:
        :param input_image_size: C x H x W format
        :param spatial_transforms:
        """
        # classification related
        self.classification_thresh = opts.classification_thresh
        self.sampling_delay = opts.downsample
        self.frame_counter = -1
        self.num_input_images = num_input_images
        self.id_class_label_map = get_class_labels(opts.categories_path)

        self.spatial_transforms = spatial_transforms

        self._init_model(opts)
        self._init_image_queue(num_input_images, input_image_size)
        self._init_output_queue(opts.output_queue_size)




    def _init_model(self, opts):
        self.n_classes = len(self.id_class_label_map)
        opts.n_classes = self.n_classes
        opts.sample_size = self.num_input_images
        opts.resume_path = opts.model_path
        self.model, _ = generate_model(opts)
        self.model.eval()

        device_ind = torch.cuda.current_device()
        print("Num of cuda devices " + str(torch.cuda.device_count()))
        print(torch.cuda.get_device_name(device_ind))

    def _init_image_queue(self, num_input_images, input_image_shape):
        self.image_queue = Queue(num_input_images)

    def _init_output_queue(self, output_queue_size):
        self.output_queue = Queue(output_queue_size)

    def classification_ready(self):
        # return True
        self.frame_counter = (self.frame_counter + 1) % self.sampling_delay
        return self.frame_counter == 0

    def __call__(self, current_frame):
        if self.classification_ready() is False:
            return None

        if self.spatial_transforms is not None:
            current_frame = self.spatial_transforms(current_frame)
        self.image_queue.enqueue(current_frame)
        clip = self.image_queue.to_tuple()
        clip = torch.stack(clip, 0)
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0)

        with torch.no_grad():
            inputs = clip.cuda()

            classifier_outputs = self.model(inputs)
            classifier_outputs = functional.softmax(classifier_outputs, dim=1)
            classifier_probabilities = classifier_outputs.cpu().numpy().squeeze()

            self.output_queue.enqueue(classifier_probabilities)

            filtered_probabilites = self.output_queue.median_filtering()

        top_filtered_class_idx = np.argmax(filtered_probabilites)
        filtered_class_prob = filtered_probabilites[top_filtered_class_idx]


        top_unfiltered_class_idx = np.argmax(classifier_probabilities)
        unfiltered_class_prob = classifier_probabilities[top_unfiltered_class_idx]

        filtered_label = self.id_class_label_map[top_filtered_class_idx]
        unfiltered_label = self.id_class_label_map[top_unfiltered_class_idx]
        print("Class {}, prob {}".format(unfiltered_label, unfiltered_class_prob))
        # return filtered_class_prob, unfiltered_class_prob, filtered_label, unfiltered_label
        return filtered_probabilites, classifier_probabilities
