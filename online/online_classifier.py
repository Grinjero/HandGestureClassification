import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional

from model import generate_model
from datasets.dataset_utils import get_class_labels
from online.online_utils import Queue, Dummy

class OnlineClassifier:
    def __init__(self,
                 model_name,
                 pretrain_path,
                 categories_path,
                 sampling_delay,
                 classification_thresh,
                 output_queue_size,
                 num_input_images,
                 input_image_size,
                 spatial_transforms):
        """
        :param model_name:
        :param pretrain_path:
        :param categories_path:
        :param sampling_delay:
        :param classification_thresh:
        :param output_queue_size:
        :param num_input_images:
        :param input_image_size: C x H x W format
        :param spatial_transforms:
        """
        # classification related
        self.classification_thresh = classification_thresh
        self.sampling_delay = sampling_delay
        self.frame_counter = 0

        self.id_class_label_map = get_class_labels(categories_path)

        self.spatial_transforms = spatial_transforms

        self._init_model(model_name, input_image_size, pretrain_path)
        self._init_image_queue(num_input_images, input_image_size)
        self._init_output_queue(output_queue_size)

    def _init_model(self, model_name, input_image_size, pretrain_path):
        # eh
        model_opts = Dummy
        model_opts.n_classes = len(self.id_class_label_map)
        # C is at the 0 index
        model_opts.sample_size = input_image_size[1]
        model_opts.model = model_name
        model_opts.no_cuda = False
        model_opts.width_mult = 1
        model_opts.pretrain_path = pretrain_path
        model_opts.arch = model_name
        model_opts.inference = True
        self.model, _ = generate_model(model_opts)

        print()
        self.model.eval()

        device_ind = torch.cuda.current_device()
        print("Num of cuda devices " + str(torch.cuda.device_count()))
        print(torch.cuda.get_device_name(device_ind))

    def _init_image_queue(self, num_input_images, input_image_shape):
        self.image_queue = Queue(num_input_images, input_image_shape, is_tensor=True)

    def _init_output_queue(self, output_queue_size):
        self.output_queue = Queue(output_queue_size, (len(self.id_class_label_map)))

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
        clip = clip.permute(1, 0, 3, 2).unsqueeze(0)

        with torch.no_grad():
            inputs = clip.cuda()

            classifier_outputs = self.model(inputs)
            classifier_outputs = functional.softmax(classifier_outputs, dim=1)
            classifier_probabilities = classifier_outputs.cpu().numpy().squeeze()

            self.output_queue.enqueue(classifier_probabilities)

            filtered_probabilites = self.output_queue.mean_filtering()


        #class_idx = np.argmax(filtered_probabilites)
        top_filtered_class_idx = np.argmax(filtered_probabilites)
        filtered_class_prob = filtered_probabilites[top_filtered_class_idx]

        top_unfiltered_class_idx = np.argmax(classifier_probabilities)
        unfiltered_class_prob = classifier_probabilities[top_unfiltered_class_idx]
        # label indexing starts from 1
        filtered_label = self.id_class_label_map[top_filtered_class_idx + 1]
        unfiltered_label = self.id_class_label_map[top_unfiltered_class_idx + 1]

        return filtered_class_prob, unfiltered_class_prob, filtered_label, unfiltered_label
