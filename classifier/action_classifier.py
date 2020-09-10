import queue
import cv2
import torch
import torch.nn.functional as F

from model import generate_model
from temporal_transforms import OnlineTemporalCrop

class ActionClassifier:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, opts):
        opts.resume_path = opts.model_path
        self.model, _ = generate_model(opts)
        self.model.eval()

        self.temporal_transform = OnlineTemporalCrop(opts.sample_duration, opts.downsample)
        # self.spatial_transforms = spatial_transforms


    def __call__(self, clip):
        """
        Returns the prediction results for the current task.
        Args:
            frames: a list of images
        Returns:
            Model prediction tensor in shape [n_classes]
        """
        with torch.no_grad():
            # inputs = clip[self.temporal_transform(clip)]
            # inputs = [self.spatial_transforms(frame) for frame in clip]
            clip = self.temporal_transform(clip)
            inputs = torch.stack(clip, 0)
            inputs = inputs.permute(1, 0, 2, 3).unsqueeze(0)

            # Transfer the data to the current GPU device.
            inputs = inputs.cuda()

            preds = self.model(inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.cpu()
            preds = preds.detach()

        return preds