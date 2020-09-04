import queue
import cv2
import torch
import torch.nn.functional as F

from model import generate_model

class ActionClassifier:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, opts, spatial_transforms, gpu_id=None):
        self.model, _ = generate_model(opts)
        self.model.eval()

        self.gpu_id = gpu_id
        if self.gpu_id is None:
            self.gpu_id = torch.cuda.current_device()

        self.spatial_transforms = spatial_transforms


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
            inputs = [self.spatial_transforms(frame) for frame in clip]
            inputs = torch.stack(inputs, 0)
            inputs = inputs.permute(1, 0, 2, 3).unsqueeze(0)

            # Transfer the data to the current GPU device.
            inputs = inputs.cuda(device=torch.device(self.gpu_id), non_blocking=True)

            preds = self.model(inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.cpu()
            preds = preds.detach()

        return preds