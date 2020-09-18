import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import time
from models.mobilenetv2 import conv_1x1x1_bn


def conv_bn(inp, oup, stride, kernel_size=3, padding=(1, 1, 1)):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, depthwise_kernel_size=3, padding=(0, 0, 0)):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.depthwise_kernel_size = depthwise_kernel_size

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, depthwise_kernel_size, stride, padding, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, depthwise_kernel_size, stride, padding, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SlowMobileNetV2(nn.Module):
    """
    In comparison to standard MobileNetV2 temporal stride is removed since the model expects
    inputs with a low spatial dimension (4 frames in most cases).
    Only the blocks with spatial dimensions 7x7 actually perform convolutions with 3D kernels,
    other blocks use (1, 3, 3) kernels that are in essence frame-wise 2D convolutions.

    The goal is to force this model to learn as much spatial information while disregarding
    temporal changes in the early layers. Using temporal information in early layers
    would only lead to degradation of accuracy when observing fast movements since there
    is a large temporal distance between each frame (correlation between frames would only be possible
    with very abstract features with large spatial receptive fields).
    """
    def __init__(self, num_classes=1000, sample_size=112, width_mult=1.):
        super(SlowMobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t (expand ratio), c (channels), n (number of repetitions), s (block strides), k (depthwise kernel size)
            [1, 16, 1, (1, 1, 1), (1, 3, 3)],
            [6, 24, 2, (1, 2, 2), (1, 3, 3)],
            [6, 32, 3, (1, 2, 2), (1, 3, 3)],
            [6, 64, 4, (1, 2, 2), (1, 3, 3)],
            [6, 96, 3, (1, 1, 1), (1, 3, 3)],
            [6, 160, 3, (1, 2, 2), (3, 3, 3)],
            [6, 320, 1, (1, 1, 1), (3, 3, 3)],
        ]

        # building first layer
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, (1, 2, 2))]
        # building inverted residual blocks
        for t, c, n, s, k in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                if isinstance(k, tuple):
                    padding = (k[0] // 2, k[1] // 2, k[2] // 2)
                else:
                    padding = k // 2
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t,
                                           depthwise_kernel_size=k, padding=padding))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # for layer in self.features:
        #     x = layer(x)
        #     print("Layer {} output size: {}".format(layer, x.size()))
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def define_arguments(model_parameter_map):
    model_parameter_map["width_mult"] = {
        "title": "--width_mult",
        "type": float,
        "default": 1
    }


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = SlowMobileNetV2(**kwargs)
    return model


if __name__ == "__main__":
    model = get_model(num_classes=17, sample_size=112, width_mult=1.)
    model = model.cuda()
    model.eval()
    print(str(model) + "\n\n\n")
    # BATCH X CHANNELS X NUM_FRAMES X H X W
    input_var = torch.randn(1, 3, 16, 112, 112).cuda()

    time_start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, 100):
            output = model(input_var)

    duration = time.perf_counter() - time_start
    print("\n\nOutput shape " + str(output.shape) + "\n\n")
    avg_execution_time = duration / 100
    avg_fps = 1 / avg_execution_time
    print("Duration {}, Average execution time {}, Average FPS {}".format(duration, avg_execution_time, avg_fps))