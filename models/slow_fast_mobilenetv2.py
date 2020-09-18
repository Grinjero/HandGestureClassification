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


class SlowFastMobileNetV2(nn.Module):
    """
    Combination of SlowMobileNetV2 and FastMobileNetV2, for more details refer to their files. Uses two "separate" pathways
    with different temporal resolutions. Slow pathway uses the SlowMobileNetV2 as its pathway an receives every alpha-th
    frame (in this work every 4-th frame), Fast pathway uses the FastMobileNetV2 as its pathway and uses every frame given
    in the forward (16 frames). In both pathways no temporal downsampling is performed.

    width_mult_slow : float
        Width multiplier for the slow pathway
    beta : float
        Width multiplier (channel capacity) ratio of the fast pathway and the slow pathway i.e. with_mult_fast = beta * width_mult_slow
        Ideally as low as possible (0.2, 0.15, 0.3)
    fusion_conv_channel_ratio : int, default is 2.
        Ratio of channel dimensions between the Slow and Fast pathways.
    fusion_kernel_size : int, default is 5.
        Kernel dimension used for fusing information from Fast pathway to Slow pathway.
    lateral_connection_section_indices : list of ints
        lateral connection will be set before sections defined in interverted_residual_setting_fast and
        interverted_residual_setting_slow
    """
    def __init__(self, num_classes=1000, sample_size=112, width_mult_slow=1.0, beta=0.2,
                 fusion_kernel_size=5,
                 fusion_conv_channel_ratio=2,
                 slow_frames=4,
                 fast_frames=16,
                 lateral_connection_section_indices=(0, 2, 3, 4),
                 input_channel=32,
                 last_channel=1280):
        super(SlowFastMobileNetV2, self).__init__()
        self.width_mult_slow = width_mult_slow
        self.width_mult_fast = width_mult_slow * beta
        self.beta = beta
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.fusion_kernel_size = fusion_kernel_size
        self.fusion_conv_channel_ratio = fusion_conv_channel_ratio
        self.lateral_connection_section_indices = lateral_connection_section_indices
        self.sample_size = sample_size

        self.input_channel = input_channel
        self.last_channel = last_channel

        assert self.sample_size % 16 == 0.
        assert self.fast_frames % self.slow_frames == 0
        self.alpha = self.fast_frames // self.slow_frames

        block = InvertedResidual
        self._make_fast_pathway(block)
        self._make_slow_pathway(block)

        self.fused_last_channel = self.fast_last_channel + self.slow_last_channel
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fused_last_channel, num_classes),
        )

        self._initialize_weights()

    def _make_fast_pathway(self, block):
        interverted_residual_setting_fast = [
            # t, c, n, s
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (1, 2, 2)],
            [6, 32, 3, (1, 2, 2)],
            [6, 64, 4, (1, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (1, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]

        # building first layer
        input_channel = int(self.input_channel * self.width_mult_fast)
        self.fast_last_channel = int(self.last_channel * self.width_mult_fast)

        self.fast_first_layer = conv_bn(3, input_channel, (1, 2, 2))

        self.fast_sections = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        self.lateral_connections_channels = []
        # building inverted residual blocks
        for section_index, block_specs  in enumerate(interverted_residual_setting_fast):
            t, c, n, s = block_specs

            # lateral connections are placed before sections
            if section_index in self.lateral_connection_section_indices:
                lateral_output_channels = input_channel * self.fusion_conv_channel_ratio
                self.lateral_connections_channels.append(lateral_output_channels)
                lateral_block = block(input_channel, lateral_output_channels,
                                      stride=(self.alpha, 1, 1),  expand_ratio=t, depthwise_kernel_size=(self.fusion_kernel_size, 1, 1),
                                      padding=(self.fusion_kernel_size // 2, 0, 0))
                self.lateral_connections.append(lateral_block)
            else:
                self.lateral_connections_channels.append(None)

            output_channel = int(c * self.width_mult_fast)
            fast_section_list = []
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                fast_section_list.append(block(input_channel, output_channel, stride, expand_ratio=t, padding=(1, 1, 1)))
                input_channel = output_channel

            self.fast_sections.append(nn.Sequential(*fast_section_list))

        # building last several layers
        self.fast_last_layer = conv_1x1x1_bn(input_channel, self.fast_last_channel)

    def _make_slow_pathway(self, block):
        interverted_residual_setting_slow = [
            # t (expand ratio), c (channels), n (number of repetitions), s (block strides), k (depthwise kernel size)
            [1, 16, 1, (1, 1, 1), (1, 3, 3)],
            [6, 24, 2, (1, 2, 2), (1, 3, 3)],
            [6, 32, 3, (1, 2, 2), (1, 3, 3)],
            [6, 64, 4, (1, 2, 2), (1, 3, 3)],
            [6, 96, 3, (1, 1, 1), (1, 3, 3)],
            [6, 160, 3, (1, 2, 2), (3, 3, 3)],
            [6, 320, 1, (1, 1, 1), (3, 3, 3)],
        ]

        input_channel = int(self.input_channel * self.width_mult_slow)
        self.slow_last_channel = int(self.last_channel * self.width_mult_slow)

        self.slow_first_layer = conv_bn(3, input_channel, (1, 2, 2))

        self.slow_sections = nn.ModuleList()
        # building inverted residual blocks
        for section_index, block_specs in enumerate(interverted_residual_setting_slow):
            t, c, n, s, k = block_specs
            output_channel = int(c * self.width_mult_slow)
            if self.lateral_connections_channels[section_index] != None:
                # entrance to this section must accomodate the lateral connection if it is present
                input_channel += self.lateral_connections_channels[section_index]

            slow_section_list = []
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                if isinstance(k, tuple):
                    padding = (k[0] // 2, k[1] // 2, k[2] // 2)
                else:
                    padding = k // 2
                slow_section_list.append(block(input_channel, output_channel, stride, expand_ratio=t,
                                               depthwise_kernel_size=k, padding=padding))
                input_channel = output_channel
            self.slow_sections.append(nn.Sequential(*slow_section_list))
        # building last several layers
        self.slow_last_layer = conv_1x1x1_bn(input_channel, self.slow_last_channel)

    def _forward_fast(self, fast_x):
        """
        :param fast_x: selected frames for fast
        :return:
            x : vector of result features
            laterals : list of feature volumes
                lateral features -> [0] is the first lateral starting from the bottom of the network
        """

        x = self.fast_first_layer(fast_x)

        laterals = []
        lateral_index = 0
        for section_counter, fast_section in enumerate(self.fast_sections):
            if section_counter in self.lateral_connection_section_indices:
                laterals.append(self.lateral_connections[lateral_index](x))
                lateral_index += 1
            else:
                laterals.append(None)

            x = fast_section(x)

        out = self.fast_last_layer(x)
        out = F.avg_pool3d(out, out.data.size()[-3:])
        out = out.view(-1, self.fast_last_channel)
        return out, laterals

    def _forward_slow(self, slow_x, laterals):
        """
        :param slow_x: subsampled frames for slow
        :param laterals:
        :return:
        """

        x = self.slow_first_layer(slow_x)

        for section_counter, slow_section in enumerate(self.slow_sections):
            if section_counter in self.lateral_connection_section_indices:
                x = torch.cat((x, laterals[section_counter]), dim=1)

            x = slow_section(x)

        out = self.slow_last_layer(x)
        out = F.avg_pool3d(out, out.data.size()[-3:])
        out = out.view(-1, self.slow_last_channel)

        return out

    def forward(self, x):
        # slow uses every alpha-th frame
        slow_x = x[:, :, 0::self.alpha, :, :]
        fast_out, laterals = self._forward_fast(x)
        slow_out = self._forward_slow(slow_x, laterals)

        x = torch.cat((fast_out, slow_out), dim=1)

        out = self.classifier(x)
        return out

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
    model_parameter_map["width_mult_slow"] = {
        "title": "--width_mult_slow",
        "default": 1.0,
        "type": float
    }
    model_parameter_map["beta"] = {
        "title": "--beta",
        "default": 0.2,
        "type": float
    }
    model_parameter_map["fusion_kernel_size"] = {
        "title": "--fusion_kernel_size",
        "default": 5,
        "type": int
    }
    model_parameter_map["fusion_conv_channel_ratio"] = {
        "title": "--fusion_conv_channel_ratio",
        "default": 2,
        "type": int
    }
    model_parameter_map["slow_frames"] = {
        "title": "--slow_frames",
        "default": 4,
        "type": int
    }
    model_parameter_map["fast_frames"] = {
        "title": "--fast_frames",
        "default": 16,
        "type": int
    }
    model_parameter_map["lateral_connection_section_indices"] = {
        "title": "--lateral_connection_section_indices",
        "default": (0, 2, 3, 4),
        "type": int,
        "nargs": "+"
    }


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = SlowFastMobileNetV2(**kwargs)
    return model

if __name__ == "__main__":
    model = get_model(num_classes=17)
    model = model.cuda()
    model.eval()
    print(str(model) + "\n\n\n")
    # BATCH X CHANNELS X NUM_FRAMES X W X H
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