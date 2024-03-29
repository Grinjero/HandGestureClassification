import torch
from torch import nn

from models import mobilenet, mobilenetv2, slow_mobilenetv2, fast_mobilenetv2, slow_fast_mobilenetv2


def generate_model(opt):
    assert opt.model in ['c3d', 'squeezenet', 'mobilenet', 'resnext', 'resnet', 'shufflenet', 'mobilenetv2',
                         'shufflenetv2', 'slow_mobilenetv2', 'fast_mobilenetv2', 'slow_fast_mobilenetv2']


    if opt.model == 'mobilenet':
        from models.mobilenet import get_fine_tuning_parameters
        model = mobilenet.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'mobilenetv2':
        from models.mobilenetv2 import get_fine_tuning_parameters
        model = mobilenetv2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'fast_mobilenetv2':
        from models.fast_mobilenetv2 import get_fine_tuning_parameters
        model = fast_mobilenetv2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'slow_mobilenetv2':
        from models.slow_mobilenetv2 import get_fine_tuning_parameters
        model = slow_mobilenetv2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'slow_fast_mobilenetv2':
        from models.slow_fast_mobilenetv2 import get_fine_tuning_parameters
        model = slow_fast_mobilenetv2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult_slow=opt.width_mult_slow,
            beta=opt.beta,
            fusion_kernel_size=opt.fusion_kernel_size,
            fusion_conv_channel_ratio=opt.fusion_conv_channel_ratio,
            slow_frames=opt.slow_frames,
            fast_frames=opt.fast_frames,
            lateral_connection_section_indices=opt.lateral_connection_section_indices)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        assert torch.cuda.is_available()
        print("Using GPU {}".format(torch.cuda.get_device_name(0)))
        print("Total number of trainable parameters: ", pytorch_total_params)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            model.load_state_dict(pretrain['state_dict'])

            if opt.inference:
                return model, []

            if opt.model in ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2', 'slow_fast_mobilenetv2']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes))
                model.module.classifier = model.module.classifier.cuda()
            elif opt.model == 'squeezenet':
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
                model.module.classifier = model.module.classifier.cuda()
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2', '']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes)
                                )
            elif opt.model == 'squeezenet':
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1, 4, 4), stride=1))
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    if opt.resume_path:
        if not opt.no_cuda:
            model = model.cuda()
        model_state = torch.load(opt.resume_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_state['state_dict'])

    return model, model.parameters()
