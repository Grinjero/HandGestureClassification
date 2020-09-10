import argparse
import models
import os, importlib
from glob import glob


def parse_scheduler(parser:argparse.ArgumentParser):
    parser.add_argument("--scheduler", type=str, choices=["MultiStepLR", "ReduceLROnPlateau"], help="Which scheduler to use (MultiStepLR | ReduceLROnPlateau)", required=True)

    parser.add_argument('--lr_patience', type=int, default=10, required=False, help='How many epochs without improving till learning rate is decayed with lr_factor. Usedy by ReduceLROnPlateau')
    parser.add_argument('--lr_steps', default=[30, 45], type=int, nargs="+", metavar='LRSteps', help='Epochs to decay learning rate by lr_factor when using MultiStepLR')
    parser.add_argument('--lr_factor', type=float, default=0.1, required=False)


def parse_model(parser:argparse.ArgumentParser):
    model_names = load_submodule_arguments(models, parser)
    model_names = [name for name in model_names if name != "__init__"]
    choices = model_names[0]
    for model_name in model_names:
        choices += " | " + model_name

    parser.add_argument("--model", type=str, choices=model_names, help="Which model to use " + str(choices))
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.', default=False)
    parser.add_argument('--resume_path', type=str, help='Save data (.pth) of previous training', required=False)
    parser.add_argument('--pretrain_path', type=str, help='Pretrained model (.pth)', required=False)

def load_submodule_arguments(parent_module, parser):
    modules = glob(parent_module.__path__[0] + "/*.py")
    module_names = []
    model_parameter_map = dict()
    for module_file in modules:
        module_name = os.path.basename(module_file)[:-3]
        module_names.append(module_name)
        if module_name == "__init__":
            continue
        importlib.invalidate_caches()
        import_path = "{}.{}".format(parent_module.__name__, module_name)
        module = importlib.import_module(import_path)

        if "define_arguments" not in module.__dict__:
            continue
        module.define_arguments(model_parameter_map)

    for key in model_parameter_map.keys():
        value = model_parameter_map[key]
        title = value["title"]
        del value["title"]
        parser.add_argument(title, **value)
    return module_names


def parse_optimizer(parser):
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='Initial learning rate')


def parse_input(parser):
    parser.add_argument('--n_classes', default=400, type=int, help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--modality', default='RGB', type=str, help='Modality of input data. RGB, Flow or RGBFlow')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--downsample', default=1, type=int, help='Downsampling. Selecting 1 frame out of N')


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='', type=str, help='Root directory path of data', required=False)
    parser.add_argument('--video_path', default='video_kinetics_jpg', type=str, help='Directory path of Videos')
    parser.add_argument('--annotation_path', default='kinetics.json', type=str, help='Annotation file path')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--dataset', default='kinetics', type=str, help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument('--n_finetune_classes', default=400, type=int, help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str, help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--mean_dataset', default='activitynet', type=str, help='dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument('--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument('--std_norm', action='store_true', help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=False)
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=250, type=int, help='Number of total epochs to run')
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--n_val_samples', default=3, type=int, help='Number of validation samples for each activity')
    parser.add_argument('--ft_portion', default='complete', type=str, help='The portion of the model to apply fine tuning, either complete or last_layer')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument('--test_subset', default='val', type=str, help='Used subset in test (val | test)')
    parser.add_argument('--scale_in_test', default=1.0, type=float, help='Spatial scale in test')
    parser.add_argument('--crop_position_in_test', default='c', type=str, help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument('--no_softmax_in_test', action='store_true', help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint', default=10, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--no_hflip', action='store_true', help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument('--norm_value', default=1, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--version', default=1.1, type=float, help='Version of the model')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

    parse_input(parser)
    parse_scheduler(parser)
    parse_model(parser)
    parse_optimizer(parser)

    args = parser.parse_args()

    return args
