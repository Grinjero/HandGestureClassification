from datasets.jester import Jester
from datasets.nv import NV

def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['kinetics', 'jester', 'ucf101', 'nvgesture']

    if opt.dataset == 'jester':
        training_data = Jester(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'nvgesture':
        training_data = NV(
            opt.video_path,
            opt.annotation_path,
            'training',
            n_samples_for_each_video=1,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)
    else:
        raise ValueError("Given dataset not available")
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['kinetics', 'jester', 'ucf101', 'nvgesture']

    if opt.dataset == 'jester':
        validation_data = Jester(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)

    else:
        raise ValueError("Given dataset not available")
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['kinetics', 'jester', 'ucf101', 'nvgesture']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    else:
        subset = 'testing'

    if opt.dataset == 'jester':
        test_data = Jester(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'nvgesture':
        test_data = NV(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    else:
        raise ValueError("Given dataset not available")
    return test_data
