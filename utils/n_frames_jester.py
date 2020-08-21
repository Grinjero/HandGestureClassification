from __future__ import print_function, division
import os
import sys

VOLUME_SPLITS = [20000, 40000, 60000, 80000, 100000, 120000]


def generate_n_frames_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        video_dir_path = os.path.join(folder_path, file_name)
        if os.path.isdir(video_dir_path) is False:
            continue
        image_indices = []
        for image_file_name in os.listdir(video_dir_path):
            if '00' not in image_file_name:
                continue
            image_indices.append(int(image_file_name[0:4]))

        if len(image_indices) == 0:
            print('no image files', video_dir_path)
            n_frames = 0
        else:
            image_indices.sort(reverse=True)
            n_frames = len(image_indices)
            print(video_dir_path, n_frames)
        with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
            dst_file.write(str(n_frames))


def class_process(dir_path):
    if not os.path.isdir(dir_path):
        return

    if len(os.listdir(dir_path)) <= 20:
        # volumed
        for volume in os.listdir(dir_path):
            vol_path = os.path.join(dir_path, volume)
            if os.path.isdir(vol_path) and volume != ".ipynb_checkpoints":
                generate_n_frames_in_folder(vol_path)

    else:
        generate_n_frames_in_folder(dir_path)


if __name__ == "__main__":
    dir_path = sys.argv[1]
    class_process(dir_path)
