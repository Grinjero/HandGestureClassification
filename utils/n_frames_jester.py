from __future__ import print_function, division
import os
import sys
from datasets.jester import VOLUME_SPLITS

def generate_n_frames_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        video_dir_path = os.path.join(folder_path, file_name)
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

    if len(os.listdir(dir_path)) <= len(VOLUME_SPLITS):
        # volumed
        for volume in os.listdir(dir_path):
            vol_path = os.path.join(dir_path, volume)
            generate_n_frames_in_folder(vol_path)

    else:
        generate_n_frames_in_folder(dir_path)


if __name__=="__main__":
    dir_path = sys.argv[1]
    class_process(dir_path)
