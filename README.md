# Hand gesture recognition
Hand gesture recognition based on https://github.com/okankop/Efficient-3DCNNs

## Models
Used models can be downloaded [here](https://drive.google.com/drive/folders/1SoPDD3dnrDaj9lWn0mxvpRD4OLyD9mls?usp=sharing)


## Dataset Preparation

### Jester

* Download videos [here](https://20bn.com/datasets/jester#download).
* Generate n_frames files using ```utils/n_frames_jester.py```

```bash
python utils/n_frames_jester.py dataset_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/jester_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist.txt, vallist.txt

```bash
python utils/jester_json.py annotation_dir_path
```

## Running the code
How to train and evaluate models examples can be seen in ```jupyter_notebooks\MbNet_SlowFast.ipynb```. 
To turn off online inference press 'q'.
 
To run online inference with SlowFast from webcamera
```bash
python online_inference_single_activations.py \
--model_path model_params/jester17_slow_fast_mobilenetv2_0.2x_RGB_16_best.pth \
--model slow_fast_mobilenetv2 \
--width_mult_slow 1.0 \
--beta 0.2 \
--fusion_kernel_size 5 \
--fusion_conv_channel_ratio 2 \
--slow_frames 4 \
--fast_frames 16 \
--dataset_config "annotation_Jester/Jester17.yaml" \
--downsample 2 \
--skip_frames 1 \
--center_crop_size 112 \
--smaller_dimension_size 112 \
--average_gesture_duration 12 \
--output_queue_size 4 \
--filter_method median \
--early_threshold 0.4 \
--late_threshold 0.10 \
--cumulative_method step \
--contrast_patience 3 \
--output_file camera_result \
camera
```

To run online inference with SlowFast on video
```bash
python online_inference_single_activations.py \
--model_path model_params/jester17_slow_fast_mobilenetv2_0.2x_RGB_16_best.pth \
--model slow_fast_mobilenetv2 \
--width_mult_slow 1.0 \
--beta 0.2 \
--fusion_kernel_size 5 \
--fusion_conv_channel_ratio 2 \
--slow_frames 4 \
--fast_frames 16 \
--dataset_config "annotation_Jester/Jester17.yaml" \
--downsample 2 \
--skip_frames 1 \
--center_crop_size 112 \
--smaller_dimension_size 112 \
--average_gesture_duration 12 \
--output_queue_size 4 \
--filter_method median \
--early_threshold 0.4 \
--late_threshold 0.10 \
--cumulative_method step \
--contrast_patience 3 \
--output_file camera_result \
video
--video_path
sample_videos/sample_17_jester.avi
```

To run online inference with MBNetV2 from webcamera
```bash
python online_inference_single_activations.py \
--model mobilenetv2 \
--model_path model_params/jester17_mobilenetv2_1.0x_RGB_16_best.pth \
--width_mult 1.0 \
--dataset_config "annotation_Jester/Jester17.yaml" \
--downsample 2 \
--skip_frames 1 \
--center_crop_size 112 \
--smaller_dimension_size 112 \
--average_gesture_duration 12 \
--output_queue_size 4 \
--filter_method median \
--early_threshold 0 .4 \
--late_threshold 0.10 \
--cumulative_method step \
--contrast_patience 3 \
--output_file camera_result \
camera
```
To run online inference with MBNetV2 from video
```bash
python online_inference_single_activations.py \
--model mobilenetv2 \
--model_path model_params/jester17_mobilenetv2_1.0x_RGB_16_best.pth \
--width_mult 1.0
--dataset_config "annotation_Jester/Jester17.yaml" \
--downsample 2 \
--skip_frames 1 \
--center_crop_size 112 \
--smaller_dimension_size 112 \
--average_gesture_duration 12 \
--output_queue_size 4 \
--filter_method median \
--early_threshold 0.60 \
--late_threshold 0.15 \
--cumulative_method step \
--plot \
video \
--video_path  sample_videos/sample_17_jester.avi
```

## Citation
Parts of the code are from the github repository of the article:

```bibtex
@article{kopuklu2019resource,
  title={Resource Efficient 3D Convolutional Neural Networks},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Kose, Neslihan and Gunduz, Ahmet and Rigoll, Gerhard},
  journal={arXiv preprint arXiv:1904.02422},
  year={2019}
}
```
