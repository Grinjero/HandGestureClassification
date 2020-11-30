# Hand gesture recognition
Hand gesture recognition from RGB videos based on https://github.com/okankop/Efficient-3DCNNs <br>
3D CNNs are used, namely, a 3D version of MobileNetV2 and a SlowFast model whose fast and slow pathways 
consist of modified MobileNetV2 networks. Each model is described in more detail in its corresponding 
file in the ``models`` folder. <br>

## Models
Trained models can be downloaded [here](https://drive.google.com/drive/folders/1SoPDD3dnrDaj9lWn0mxvpRD4OLyD9mls?usp=sharing)
<br>Models are trained to recognize a subset of 17 gestures from the Jester dataset. 
Full list of gesture can be found in ``annotation_Jester\classInd_17_classes.txt``.

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
Examples for model training and evaluation can be seen in ```jupyter_notebooks\MbNet_SlowFast.ipynb```. 
To turn off online inference press 'q'.

Inference in action can be seen in the ``result_videos`` folder.
 
Run online inference with SlowFast from webcamera:
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

Run online inference with SlowFast on video:
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

Run online inference with MBNetV2 from webcamera:
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
Run online inference with MBNetV2 from video:
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
--early_threshold 0.4 \
--late_threshold 0.10 \
--cumulative_method step \
--plot \
video \
--video_path  sample_videos/sample_17_jester.avi
```

## Results

|         Model        | Number of parameters | MFLOPs | Forward pass frequency | Accuracy |
|:--------------------:|:--------------------:|:------:|:----------------------:|:--------:|
|      MobileNetV2     |         2.38M        |   444  |           24           |   95.05  |
| SlowFast MobileNetV2 |         2.5M         |   478  |           16           |   95.56  |

Speed measured on a NVIDIA GeForce GTX 1650 graphics card.

## Citation
Code is build upon github repositories of articles:

```bibtex
@article{DBLP:journals/corr/abs-1904-02422,
  author    = {Okan Köpüklü} and
               Neslihan Kose and
               Ahmet Gunduz and
               Gerhard Rigoll},
  title     = {Resource Efficient 3D Convolutional Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1904.02422},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.02422}
}
```
```bibtext
@article{DBLP:journals/corr/abs-1901-10323,
  author    = {Okan Köpüklü} and
               Ahmet Gunduz and
               Neslihan Kose and
               Gerhard Rigoll},
  title     = {Real-time Hand Gesture Detection and Classification Using Convolutional
               Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1901.10323},
  year      = {2019}
}
```
