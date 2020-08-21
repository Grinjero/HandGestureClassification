python main.py --root_path "" \
	--video_path "D:\MachineLearning\Datasets\Jester\20bn-jester-v1" \
	--annotation_path "D:\FilipFaks\DiplomskiRad\Efficient-3DCNNs\annotation_Jester\jester.json" \
	--result_path "results\mobilenet" \
	--dataset jester \
	--n_classes 27 \
	--model slow_mobilenetv2 \
	--width_mult 1 \
	--train_crop random \
	--sample_duration 16 \
	--downsample 2 \
	--batch_size 1 \
	--n_threads 1 \
	--n_val_samples 1 \
	--checkpoint 1