This folder contains the modifications to `val.py`, `train.py`, `utils/metrics.py`, `utils/plots.py`, `utils/torch_utils.py` and yaml files for training a YOLO model.

The contents of this folder can be directly copied to yolov5 folder which is cloned from https://github.com/ultralytics/yolov5.git to use the modified version of the scripts and rest of the files.

`val_symbols_detection.py` is created from `val.py` to output the images with predictions with the given confidence threshold. These images are created when `--save-txt` argument is given. Also the output results table now also includes the results at given confidence threshold and at the max f1 score confidence threshold, and these will be saved as text files into the experiment folder.

`utils/metrics.py` is modified and contains functions to calculate symbol detection metrics.
`train.py` and `utils/torch_utils.py` is modified modifed to have early stopping tolerance, currently it is harcoded in the line 255 of `train.py` the variable name which adjusts this tolerance is `min_delta`.`
in `plots.py` a function to create bounding boxes with pink overlay if the detection is FP is implemented. This function is used on `val_symbols_detection.py` to visualize the detections. 


example usage of `val_symbols_detection.py` 

```
python val.py --weights <checkpoint_path> --data <path_to_data_yaml> --conf-thres <confidence_threshold> --name <experiment_name> --save-txt --task test --augment
```


To train a yolo model you need to create data from the generator.

To make the generator work the apropiate folders should be there, for more information go to `generator/README.md`

example generator usage

```
python generate.py \
 --examples_nr 40000 \
 --save_images_dir data/NatoSymbols/images/train \
 --save_labels_dir data/NatoSymbols/labels/train \
 --save_rotations_dir data/NatoSymbols/rotations/train \
 --real_backgrounds_ratio 0.3 \
 --real_symbols_ratio 0.5 \
 --save_dim_h 1156 --save_dim_w 1540
```

for training yolo you will need train and val data, which needs to be separate.

after generating the required data do not forget to change the `path: /data/NatoSymbols` to the correct path if its not correct in `data/NATO-Symbols.yaml`

`data/hyps/hyp.NatoSymbols.yaml` file has the augmentations settings for the training, all augmentations configurations are default except HSV augmentations, they are disabled.

example of how to train YOLOv5

```
python train.py \
--data data/NATO-Symbols.yaml \
--weights yolov5l.pt \
--epochs 300  \
--img 640 \
--name  LargeModel \
--batch 64 \
--hyp data/hyps/hyp.NatoSymbols.yaml \

```