# Validation

This validation script outputs the detections on the image, overlays the detection with red color if its false positive. Saves the metric scores. These metrics include Precision, Recall, mAP50, mAP50-95, P@R(80), R@P(80)

(To understand the metrics please check [this thesis](https://comserv.cs.ut.ee/ati_thesis/datasheet.php?id=77397))

Use the script like this 

```
python .\standAloneVal.py --predictions <path_to_predictions> --data "<path_to_data_yaml>" --project <project_folder_path> --name <experiment_name> --conf-thres 0.7 --save-txt 
```

## Argument Explanations

### --conf-tresh

`--conf-tresh` is used on the visualization of detections to limit the bounding boxes drawn to be higher than the set confidence threshold because if labels include detections with 0.001 confidence that would create very crowded and not understandable visualizations.

### --predictions

`--predictions` needs to have text files with image names of the corresponding image. The text files should have the labels in the form of `<class index> <normalized centerX> <normalized centerY> <normalized width> <normalized height> <confidence of prediction>` if YOLO is used val.py can output the prediction labels in this form exactly. 

(P.S. it outputs this when `save_txt` and `save_conf` is used as arguments on validation)


Example of label file
```
2 0.441754 0.327459 0.107525 0.0645831 0.945121
6 0.544107 0.245101 0.0473278 0.0535744 0.903171
27 0.593751 0.214678 0.0264113 0.053698 0.88002
27 0.586679 0.252489 0.0528301 0.0560029 0.209475
16 0.627149 0.237434 0.0453735 0.0342767 0.0228279
26 0.627053 0.237503 0.0455242 0.0338723 0.014535
11 0.62665 0.237173 0.0466324 0.0337543 0.00802473

```

### --data

`--data` needs a yaml file in the form of YOLO data yaml files.

Ideally the data folder structure would be;

```
data
|-train
    |-images
    |-labels
|-val
    |-images
    |-labels
|-test
    |-images
    |-labels

```
the scripts finds the labels automaticly on the directory when images path is given. 

Example yaml file
```
path:  .\data
train: train\images
val:  val\images
test:  test\images

nc: 31
names: ["advance_to_contact", "ambush", "attack", "attack_by_fire", "block", "breach", "clear", "contain", "control", "counterattack", "cover", "delay", "deny", "destroy", "disrupt", "fix", "guard", "isolate", "main_attack", "neutralize", "occupy", "penetrate", "retain", "retire", "screen", "secure", "seize", "support_by_fire", "suppress", "turn", "withdraw"]

```
