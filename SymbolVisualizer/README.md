This script visualizes the detected symbols using the output label file of yolo. If that file includes rotations as the last collumn then it also rotates the symbols.
```
python SymbolVisualizer.py --images /path/to/image/folder --classTemplates ./VisualizerClassesOriginalRed/
```

- program assumes that the label text files have same names with the corresponding images
- program assumes that the used templates are named as their class indexes.
- if ```--useOriginalClassColors 0``` then a grayscaled version of the symbols will be placed if ```--useOriginalClassColors 1```, then the original colors of templates will be used.
- if ```--labels path/to/labels/folder``` is not given program assumes the label files are in the same folder as the images.
- if ```--useTrajectort True``` for trajectory symbols similarity prediction is used (Also need to unzip data files for now)

As a result this program overlayes the predicted symbols by using the templates located in the folder and saves that image to ```VisualizedDetections``` folder.



