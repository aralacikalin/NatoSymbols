This script visualizes the detected symbols using the output label file of yolo. If that file includes rotations as the last collumn then it also rotates the symbols.
```
python SymbolVisualizer.py --yoloText path/to/yolo/label.txt  --image /path/to/original/image.jpg --classTemplates ./VisualizerClassesOriginalRed/ --useOriginalClassColors 1
```

- if --useOriginalClassColors 0 then a grayscaled version of the symbols will be placed if --useOriginalClassColors 1, then the original colors of templates will be used.

Then it creates 2 versions of visualizations, one is overlayed on top of the original, one is only the detected visualizations.



