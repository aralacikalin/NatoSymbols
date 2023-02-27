This is a generator for generating training images with labels. The generator samples uniformly over the unique labels.

The generator can be used with following command
```
python generate.py
```
This produces one example and stores the image in images/train folder. To create more examples than one, then run
```
python generate.py --examples_nr 10
```
More info about the possible arguments can be found in generate.py.

```
python generate.py --examples_nr 10 --real_backgrounds_ratio 0.2
```
this produces 2 images out of 10 with real background
