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

`--real_backgrounds_ratio <ratio>` argument will produce `<ratio>` amount of images with real backgrounds. For this to work `data\real_backgrounds` should contain the real background images with their label files or a custom path can be given with `--real_backgrounds_dir <path>` argument.

`--real_symbols_ratio <ratio>` makes the `<ratio>` amount of symbols to be from the real data. For this to work `data/real_symbols` and `data/real_clean_symbols` should have folder that has the real data symbols example of the hierarchy `data/real_symbols/RealTrainSymbols` and `data/real_clean_symbols/RealTrainSymbolsCleaned` or you can give other paths with `--real_symbols_dir <path>` and  `--real_symbols_clean_dir <path>` but that path should contain at least 1 folder which has the symbols. The names of the real symbols and cleaned version of those should have the same names.

Giving `--real_symbols_in_real_backgrounds` argument will enable generation of real symbols on real backgrounds if real background generation is enabled.
