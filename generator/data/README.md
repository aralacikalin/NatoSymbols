Download the contents of this folder from https://drive.google.com/file/d/1dXqRhmRNoUScS8EHUAEb_anMPkWHCwcO/view?usp=drive_link

The images that are used by generate.py must be stored in 3 separate folders.

"background elements" folder contains templates for background symbols in the noise folder and elements that can be combined to generate composite background symbols like location markers and unit symbols. In the unit symbols folder there must be at least one symbol for each label in maneuver_units, support_units and unit_sizes. (Will add these as .txt files).

The templates folder contains mission task symbol examples. It contains folders with all the desired mission task symbols from which the generator uniformly samples. For each mission task type, there must be corresponding mission task's name in labels.txt. The mission tasks order in labels.txt is used to convert string to integer. By default it is alphabetical order. If you want to add a symbols type or remove a symbol type, you must add a folder with images and also add the symbol name to the labels.txt.

The real data folder contains extracted elements from the real annex C films. There are folders for training (originating from 71 images) and validation (11 images). For extracted symbols, a cleaned version (without background clutter, in folder with tag "Clean") must exist which can be used to rediscover the bounding box after rotation.

