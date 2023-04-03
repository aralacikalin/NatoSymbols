The utilty functions are described in the following table.

| Function | Describtion |
| ------------- | ------------- |
| class_combiner | Takes as the argument the labels file (plus output file) and the .txt file where the classes that are combined are in consecutive rows. If there is wish to combine into multiple classes then there needs to be empty row between the labels. |
| inverse_images | Inverses the images in the given folder. If the output_dir is the same as input_dir, then overwrites the images in the input folder.

The class_combiner example .txt file is combiner.txt. In the case of combiner.txt the labels cover, guard, screen will be combined into one and the labels retire, delay, withdraw will be combined into one. After the classes are combined, make sure that the <data_yaml> file is correct.
