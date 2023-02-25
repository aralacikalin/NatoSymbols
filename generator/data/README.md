Drive folder for images https://drive.google.com/drive/folders/1Th9QduyTqunjkDcdwHUFlOWjdMa-ae9N

The images that are used by generate.py must be stored in 3 separete folders. There must be a folder for mission task symbols, unit symbols and extras.

The mission task symbols folder contains all the desired mission task symbols from which the generator uniformly samples. There must be atleast one symbol in this folder. For each mission task in that folder, there must be corresponding mission task's name in labels.txt. The mission tasks order in labels.txt is used to convert string to integer. By default it is alphabetical order.

The unit symbols folder is more strict than mission task symbols folder. There must be at least one symbol for each label in maneuver_units, support_units and unit_sizes. (Will add these as .txt files).

The extras contains all the other images which generator needs. There must be atleast one example of each symbol. (Will add .txt file to check if all the symbols are present)
