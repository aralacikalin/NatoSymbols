This is a tool for labeling the rotations of the symbols. Use this after labeling the data.

# Video Documentation for Labeling



https://github.com/aralacikalin/NatoSymbols/assets/43911164/58e607b6-cece-4b00-bf8f-ad55b9462ad2


https://github.com/aralacikalin/NatoSymbols/assets/43911164/72096588-0862-47d2-bf5e-e74d20df4f25



```
python SymbolRotationLabeler.py
```

- Program will as if you want to label a single file or multiple files.
- Then select the location using the folder explorer.

- the template image will be overlayed with the original symbol, turn the template left or right with the usage of ```a``` and ```d``` keys of the keyboard.
- You can increase and decrease the rate of rotation for each step with ```w``` and ```s``` keys.
- if you press ```esc``` and if there is a yolo label file of the image or a rotations folder, it will use the rotations from those files. 
- when you press ```enter``` you will lock in the rotation for the symbol and pass to the next one.
-when the current image is fully labeled, program will output the rotations to the folder ```/rotations```



