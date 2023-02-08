This script alings the plastic film image to underlying map by detecting markers on the film
```
python ./MarkerDetectionWithROI.py -t ./testTemplate6.jpg -t2 ./crossTemplate7.jpg -i ./path/to/film/images
```

- The program will wask which marker you want to search.
- Then you need to select an approximate area on the image by clicking and dragging and finally pressing ```enter```
- Then script will find the exact location and show you.
- If the detected area is wrong you can press ```escape``` to see the next best detection.
- When you find a good match press ```enter``` to continue.
- Then you need to enter the coordinates of the marker.
- if you searched for hashtag marker then enter the coordinates in this order: top left, top right, bottom left, bottom right
- if there are at least 3 points detected the program will output the alingned image in the folder ```/PlacedOnMap```



