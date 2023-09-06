from tkinter.filedialog import askdirectory
import glob
import os
import collections

import cv2
import imutils
import numpy as np
from scipy import ndimage

def ExtractSymbol(image,boundingBoxCoordinates):

    x,y,w,h=boundingBoxCoordinates

    imageW=image.shape[1]
    imageH=image.shape[0]

    x=int(x*imageW)
    w=int(w*imageW)

    y=int(y*imageH)
    h=int(h*imageH)


    # img = cv2.rectangle(symbolsImage, (x1, y1 - 20), (x1 + w, y1), color, -1)
    symbolXStart=int(x-w/2)
    symbolyStart=int(y-h/2)
    symbolXEnd=int(x+w/2)
    symbolyEnd=int(y+h/2)
    symbolCopy=image[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd].copy()
    return symbolCopy


classIndexToName={
    0:"advance_to_contact",
    1:"ambush",
    2:"attack",
    3:"attack_by_fire",
    4:"block",
    5:"breach",
    6:"clear",
    7:"contain",
    8:"control",
    9:"counterattack",
    10:"cover",
    11:"delay",
    12:"deny",
    13:"destroy",
    14:"disrupt",
    15:"fix",
    16:"guard",
    17:"isolate",
    18:"main_attack",
    19:"neutralize",
    20:"occupy",
    21:"penetrate",
    22:"retain",
    23:"retire",
    24:"screen",
    25:"secure",
    26:"seize",
    27:"support_by_fire",
    28:"suppress",
    29:"turn",
    30:"withdraw"

}

symbolIndexes={ key:0 for key,val in classIndexToName.items()}

folder: str = askdirectory()
folderPath=folder+"/"
rotationsPath=folderPath+"rotations/"

subdirs=os.walk(folderPath)


subDatasetFolderNames=[*subdirs][0][1]
extractedSymbolsPath="./ExtractedSymbols"

if(not os.path.exists(extractedSymbolsPath)):
    os.mkdir(extractedSymbolsPath)
imagePaths=[]
for folderName in subDatasetFolderNames:
    imagesPaths=glob.glob(folderPath+folderName+"/*.jpg")
    imagesPaths+=glob.glob(folderPath+folderName+"/*.png")
    imagePaths.append([folderName,imagesPaths])

allLabels=[]
for datasetname, datasetImages in imagePaths:
    print(datasetname)
    for imagePath in datasetImages:
        image = cv2.imread(imagePath)
        imageShape=image.shape
        imageName=os.path.basename(imagePath)
        imageBasePath=imagePath.replace(imageName,"")
        imageName=imageName.split(".")[0]
        labelPath=imageBasePath+imageName+".txt"

        rotationsPath=imageBasePath+"rotations/"+imageName+".txt"
        if(not os.path.exists(labelPath)):
            continue
            
        with open(labelPath) as labelFile:
            # if there are rotations
            if(os.path.exists(rotationsPath)):
                with open(rotationsPath) as rotationFile:
                    for symbolInfo,rotation in zip(labelFile.readlines(),rotationFile.readlines()):
                        symbolInfo=symbolInfo.split(" ")
                        symbolLabel=int(symbolInfo[0])
                        boundingBoxCoordinates=[float(symbolInfo[1]),float(symbolInfo[2]),float(symbolInfo[3]),float(symbolInfo[4])]
                        rotation=int(rotation)
                        extractedSymbol=ExtractSymbol(image,boundingBoxCoordinates)
                        # extractedSymbol=imutils.rotate_bound(extractedSymbol,rotation)
                        #? threshold that works for these extracted symbols
                        # extractedSymbol=cv2.cvtColor(extractedSymbol, cv2.COLOR_BGR2GRAY)
                        
                        extractedSymbol=np.min(extractedSymbol,axis=2)
                        thresh_type = cv2.THRESH_BINARY

                        thresholdVal,_ = cv2.threshold(extractedSymbol,0,255,thresh_type+cv2.THRESH_OTSU)

                        
                        # extractedSymbol = ndimage.rotate(extractedSymbol, -rotation, mode='constant',cval=255) #resetting rotations
                        extractedSymbol = ndimage.rotate(extractedSymbol, -rotation, mode='constant',cval=255)
                        # thresholdVal,extractedSymbol = cv2.threshold(extractedSymbol,thresholdVal,255,thresh_type)
                        extractedSymbol[extractedSymbol <= 150] = 0
                        extractedSymbol[extractedSymbol > 150] = 255

                        extractedSymbolsDatasetPath=extractedSymbolsPath+"/"+datasetname
                        if(not os.path.exists(extractedSymbolsDatasetPath)):
                            os.mkdir(extractedSymbolsDatasetPath)
                        
                        # cv2.imshow("extractedSymbol",extractedSymbol)
                        # key = cv2.waitKey(0)
                        symbolIndexes[symbolLabel]+=1
                        cv2.imwrite(f"{extractedSymbolsDatasetPath}/{classIndexToName[symbolLabel]}{symbolIndexes[symbolLabel]}.png",extractedSymbol)
                        # print(classIndexToName[symbolLabel])
            else:
                for symbolInfo in labelFile.readlines():
                        symbolInfo=symbolInfo.split(" ")
                        symbolLabel=int(symbolInfo[0])
                        boundingBoxCoordinates=[float(symbolInfo[1]),float(symbolInfo[2]),float(symbolInfo[3]),float(symbolInfo[4])]
                        extractedSymbol=ExtractSymbol(image,boundingBoxCoordinates)
                        # extractedSymbol=imutils.rotate_bound(extractedSymbol,rotation)
                        #? threshold that works for these extracted symbols
                        # extractedSymbol=cv2.cvtColor(extractedSymbol, cv2.COLOR_BGR2GRAY)
                        extractedSymbol=np.min(extractedSymbol,axis=2)
                        thresh_type = cv2.THRESH_BINARY
                        thresholdVal,_ = cv2.threshold(extractedSymbol,0,255,thresh_type+cv2.THRESH_OTSU)

                        extractedSymbol = ndimage.rotate(extractedSymbol, 0, mode='constant',cval=255)
                        thresholdVal,extractedSymbol = cv2.threshold(extractedSymbol,thresholdVal,255,thresh_type)
                        # extractedSymbol[extractedSymbol <= 150] = 0
                        # extractedSymbol[extractedSymbol > 150] = 255

                        extractedSymbolsDatasetPath=extractedSymbolsPath+"/"+datasetname
                        if(not os.path.exists(extractedSymbolsDatasetPath)):
                            os.mkdir(extractedSymbolsDatasetPath)
                        
                        # cv2.imshow("extractedSymbol",extractedSymbol)
                        # key = cv2.waitKey(0)
                        symbolIndexes[symbolLabel]+=1
                        cv2.imwrite(f"{extractedSymbolsDatasetPath}/{classIndexToName[symbolLabel]}{symbolIndexes[symbolLabel]}.png",extractedSymbol)
                        # print(classIndexToName[symbolLabel])

print("Extracted Symbols:")
totalSymbols=0
for key, val in symbolIndexes.items():
    totalSymbols+=val
    if(val==0):
        continue
    print(f"{classIndexToName[key]}:{val}")
print(f"Total Symbols Excracted: {totalSymbols}")
