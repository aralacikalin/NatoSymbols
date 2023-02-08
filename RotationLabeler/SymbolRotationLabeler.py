import cv2
import glob
import os
from numpy import typing as ndt
import numpy as np
import imutils

from tkinter.filedialog import askdirectory,askopenfilenames


while(True):

    folderOrFileInput=input("Open Folder(1) or File(2): ")
    try:
        folderOrFileInput=int(folderOrFileInput)
        if(folderOrFileInput!=1 and folderOrFileInput!=2):
            print("Wrong input try again!")
            continue
        break
    except ValueError:
        print("Wrong input try again!")

if(folderOrFileInput==1):
    folder: str = askdirectory()
    folderPath=folder+"/"
    imagesPaths=glob.glob(folderPath+"*.jpg")
    rotationsPath=folderPath+"rotations/"



elif(folderOrFileInput==2):
    file: list[str] = list(askopenfilenames(filetypes=[
                    ("image", ".jpeg"),
                    ("image", ".png"),
                    ("image", ".jpg"),
                ]))
    filename=os.path.basename(file[0])
    folder=file[0].split(filename)[0]
    folderPath=folder
    imagesPaths=file
    rotationsPath=folderPath+"rotations/"

if(not os.path.exists(rotationsPath)):
    os.mkdir(rotationsPath)







# folderPath="../LabeledData/GoodFilmsCropped"

# imagesPaths=glob.glob("../LabeledData/GoodFilmsCropped/"+"*.jpg")
# imagesPaths=glob.glob(folderPath+"*.jpg")

# print(imagesPaths)

classesPath="../VisualizerClassesOriginalRed"
# rotationsPath="../LabeledData/GoodFilmsCropped/rotations/"
# rotationsPath="../test/rotations/"


def VisualizeSymbol(symbolsImage,boundingBoxCoordinates,defaultRotation,symbolClass,classes,classesOriginal,symbolsImageOriginal):
    LEFT_ARROW_KEY=97
    RIGHT_ARROW_KEY=100
    UP_ARROW_KEY=119
    DOWN_ARROW_KEY=115
    ENTER_KEY=13
    ESCAPE_KEY = 27
    x,y,w,h=boundingBoxCoordinates

    if defaultRotation is None:
        symbolRotation:int=0
    else:
        symbolRotation=-defaultRotation
    imageW=symbolsImage.shape[1]
    imageH=symbolsImage.shape[0]

    x=int(x*imageW)
    w=int(w*imageW)

    y=int(y*imageH)
    h=int(h*imageH)


    # img = cv2.rectangle(symbolsImage, (x1, y1 - 20), (x1 + w, y1), color, -1)
    symbolXStart=int(x-w/2)
    symbolyStart=int(y-h/2)
    symbolXEnd=int(x+w/2)
    symbolyEnd=int(y+h/2)
    # copyClassImg=classes[symbolClass].copy()
    thresh_type = cv2.THRESH_BINARY


    # copyClassImg=cv2.bitwise_not(classes[symbolClass].copy())
    thresholdVal,_ = cv2.threshold(classes[symbolClass],0,255,thresh_type+cv2.THRESH_OTSU)
    _,copyClassImg=cv2.threshold(classes[symbolClass],thresholdVal,255,thresh_type)
    copyClassImg=cv2.bitwise_not(copyClassImg)
    newX=symbolXEnd-symbolXStart
    newY=symbolyEnd-symbolyStart

    classCols=copyClassImg.shape[1]
    classRows=copyClassImg.shape[0]

    rotationVal=1
    while(True):
        rotatedClass=imutils.rotate_bound(copyClassImg,symbolRotation)
        rotatedClassOriginal=imutils.rotate_bound(classesOriginal[symbolClass],symbolRotation)
        #? getting the new bbox for rotated template class
        nonZeroIndexesRotatedClass=np.where(rotatedClass!=0)
        nonZeroIndexesRotatedClassT=np.where(rotatedClass.T!=0)
        topBound=nonZeroIndexesRotatedClass[0][0]
        bottomBound=nonZeroIndexesRotatedClass[0][-1]

        leftBound=nonZeroIndexesRotatedClassT[0][0]
        rightBound=nonZeroIndexesRotatedClassT[0][-1]
        rotatedClass=rotatedClass[topBound:bottomBound,leftBound:rightBound].copy()
        rotatedClassOriginal=rotatedClassOriginal[topBound:bottomBound,leftBound:rightBound].copy()
        resizedClass=cv2.resize(rotatedClass,(newX,newY),interpolation = cv2.INTER_AREA)
        resizedClassOriginal=cv2.resize(rotatedClassOriginal,(newX,newY),interpolation = cv2.INTER_AREA)

        thresh_type = cv2.THRESH_BINARY


        thresholdVal,_ = cv2.threshold(resizedClass,0,255,thresh_type+cv2.THRESH_OTSU)
        _,resizedClass=cv2.threshold(resizedClass,thresholdVal,255,thresh_type)
        binaryClass=cv2.bitwise_not(resizedClass)
        resizedClassOriginal[binaryClass!=0]=(255,255,255)
        symbolCopy=symbolsImage[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd].copy()
        # symbolCopy=cv2.bitwise_and(symbolsImage[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd],resizedClassOriginal)


        img_bg = cv2.bitwise_and(symbolCopy, symbolCopy, mask=binaryClass)

        img_fg = cv2.bitwise_and(resizedClassOriginal, resizedClassOriginal, mask=cv2.bitwise_not(binaryClass))
        symbolCopy=cv2.add(img_bg,img_fg)


        cv2.imshow("overlayed",symbolCopy)
        # cv2.imshow("templateRotated",resizedClassOriginal)
    
        key = cv2.waitKey(0)

        while(key!=ENTER_KEY and key!=27 and key!=LEFT_ARROW_KEY and key != DOWN_ARROW_KEY and key != UP_ARROW_KEY and key != RIGHT_ARROW_KEY):
            key = cv2.waitKey(0)
            continue
        # cv2.destroyAllWindows()
        
        #enter key to accept the rotation
        if key == ENTER_KEY:
            cv2.destroyAllWindows()
            break
        # end program with keyboard 'esc'
        elif key == LEFT_ARROW_KEY:
            symbolRotation-=rotationVal
            if(symbolRotation<=-359):
                symbolRotation=0
            

            print(f"Next Rotation={-symbolRotation}")
            continue
        elif key == RIGHT_ARROW_KEY:
            symbolRotation+=rotationVal
            if(symbolRotation>=0):
                symbolRotation=-359

            print(f"Next Rotation={-symbolRotation}")
            continue
        elif key == UP_ARROW_KEY:
            rotationVal+=1

            print(f"Interval={rotationVal}")
            continue
        elif key == DOWN_ARROW_KEY:
            rotationVal-=1

            print(f"Interval={rotationVal}")

            continue
        elif key == ESCAPE_KEY and defaultRotation is not None:
            print(f"Default Rotation is used={defaultRotation}")
            symbolRotation=-defaultRotation
            break

    cv2. destroyAllWindows()

    print(f"Final Rotation={-symbolRotation}")
    return -symbolRotation
    



def getClasses(classPath):
    classesDict ={}
    classesDictRed ={}
    files:list[str]=os.listdir(classPath)
    for path in files:
        classNumber=int(path.split(".")[0])
        fullPathImage: str=classPath+"/"+path
        fullPathImage=fullPathImage.replace("\\","/")
        #TODO get only a mask of the symbols
        symbol=cv2.imread(fullPathImage,cv2.IMREAD_GRAYSCALE)
        symbolRed=cv2.imread(fullPathImage)
        classesDict[classNumber]=symbol
        classesDictRed[classNumber]=symbolRed
    return classesDict,classesDictRed


def main():
    classesDict,classesDictRed  = getClasses(classesPath)

    for imagePath in imagesPaths:
        image = cv2.imread(imagePath)
        imageName=os.path.basename(imagePath)
        imageName=imageName.split(".")[0]

        with open(folderPath+imageName+".txt","r") as textFile:
            rotationsList=[]
            for i,row in enumerate(textFile.readlines()):
                yoloRow=row.split(" ")
                classIndex =int(yoloRow[0])
                boundingBoxCoordinates=[float(yoloRow[1]),float(yoloRow[2]),float(yoloRow[3]),float(yoloRow[4])]
                try:
                    rotation=int(yoloRow[5])
                except IndexError:
                    rotation=None
                    try:
                        with open(rotationsPath+imageName+".txt","r") as rotationFile:
                            rotations=rotationFile.readlines()
                            rotation=int(rotations[i])
                    except:
                        rotation=None

                angle=VisualizeSymbol(image,boundingBoxCoordinates,rotation,classIndex,classesDict,classesDictRed,None)
                rotationsList.append(angle)


            with open(rotationsPath+imageName+".txt","w+") as rotationFile:
                for rotation in rotationsList:
                    rotationFile.write(str(rotation)+"\n")
        print(f"Rotation File Saved To: {rotationsPath+imageName}.txt")
            

if __name__=="__main__":
    main()