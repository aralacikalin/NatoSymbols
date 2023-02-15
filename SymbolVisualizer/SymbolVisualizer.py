
import argparse
import os

import sys
import cv2
import numpy as np
import imutils

def argParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yoloText', type=str, help='path of the yolo output text')
    parser.add_argument('--image', type=str, help='path of the image')
    parser.add_argument('--classTemplates', type=str, help='path of the template of classes')
    parser.add_argument('--useOriginalClassColors', type=int,default=0, help='path of the template of classes')
    opt = parser.parse_args()
    return opt

def getYoloOutput(filePath):

    yoloOutput=[]
    with open(filePath) as f:
        for line in f:
            output=line.split(" ")
            outputConverted=[float(i) for i in output]
            outputConverted[0]=int(outputConverted[0])
            yoloOutput.append(outputConverted)

    return yoloOutput

def OrganizeYoloOutput(yoloOutput):
    newOutput=[]
    for out in yoloOutput:
        newItem=[]
        #class
        newItem.append(out[0])
        #bbox info
        newItem.append([out[1],out[2],out[3],out[4]])
        # when yolo outputs the angle we can get it like this
        try:
            newItem.append(int(out[5]))
        except IndexError:
            newItem.append(0)

        newOutput.append(newItem)
    return newOutput


def getClasses(classPath):
    classesDict={}
    classesDictRed={}
    files=os.listdir(classPath)
    for path in files:
        classNumber=int(path.split(".")[0])
        fullPathImage=classPath+"/"+path
        fullPathImage=fullPathImage.replace("\\","/")
        #TODO get only a mask of the symbols
        symbol=cv2.imread(fullPathImage,cv2.IMREAD_GRAYSCALE)
        symbolRed=cv2.imread(fullPathImage)
        classesDict[classNumber]=symbol
        classesDictRed[classNumber]=symbolRed
    return classesDict,classesDictRed



def VisualizeSymbol(symbolsImage,boundingBoxCoordinates,symbolRotation,symbolClass,classes,classesOriginal,symbolsImageOriginal):
    x,y,w,h=boundingBoxCoordinates

    symbolRotation=-symbolRotation
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
    # copyClassImg=cv2.resize(copyClassImg,(newX,newY),interpolation = cv2.INTER_AREA)
    rotatedClass=imutils.rotate_bound(copyClassImg,symbolRotation)
    rotatedClassOriginal=imutils.rotate_bound(classesOriginal[symbolClass],symbolRotation)
    # cv2.imshow("rotatedClass",rotatedClass)
    # cv2.waitKey()
    # rotationMatrix= cv2.getRotationMatrix2D((classCols/2,classRows/2),symbolRotation,1) 
    # rotatedClass= cv2.warpAffine(copyClassImg,rotationMatrix,(classCols,classRows)) 
    # cv2.imshow("rotatedClass",rotatedClass)

    #? getting the new bbox for rotated template class
    nonZeroIndexesRotatedClass=np.where(rotatedClass!=0)
    nonZeroIndexesRotatedClassT=np.where(rotatedClass.T!=0)
    topBound=nonZeroIndexesRotatedClass[0][0]
    bottomBound=nonZeroIndexesRotatedClass[0][-1]

    leftBound=nonZeroIndexesRotatedClassT[0][0]
    rightBound=nonZeroIndexesRotatedClassT[0][-1]
    rotatedClass=rotatedClass[topBound:bottomBound,leftBound:rightBound].copy()
    rotatedClassOriginal=rotatedClassOriginal[topBound:bottomBound,leftBound:rightBound].copy()
    # cv2.imshow("rotatedClassOriginal",rotatedClassOriginal)
    # cv2.waitKey()


    # print(leftBound,rightBound,topBound,bottomBound,rotatedClass.shape)

    # print(rotatedClass.shape)

    # cv2.waitKey()

    # cv2.imshow("rotatedClassnewBBox",rotatedClass)

    resizedClass=cv2.resize(rotatedClass,(newX,newY),interpolation = cv2.INTER_AREA)
    resizedClassOriginal=cv2.resize(rotatedClassOriginal,(newX,newY),interpolation = cv2.INTER_AREA)

    # _, binaryClass= cv2.threshold(resizedClass, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # print(resizedClass.shape,symbolsImage[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd].shape)
    # cv2.imshow("resized",resizedClass)
    # cv2.imshow("symbol",symbolsImage[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd])
    # cv2.waitKey()
    
    binaryClass=cv2.bitwise_not(resizedClass)
    whiteBackground=np.array(resizedClassOriginal.shape,dtype=resizedClassOriginal.dtype)

    resizedClassOriginal[binaryClass!=0]=(255,255,255)
    # resizedClassOriginal=cv2.bitwise_and(resizedClassOriginal,whiteBackground,mask=cv2.bitwise_not(binaryClass))


    
    symbolsImage[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd]=cv2.bitwise_and(symbolsImage[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd],binaryClass)
    # symbolsImageOriginal[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd]=cv2.bitwise_and(symbolsImageOriginal[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd],resizedClassOriginal,mask=cv2.bitwise_not(binaryClass))
    #? works for overlapping detections 
    symbolsImageOriginal[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd]=cv2.bitwise_and(symbolsImageOriginal[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd],resizedClassOriginal)
    # symbolsImage[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd]=binaryClass

    # cv2.imshow("symbolsImage",symbolsImage[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd])
    # cv2.imshow("symbolsImageOriginal",symbolsImageOriginal[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd])
    # cv2.imshow("resizedClassOriginal",resizedClassOriginal)
    # cv2.imshow("wholeImageOriginal",symbolsImageOriginal)
    # cv2.imshow("wholeImage",symbolsImage)
    # cv2.imshow("binaryClass",binaryClass)
    # cv2.waitKey()
    # cv2. destroyAllWindows()
    # symbolsImage[symbolyStart:symbolyEnd,symbolXStart:symbolXEnd]=resizedClassSymbol

        # symbolsImage = cv2.putText(symbolsImage, str(symbolClass), (x, y),cv2.FONT_HERSHEY_SIMPLEX, 2.6, (255), 4)

    # cv2.rectangle(symbolsImage,(0,0),(w,h),(1), 10)
    # cv2.imshow("Test",img)
    
    # symbolsImage=cv2.bitwise_not(symbolsImage) #!invert the image
    
    # cv2.waitKey()
    # cv2. destroyAllWindows()

    



def main():
    opt=argParser()
    imageName=os.path.basename(opt.yoloText)

    yoloOutput=getYoloOutput(opt.yoloText)
    yoloOutput=OrganizeYoloOutput(yoloOutput)
    useOriginalClassColors=opt.useOriginalClassColors
    image = cv2.imread(opt.image)

    imageSize=image.shape

    print(imageSize)
    # exit()
    # imageSize=(3472,4640)
    # imageSize=(770,578)
    imageSize=(imageSize[1],imageSize[0])
    w,h=imageSize
    symbolsImageOriginal=np.ones((h,w,3),dtype=np.uint8)
    symbolsImageOriginal*=255
    symbolsImage=np.ones((h,w),dtype=np.uint8)
    symbolsImage*=255


    # symbolsImage=cv2.bitwise_not(symbolsImage)
    # cv2.imshow("Test",symbolsImage,)
    # cv2.waitKey()
    # cv2. destroyAllWindows()


    classesImages,classesImagesRed=getClasses(opt.classTemplates)

    for out in yoloOutput:
        VisualizeSymbol(symbolsImage,out[1],out[2],out[0],classesImages,classesImagesRed,symbolsImageOriginal)

#?if classes are not red and want a red overlay
    if(not useOriginalClassColors):

        redImage=symbolsImage.copy()
        redImage=np.zeros((h,w,3),dtype=np.uint8)
        # redImage=cv2.cvtColor(symbolsImage, cv2.COLOR_GRAY2RGB)
        print(redImage.shape)
        print(symbolsImage.shape)

        redImage[:,:,2]=cv2.bitwise_not(symbolsImage)
        # redImage=cv2.bitwise_not(redImage)
        blackPixels=np.where(cv2.bitwise_not(symbolsImage)!=0)
    #     blackOnRed=np.where(
    #     (redImage[:, :, 0] == 0) & 
    #     (redImage[:, :, 1] == 0) & 
    #     (redImage[:, :, 2] == 0)
    # )
    #     print(blackOnRed)
    #     redImage[blackOnRed]=[255,255,255]
        # cv2.imshow("redImage",redImage)

        # redImage[blackPixels]=255

        # image[cv2.bitwise_not(symbolsImage)>0]=0
        # image += redImage*(cv2.bitwise_not(symbolsImage)>0)  
        img_bg = cv2.bitwise_or(redImage, redImage, mask=cv2.bitwise_not(symbolsImage))
        # img_fg = cv2.bitwise_or(image, image, mask=symbolsImage)
        # image[:,:,2]=symbolsImage
        # cv2.imshow("symbolsImage",cv2.bitwise_not(symbolsImage))
        

        image[blackPixels[0], blackPixels[1], :] = img_bg[blackPixels[0],blackPixels[1],:]
        # image[blackPixels[0], blackPixels[1], :] = [0, 0, symbolsImage[blackPixels[0], blackPixels[1]]]
        finalImg=cv2.add(image,img_bg)

#? if red classes are presented
    else:
        thresh_type = cv2.THRESH_BINARY


        thresholdVal,_ = cv2.threshold(symbolsImage,0,255,thresh_type+cv2.THRESH_OTSU)
        _,symbolsImage=cv2.threshold(symbolsImage,thresholdVal,255,thresh_type)
        # cv2.imshow("symbolsImageOriginal",cv2.bitwise_not(symbolsImageOriginal))

        img_bg = cv2.bitwise_and(symbolsImageOriginal, symbolsImageOriginal, mask=cv2.bitwise_not(symbolsImage))

        img_fg = cv2.bitwise_and(image, image, mask=symbolsImage)
        finalImg=cv2.add(img_bg,img_fg)

    # cv2.imshow("img_bg",img_bg)
    # cv2.imshow("img_fg",img_fg)
    # cv2.imshow("finalImg",finalImg)
    
    # cv2.waitKey()
    # exit()
    
    if(not os.path.exists()):
        os.mkdir("./VisualizedDetections")

    cv2.imwrite(f"VisualizedDetections/{imageName}Visualization.png",finalImg)
    cv2.imwrite(f"VisualizedDetections/{imageName}Visualization-original.png",symbolsImageOriginal)
    # Test arguments: --yoloText D:\Workplace\Symbols\YOLO-Detection\yolo-output\exp3\labels\IMG_20221031_154620.txt --classTemplates ".\VisualizerClasses"
    #? with new arguments
    # Test arguments: --yoloText ./GeneratorFiles/GeneretedYoloLabels/yolo-img29.txt --image ./GeneratorFiles/imgs/img29.jpg --classTemplates .\VisualizerClassesOriginalRed\ --useOriginalClassColors 1
    #! use these for demo
    # Test arguments: D:/Miniconda3.7/envs/symbols3/python.exe ./SymbolVisualizer.py --yoloText .\example\examplefalseremoved.txt  --image .\example\example.jpg --classTemplates .\VisualizerClassesOriginalRed\ --useOriginalClassColors 1
        

main()