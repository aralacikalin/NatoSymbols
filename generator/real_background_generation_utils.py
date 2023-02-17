
import glob
from generator_utils import *



def generate_image_with_real_background(boundingBoxesToRemove,sample,canvas,
                   dim = (4624,3468)):
    # canvas = np.full(dim, 255) #Size of final image
    canvas=canvas.copy()
    
    location_placement = []
    locations = [] #Store location of the symbols
    locations_units = []
    
    labels = [] #Store labels
    labels_units = []
    
    rotations = []
    

    sample_labels = list(sample.keys())

    for symbolRange in boundingBoxesToRemove: # Nr of symbols on image
        label = sample_labels[randint(0,len(sample_labels)-1)]
        startRangePoint,endRangePoint=symbolRange
        dim=(endRangePoint[0]-startRangePoint[0],endRangePoint[1]-startRangePoint[1])
        img_scale = random.uniform(0.8,1.2)
        img = get_random(label, sample)
        img = resize_by_scale(img, img_scale)

        #? to fit the symbol to original symbol bounding box, but stretches symbol, so not correct. 
 
        img, rotation = augment(img, apply_rotation=True, apply_transformation=True, apply_boldness=True, scale_to_binary = True)
        symbolDim=(img.shape[1],img.shape[0])
        if(symbolDim[0]>dim[0] or symbolDim[1]>dim[1]):

            widthRatio=dim[0]/symbolDim[0]
            heightRatio=dim[1]/symbolDim[1]
            if(widthRatio<heightRatio):
                #todo use widthRation to resize
                img = cv2.resize(img, (int(round(img.shape[1]*widthRatio)),int(round(img.shape[0]*widthRatio))))
                symbolDim=(img.shape[1],img.shape[0])

                newPlacementX=dim[0]
                newPlacementY=dim[1]
            else:
                img = cv2.resize(img, (int(round(img.shape[1]*heightRatio)),int(round(img.shape[0]*heightRatio))))
                symbolDim=(img.shape[1],img.shape[0])

                newPlacementX=dim[0]
                newPlacementY=dim[1]
        

        



        labels.append(label)
        rotations.append(rotation)
        
        #Check if there is overlap with current symbols.
        #If there is overlap the generate new locations and check again.

        #? to counteract the -10 in get points and not to get a empty range
        newPlacementX=dim[0]+11
        newPlacementY=dim[1]+11
        point1, point2 = get_points((newPlacementY,newPlacementX), img, locations, locations_units,location_placement)
        point1=startRangePoint[1]+point1
        point2=startRangePoint[0]+point2


        #? checks if with plus 15 will it be out of canvas
        if(point1+img.shape[0]>canvas.shape[0]):
            point1-=15
        elif(point1+img.shape[0]<canvas.shape[0]):
            point1+=15
            
        if(point1+img.shape[1]>canvas.shape[1]):
            point2-=15
        elif(point1+img.shape[0]<canvas.shape[0]):
            point2+=15
            

        #We append upper left corner point and lower right corner point.
        locations.append(((point1,point2),(point1+img.shape[0],point2+img.shape[1])))
        #If there is overlap we don't want to overwrite black pixels with white background.
        canvas = place_symbol(canvas, img, point1, point2)
 
    return canvas, locations, labels, rotations, locations_units, labels_units

def ProcessBackgrounds(backgroundPath):
    imagesPath=glob.glob(backgroundPath+"/*.jpg")
    imagesPath+=glob.glob(backgroundPath+"/*.png")
    backgroundImageList=[]
    for imagePath in imagesPath:
        imageName=os.path.basename(imagePath).split(".")[0]
        parentFolderPath=os.path.dirname(imagePath)
        image=cv2.imread(imagePath,0)
        removedLabelsImage=image.copy()
        imageshape=image.shape


        boundingBoxesToRemove=[]
        with open(parentFolderPath+"/"+imageName+".txt") as labelFile:
            for label in labelFile.readlines():
                label=label.split(" ")
                boundingBoxCoordinatesInfoNormalized=ParseBoundingBoxInfo(label)
                boundingBoxPoint=GetBoundingBoxPoints(boundingBoxCoordinatesInfoNormalized,imageshape)
                boundingBoxesToRemove.append(boundingBoxPoint)
        for boundingBox in boundingBoxesToRemove:
            startPoint=boundingBox[0]
            endPoint=boundingBox[1]
            removedLabelsImage=cv2.rectangle(removedLabelsImage, startPoint, endPoint, (255, 255, 255), -1)

        dim=(imageshape[0],imageshape[1])
        backgroundImageList.append([removedLabelsImage,boundingBoxesToRemove,dim])
    return backgroundImageList

def ParseBoundingBoxInfo(label):
    boundingBoxCoordinates=[float(label[1]),float(label[2]),float(label[3]),float(label[4])]
    return boundingBoxCoordinates

def GetBoundingBoxPoints(boundingBoxCoordinatesInfoNormalized,imageshape):
    x,y,w,h=boundingBoxCoordinatesInfoNormalized

    imageW=imageshape[1]
    imageH=imageshape[0]

    x=int(x*imageW)
    w=int(w*imageW)

    y=int(y*imageH)
    h=int(h*imageH)

    symbolXStart=int(x-w/2)
    symbolyStart=int(y-h/2)
    symbolXEnd=int(x+w/2)
    symbolyEnd=int(y+h/2)
    return[(symbolXStart,symbolyStart),(symbolXEnd,symbolyEnd)]
