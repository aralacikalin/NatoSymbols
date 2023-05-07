# import the necessary packages
from collections import defaultdict
from functools import wraps
import io 
import multiprocessing
from operator import xor
import signal
import numpy as np
import numpy.typing as npt
import argparse
import imutils
import glob
import cv2
import matplotlib.pyplot as plt
import os
import contextily as ctx
import pygeodesy
import scipy.ndimage as ndi
import skan
import scipy
from numba import jit
import mgrs
import time
import matplotlib.image as mpimg


class BoundingBoxWidget():
    def __init__(self,image,windowName='Select Marker Area'):
        self.original_image = image
        self.clone = self.original_image.copy()
        self.windowName=windowName
        cv2.namedWindow(self.windowName,cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(self.windowName, cv2.WND_PROP_FULLSCREEN  , cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.windowName, self.extract_coordinates)
        cv2.imshow(self.windowName,self.clone )

        # Bounding box reference points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clone = self.original_image.copy()
            
            self.image_coordinates = [(x,y)]
            cv2.circle(self.clone,(x,y),2,(0,255,0),-1)
            cv2.imshow(self.windowName, self.clone)

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            if x<0:
                x=0
            if y<0:
                y=0

            if x>=self.original_image.shape[1]:
                x=self.original_image.shape[1]-1
            if y>=self.original_image.shape[0]:
                y=self.original_image.shape[0]-1
            self.image_coordinates.append((x,y))

            # Draw rectangle 
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 10)
            cv2.imshow(self.windowName, self.clone) 

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def GetBoundingBoxPoints(self):
        return self.image_coordinates

    def show_image(self):
        cv2.imshow(self.windowName, self.clone)
        return self.clone


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


#? Affline transformation matrix function for 4 points source: https://stackoverflow.com/questions/67109232/image-warping-python
def affine(x, y):
    # b = np.array([y[0][0], y[0][1], y[1][0], y[1][1], y[2][0], y[2][1], y[3][0], y[3][1],y[4][0], y[4][1]])

    b=np.array(y)
    b=b.flatten()
    # print("B: ",b)

    # A = np.array([
    #             [x[0][0], x[0][1], 1, 0, 0, 0],[0, 0, 0, x[0][0], x[0][1], 1],
    #             [x[1][0], x[1][1], 1, 0, 0, 0],[0, 0, 0, x[1][0], x[1][1], 1],
    #             [x[2][0], x[2][1], 1, 0, 0, 0],[0, 0, 0, x[2][0], x[2][1], 1],
    #             [x[3][0], x[3][1], 1, 0, 0, 0],[0, 0, 0, x[3][0], x[3][1], 1],
    #             [x[4][0], x[4][1], 1, 0, 0, 0],[0, 0, 0, x[4][0], x[4][1], 1]
    #             ])

    A=[]
    for point1,point2 in x:
        A.append([point1,point2,1,0,0,0])
        A.append([0,0,0,point1,point2,1])
    A=np.array(A)
    # print("A: ",A)

    coef = np.linalg.inv(A.T @ A) @ A.T @ b
    M = np.array([
                    [coef[0], coef[1], coef[2]],
                    [coef[3], coef[4], coef[5]]
                ], 
                dtype=np.float64)
    return M


def extractWebMercatorCoordinates(stringMGRSPoints):

    mercPoints=[]
    m = mgrs.MGRS()

    for point in stringMGRSPoints:
        llPoint=m.toLatLon(point)
        wmPoint=pygeodesy.toWm(*llPoint)
        mercPoint=[wmPoint.x,wmPoint.y]
        mercPoints.append(mercPoint)

    mercPoints=np.array(mercPoints)
    return mercPoints


def MapPreparing(image,stringMGRSPoints,detectedPoints,verbose=False,detectionImage=None):
    mercPointsOriginalScale=extractWebMercatorCoordinates(stringMGRSPoints)
    percentageMore=0.30
    height=image.shape[0]
    width=image.shape[1]

    originalImgShape=np.array([width,height])


    oMins=np.amin(mercPointsOriginalScale,axis=0)
    oMaxs=np.amax(mercPointsOriginalScale,axis=0)

    detectedPointsMins=np.amin(detectedPoints,axis=0)
    detectedPointsMaxs=np.amax(detectedPoints,axis=0)
    # print(mercPointsOriginalScale)
    # print(oMins)
    # print(oMaxs)
    # print("detedted")
    # print(detectedPointsMins)
    # print(detectedPointsMaxs)


    oNewImgShape=(oMaxs-oMins)

    detectedPointsShape=(detectedPointsMaxs-detectedPointsMins)

    oMinsShapeAddPercentage=detectedPointsMins/detectedPointsShape
    oMaxsShapeAddPercentage=(originalImgShape-detectedPointsMaxs)/detectedPointsShape

    # print(oNewImgShape)
    # exit()

    #TODO: calculate percentage add for all sides (left of the ROI, right, top, bottom)
    #? dynamicly calculate the imgshapeadds (percentage to add on all sides) to get more correct map for the result instead of a really big map
    # oImgShapeAdds=oNewImgShape*percentageMore



    oMinsShapeAdds=oNewImgShape*oMinsShapeAddPercentage
    oMaxsShapeAdds=oNewImgShape*oMaxsShapeAddPercentage

    #? this swtich is needed because on canvas coordinates get bigger when on right and bottom but on mgrs bottom gets smaller(y) and left gets bigger(x)
    correctedMinus=np.array([oMinsShapeAdds[0],oMaxsShapeAdds[1]])
    correctedPlus=np.array([oMaxsShapeAdds[0],oMinsShapeAdds[1]])

    oMinsShapeAdds=correctedPlus
    oMaxsShapeAdds=correctedMinus


    # WMCoordinatesBigMap=np.concatenate([oMins-oImgShapeAdds,oMaxs+oImgShapeAdds])
    WMCoordinatesBigMap=np.concatenate([oMins-oMaxsShapeAdds,oMaxs+oMinsShapeAdds])
    # WMCoordinatesBigMap=np.concatenate([oMins-oMinsShapeAdds,oMaxs+oMaxsShapeAdds])

    #plotting the map
    ghent_img, ghent_ext = ctx.bounds2img(
                                    *WMCoordinatesBigMap,
                                    ll=False,
                                    source=ctx.providers.OpenStreetMap.Mapnik,
                                    zoom=13,
                                    )
    f, ax = plt.subplots(1,figsize=(30,30))

    ax.imshow(ghent_img, extent=ghent_ext)

    ax.axis('off')

    plt.xlim(WMCoordinatesBigMap[0],WMCoordinatesBigMap[2])
    plt.ylim(WMCoordinatesBigMap[1],WMCoordinatesBigMap[3])
    
    mapSaveLoc="./map-test3456.png"
    imgdata = io.BytesIO()

    plt.savefig(imgdata, bbox_inches='tight', pad_inches=0)
    imgdata.seek(0)
    mapImg=mpimg.imread(imgdata)
    mapImg=cv2.normalize(mapImg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    mapImg=cv2.cvtColor(mapImg,cv2.COLOR_RGB2BGR)
    plt.clf()
    plt.cla()
    plt.close()
    # verbose=True
    if(verbose):

        cv2.imshow("map", mapImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    mercPoints=mercPointsOriginalScale #*0.1

    mins=np.amin(mercPoints,axis=0)
    maxs=np.amax(mercPoints,axis=0)
    
    newImgShape=(maxs-mins)
    

    
    imgShapexAdd=int(newImgShape[0]*percentageMore)
    imgShapeyAdd=int(newImgShape[1]*percentageMore)

    imgShapeAdds=newImgShape*percentageMore
    

    # print(mercPoints)
    dst=mercPoints-mins
    # print(dst)
    # dst=dst+imgShapeAdds
    dst=dst+oMaxsShapeAdds
    # dst=dst+oMinsShapeAdds



    src=np.array(detectedPoints,dtype=np.float32)
    
    # imgShape=(np.rint(newImgShape+imgShapeAdds*2)).astype(int)
    imgShape=(np.rint(newImgShape+oMinsShapeAdds+oMaxsShapeAdds)).astype(int)


    m= affine(src,dst)
    
    pImg=cv2.warpAffine(image,m,imgShape,borderValue=(255,255,255))

    pImg = cv2.flip(pImg, 0)
    # print(pImg.shape)
    resizeScale=0.1
    resizedImageShape=(int(pImg.shape[1]*resizeScale),int(pImg.shape[0]*resizeScale))
    resizedpImage =cv2.resize(pImg,resizedImageShape,interpolation = cv2.INTER_AREA)

    if(detectionImage is not None):
        detectionImageWarped=cv2.warpAffine(detectionImage,m,imgShape,borderValue=(255,255,255))
        detectionImage=detectionImageWarped

        detectionImage = cv2.flip(detectionImage, 0)
        detectionImage =cv2.resize(detectionImage,resizedImageShape,interpolation = cv2.INTER_AREA)

    imgShape=resizedImageShape
    pImg=resizedpImage



    
    verbose=False
    if(verbose):
        cv2.imshow("original", image)
        cv2.imshow("Perspective shifted", pImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return pImg,mapImg,imgShape,detectionImage


def image_overlay_second_method(img1, img2, location, min_thresh=0.2, originalImg=None , is_transparent=False,alpha=0.3,verbose=False):
    h, w = img1.shape[:2]
    h1, w1 = img2.shape[:2]
    x, y = location
    roi = img1[y:y + h1, x:x + w1]

    
    thresh_type = cv2.THRESH_BINARY_INV
    
    minImgOriginal=np.min(originalImg,axis=2)
    

    if(verbose==True):
        cv2.imshow("minImg",minImgOriginal)
        

    thresholdVal,_ = cv2.threshold(minImgOriginal,0,255,thresh_type+cv2.THRESH_OTSU)
    minImgTarget=np.min(img1,axis=2)
    _,mask=cv2.threshold(minImgTarget,thresholdVal,255,thresh_type)

    mask_inv = cv2.bitwise_not(mask)
    
    mask=mask[y:y + h1, x:x + w1]
    mask_inv=mask_inv[y:y + h1, x:x + w1]

    img_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

    dst = cv2.add(img_bg, img_fg)
    if is_transparent:
        dst = cv2.addWeighted(dst, alpha, img1[y:y + h1, x:x + w1], 0.9, None)
    img1[y:y + h1, x:x + w1] = dst
    return img1


def multiScaleMatchingT(image,template,visualize):
    minImage=np.min(image,axis=2)
    minImage = cv2.GaussianBlur(minImage,(7,7),0)
    # minImage=cv2.medianBlur(minImage,3)
    # print(minImage.shape)

    edgedImage = minImage
    # edgedImage = cv2.Canny(minImage, 50, 200)
    # edgedImage = cv2.Laplacian(edgedImage,cv2.CV_8U)

    found = None
    # found = np.inf
    # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = np.min(template,axis=2)
    # mask=template.copy().astype(np.uint8)
    # mask[mask>=240]=0
    # template=cv2.GaussianBlur(template,(3,3),0)

    # loop over the scales of the image
    i=0
    results=[]
    for scale in np.linspace(0.2, 1, 40)[::-1]: #! 50 worked
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(template, width = int(template.shape[1] * scale))
        resizedMask=resized.copy()
        resizedMask[resizedMask>=240]=0
        # resized = cv2.GaussianBlur(resized,(3,3),0)
        # resized = cv2.Laplacian(resized,cv2.CV_8U)
        # resizedMask=imutils.resize(mask, width = int(template.shape[1] * scale))
        tH=resized.shape[0]
        tW=resized.shape[1]
        # resized = cv2.Canny(resized, 50, 200)
        if visualize:
            cv2.imshow("tempVisualize", resized)
            # cv2.imshow("tempmask", resizedMask)
            
            # a = imutils.resize(template, width = int(template.shape[1] * scale))
            # cv2.imshow("tempactual", a)

        r = minImage.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if image.shape[0] < tH or image.shape[1] < tW:
            i+=1
            continue

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        result = cv2.matchTemplate(edgedImage, resized, cv2.TM_CCOEFF_NORMED,mask=resizedMask) #! change it to TM_CCORR_NORMED (didnt work) try TM_CCOEFF_NORMED (didnt work) try TM_SQDIFF_NORMED


        result[result==np.inf]=0

        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        # check to see if the iteration should be visualized
        if visualize:
            cv2.imshow("TemplateResult", result)
            cv2.waitKey(0)
            # draw a bounding box around the detected region
            clone = np.dstack([edgedImage, edgedImage, edgedImage])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)
            print("maxval: ",maxVal)
            cv2.waitKey(0)


        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
        # if found is np.inf or maxVal < found[0]: #! for TM_SQDIFF_NORMED
            found = (maxVal, maxLoc,  tH,tW)
            results.append(found)

    results.sort(reverse=True)
    
    print(results[0])
    return results


#HelperFunctions for intersection detection
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections


# @jit(forceobj=True)
def PointsByLinesMinImage(img,howManyLines,cutoffpercentile=10,threshold=None,verbose=False):

    hsvImg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    r=1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adapt_type =cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV

    blur=cv2.medianBlur(img,7)
    minImg=np.min(blur,axis=2)
    brightnessFlattened=minImg.flatten()
    firstPercentile=np.percentile(brightnessFlattened,cutoffpercentile)
    #TODO: fix thresholding for better results
    #! causes the line not to appear when marker is drawn with slightly less ink, try different thresholding to fix it 
    _,bin_img = cv2.threshold(minImg,firstPercentile*3,255,type=thresh_type) 



    # blur=cv2.medianBlur(gray,5)
    _,bin_img = cv2.threshold(minImg,0,255,thresh_type+cv2.THRESH_OTSU)
    if(threshold is not None):
        _,bin_img = cv2.threshold(minImg,threshold,255,thresh_type)

    # bin_img =cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 61, 3)





    centerLinesImg = bin_img.copy()
    # skel = cv2.ximgproc.thinning(centerLinesImg, None, 1)
    size = np.size(centerLinesImg)
    
    # edged = cv2.Canny(bin_img, 50, 100)
    # if(verbose):
    # 	#?SHOWS IMAGE BEFORE LINE DETECTION
    # 	cv2.imshow("preprocessed",edged)
    # 	cv2.waitKey(0)
    # 	cv2.destroyAllWindows()

    
    
    borderColor = (0, 0, 0)
    borderThickness = 1
    bin_img = cv2.copyMakeBorder(bin_img, borderThickness, borderThickness, borderThickness, borderThickness,
                                    cv2.BORDER_CONSTANT, None, borderColor)
    skel = cv2.ximgproc.thinning(bin_img, thinningType = cv2.ximgproc.THINNING_ZHANGSUEN)

    #TODO: test the other skeletoning method
    #!Test this with this thresholding
    # skel = cv2.ximgproc.thinning(bin_img, thinningType = cv2.ximgproc.THINNING_GUOHALL)

    rho, theta, thresh = 1, np.pi/360, 150
    #? parameters for full marker and detecting all points at once
    # lines = cv2.HoughLines(skel, rho, theta, thresh)

    #? parameters for smaller image
    lines = cv2.HoughLines(skel, 1, np.pi/360, 40)
    #TODO check this below condition it might break things
    if(lines is None):
        lines = cv2.HoughLines(skel, 1, np.pi/360, 10)
    

    strong_lines = np.zeros([howManyLines,1,2])

    minLineLength = 2
    maxLineGap = 100

    n2 = 0
    for n1 in range(0,len(lines)):
        for rho,theta in lines[n1]:
            if n1 == 0:
                if rho < 0:
                    rho*=-1
                    theta-=np.pi
                strong_lines[n2] = [rho,theta]
                n2 = n2 + 1
            else:
                if rho < 0:
                    rho*=-1
                    theta-=np.pi
                closeness_rho = np.isclose(rho,strong_lines[0:n2,0,0],atol = 10.0)
                # print(rho,strong_lines[0:n2,0,0],closeness_rho)
                closeness_theta = np.isclose(theta,strong_lines[0:n2,0,1],atol = np.pi/36.0)
                closeness = np.all([closeness_rho,closeness_theta],axis=0)
                if not any(closeness) and n2 < howManyLines:
                    strong_lines[n2] = [rho,theta]#lines[n1]
                    n2 = n2 + 1

    print(strong_lines)

    #TODO: when theta is close to 0 its a verticle line and when its close to pi/2 its horizontal

    lineImg=img.copy()
    allLinesImg=img.copy()
    skelWithLines=skel.copy()
    skelWithLines=cv2.cvtColor(skelWithLines,cv2.COLOR_GRAY2BGR)



    #draws the lines
    for line in strong_lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int((x0 + 3000*(-b))*r)
            y1 = int((y0 + 3000*(a))*r)
            x2 = int((x0 - 3000*(-b))*r)
            y2 = int((y0 - 3000*(a))*r)
            cv2.line(skelWithLines,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.line(lineImg,(x1,y1),(x2,y2),(0,255,0),1)
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int((x0 + 3000*(-b))*r)
            y1 = int((y0 + 3000*(a))*r)
            x2 = int((x0 - 3000*(-b))*r)
            y2 = int((y0 - 3000*(a))*r)
            cv2.line(allLinesImg,(x1,y1),(x2,y2),(0,255,0),1)

    segmentedLines= segment_by_angle_kmeans(strong_lines)
    points= segmented_intersections(segmentedLines)
    for i in range(len(points)):
        points[i][0]=int(points[i][0]*r)
        points[i][1]=int(points[i][1]*r)

    pointsImg=img.copy()
    for x,y in points:
        cv2.circle(pointsImg,(x,y),1,(0,255,0),-1)
        # cv2.circle(skel,(x,y),2,(0,255,0),-1)

    skel[skel==255]=1
    
    

    pointsImg=img.copy()
    for x,y in points:
        cv2.circle(pointsImg,(x,y),1,(0,255,0),-1)
        # cv2.circle(skel,(x,y),2,(0,255,0),-1)

    
    

    


    skel[skel==1]=255


    if(verbose):
        # cv2.imshow("Original", img)
        cv2.imshow("Binary image",bin_img)
        cv2.imshow("skel Image",skel)
        cv2.imshow("skel Image With Selected Lines",skelWithLines)
        # cv2.imshow("Original", img)
        cv2.imshow("All Lines",allLinesImg)
        cv2.imshow("Lines",lineImg)
        cv2.imshow("Points",pointsImg)


        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return points


def GetVerticleHorizontalLines(lines,outputLines,thetaThresh,isHorizontal):
    tempLines=[]
    for line in lines:
        # print(line)
        rho,theta= line[0]
        if rho < 0:
                rho*=-1
                theta-=np.pi
        if(isHorizontal):
            lineDirection=np.pi/2
        else:
            lineDirection=0

        isVerticle=np.isclose(theta,lineDirection,atol=thetaThresh)
        if(isVerticle):
            tempLines.append([rho,theta])
            # if(len(tempLines)==2):
            #     break
    tempLines=np.array(tempLines)

    # avarageLine=np.mean(tempLines,0)
    # outputLines.append([avarageLine])

    # print(avarageLine)

    if(len(tempLines)>=2):
        outputLines.append([(2*tempLines[0]+tempLines[1])/3])
        return
    
    outputLines.append([tempLines[0]])

    # print(lines)

#returns Points in the local context
#?Segmented skel line detection
def PointsByLinesMinImageHashtag(img,howManyLines,cutoffpercentile=10,threshold=None,verbose=False):

    hsvImg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    r=1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adapt_type =cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type = cv2.THRESH_BINARY_INV

    blur=cv2.medianBlur(img,3)
    minImg=np.min(blur,axis=2)
    brightnessFlattened=minImg.flatten()
    firstPercentile=np.percentile(brightnessFlattened,cutoffpercentile)
    #TODO: fix thresholding for better results
    #! causes the line not to appear when marker is drawn with slightly less ink, try different thresholding to fix it 
    _,bin_img = cv2.threshold(minImg,firstPercentile*3,255,type=thresh_type) 



    # blur=cv2.medianBlur(gray,5)
    _,bin_img = cv2.threshold(minImg,0,255,thresh_type+cv2.THRESH_OTSU)

    if(threshold is not None):
        _,bin_img = cv2.threshold(minImg,threshold,255,thresh_type)

    # bin_img =cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 61, 3)





    centerLinesImg = bin_img.copy()
    # skel = cv2.ximgproc.thinning(centerLinesImg, None, 1)
    size = np.size(centerLinesImg)
    
    
    borderColor = (0, 0, 0)
    borderThickness = 1
    bin_img = cv2.copyMakeBorder(bin_img, borderThickness, borderThickness, borderThickness, borderThickness,
                                    cv2.BORDER_CONSTANT, None, borderColor)
    skel = cv2.ximgproc.thinning(bin_img, thinningType = cv2.ximgproc.THINNING_ZHANGSUEN)

    #TODO: test the other skeletoning method
    #!Test this with this thresholding
    # skel = cv2.ximgproc.thinning(bin_img, thinningType = cv2.ximgproc.THINNING_GUOHALL)

    rho, theta, thresh = 1, np.pi/360, 150
    #? parameters for full marker and detecting all points at once
    # lines = cv2.HoughLines(skel, rho, theta, thresh)

    
    hashtagHeight = skel.shape[0]
    hashtagWidth = skel.shape[1]
    hCut = hashtagHeight // 2
    wCut = hashtagWidth // 2




    left=skel.copy()

    right=skel.copy()
    top=skel.copy()
    bottom=skel.copy()

    left[:,:wCut]=0

    right[:,wCut:]=0
    top[:hCut,:]=0
    bottom[hCut:,:]=0
    
    # cv2.imshow("left",left)
    # cv2.imshow("right",right)
    # cv2.imshow("top",top)
    # cv2.imshow("bottom",bottom)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #? parameters for smaller image

    leftLines = cv2.HoughLines(left, 1, np.pi/360, 40)
    rightLines = cv2.HoughLines(right, 1, np.pi/360, 40)
    bottomLines = cv2.HoughLines(top, 1, np.pi/360, 40)
    topLines = cv2.HoughLines(bottom, 1, np.pi/360, 40)
    lines=[]
    # lines.append(leftLines[0])
    # lines.append(rightLines[0])
    # lines.append(bottomLines[0])
    # lines.append(topLines[0])

    # print(lines)
    # print(leftLines[0])
    # print(rightLines[0])
    # print(bottomLines[0])
    # print(topLines[0])

    #TODO:  (2best +1 second best)/3 or (4x best +2second best + 1third)/7
    thetaThresh=np.pi/15
    tempLines=[]


    GetVerticleHorizontalLines(leftLines,lines,thetaThresh,False)
    GetVerticleHorizontalLines(rightLines,lines,thetaThresh,False)
    GetVerticleHorizontalLines(topLines,lines,thetaThresh,True)
    GetVerticleHorizontalLines(bottomLines,lines,thetaThresh,True)
    
    

    # cv2.imshow("Binary image",bin_img)
    # cv2.imshow("skel Image",skel)
    # cv2.imshow("img",img)
    # cv2.imshow("minImg",minImg)


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    strong_lines = np.zeros([howManyLines,1,2])

    minLineLength = 2
    maxLineGap = 100

    n2 = 0
    for n1 in range(0,len(lines)):
        for rho,theta in lines[n1]:
            if n1 == 0:
                if rho < 0:
                    rho*=-1
                    theta-=np.pi
                strong_lines[n2] = [rho,theta]
                n2 = n2 + 1
            else:
                if rho < 0:
                    rho*=-1
                    theta-=np.pi
                closeness_rho = np.isclose(rho,strong_lines[0:n2,0,0],atol = 10.0)
                # print(rho,strong_lines[0:n2,0,0],closeness_rho)
                closeness_theta = np.isclose(theta,strong_lines[0:n2,0,1],atol = np.pi/36.0)
                closeness = np.all([closeness_rho,closeness_theta],axis=0)
                if not any(closeness) and n2 < howManyLines:
                    strong_lines[n2] = [rho,theta]#lines[n1]
                    n2 = n2 + 1

    print(strong_lines)

    #TODO: when theta is close to 0 its a verticle line and when its close to pi/2 its horizontal

    lineImg=img.copy()
    allLinesImg=img.copy()
    skelWithLines=skel.copy()
    skelWithLines=cv2.cvtColor(skelWithLines,cv2.COLOR_GRAY2BGR)



    #draws the lines
    for line in strong_lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int((x0 + 3000*(-b))*r)
            y1 = int((y0 + 3000*(a))*r)
            x2 = int((x0 - 3000*(-b))*r)
            y2 = int((y0 - 3000*(a))*r)
            cv2.line(skelWithLines,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.line(lineImg,(x1,y1),(x2,y2),(0,255,0),1)
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int((x0 + 3000*(-b))*r)
            y1 = int((y0 + 3000*(a))*r)
            x2 = int((x0 - 3000*(-b))*r)
            y2 = int((y0 - 3000*(a))*r)
            cv2.line(allLinesImg,(x1,y1),(x2,y2),(0,255,0),1)

    strong_lines=lines
    segmentedLines= segment_by_angle_kmeans(strong_lines)
    points= segmented_intersections(segmentedLines)
    for i in range(len(points)):
        points[i][0]=int(points[i][0]*r)
        points[i][1]=int(points[i][1]*r)

    pointsImg=img.copy()
    for x,y in points:
        cv2.circle(pointsImg,(x,y),1,(0,255,0),-1)
        # cv2.circle(skel,(x,y),2,(0,255,0),-1)

    skel[skel==255]=1
    bin_temp=bin_img.copy()
    bin_temp[bin_temp==255]=1
    

    convFilter=np.array([[1, 1, 1],
                        [1, 10, 1],
                        [1, 1, 1],
                        ])
    degreeImage=cv2.filter2D(skel, -1, convFilter)
    degreeImage[degreeImage == 14]=255

    rangeFromCandiditePoint=5

    newRangeStartY=points[0][1]-rangeFromCandiditePoint
    newRangeEndY=points[0][1]+rangeFromCandiditePoint

    newRangeStartX=points[0][0]-rangeFromCandiditePoint
    newRangeEndX=points[0][0]+rangeFromCandiditePoint

    newRangedArray=degreeImage[newRangeStartY:newRangeEndY,newRangeStartX:newRangeEndX]

    intersectionPoints=np.where(newRangedArray==255)
    xlen=len(intersectionPoints[1])
    ylen=len(intersectionPoints[0])
    if(xlen==0):
        degreeImage[degreeImage == 13]=255
        intersectionPoints=np.where(newRangedArray==255)
        xlen=len(intersectionPoints[1])
        ylen=len(intersectionPoints[0])


    xsum=intersectionPoints[1].sum()
    ysum=intersectionPoints[0].sum()
    averageX=xsum//xlen
    averageY=ysum//ylen
    newPoint=[[averageX+newRangeStartX,averageY+newRangeStartY]]
    # points=newPoint

    pointsImg=img.copy()
    for x,y in points:
        cv2.circle(pointsImg,(x,y),1,(0,255,0),-1)
        # cv2.circle(skel,(x,y),2,(0,255,0),-1)

    
    skel[skel==1]=255

    if(verbose):
        # cv2.imshow("Original", img)
        cv2.imshow("degreeImage",degreeImage)
        cv2.imshow("Binary image",bin_img)
        cv2.imshow("skel Image",skel)
        cv2.imshow("skel Image With Selected Lines",skelWithLines)
        # cv2.imshow("Original", img)
        cv2.imshow("All Lines",allLinesImg)
        cv2.imshow("Lines",lineImg)
        cv2.imshow("Points",pointsImg)


        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return points


#returns Points in order of top left, top right, bottom left, bottom right
def ReturnPoints(pointsOnBigImage):
    """
    returns Points in order of top left, top right, bottom left, bottom right
    """

    # get top and bottom points

    sortedPoints=pointsOnBigImage[pointsOnBigImage[:, 1].argsort()]
    topPoints=sortedPoints[:2]
    bottomPoints=sortedPoints[2:]


    # get left and right top points
    sortedTopPoints=topPoints[topPoints[:, 0].argsort()]
    leftTopPoint=sortedTopPoints[0]
    rightTopPoint=sortedTopPoints[1]


    # get left and right bottom points
    sortedBottomPoints=bottomPoints[bottomPoints[:, 0].argsort()]
    leftBottomPoint=sortedBottomPoints[0]
    rightBottomPoint=sortedBottomPoints[1]

    return leftTopPoint,rightTopPoint,leftBottomPoint,rightBottomPoint


def getMGRSPoints(templateIndex,image,markerbbox,leftTop=0,rightTop=0,leftBot=0,RightBot=0):
    startX,startY,endX,endY =markerbbox

    edgeCase=False
    pixelPecentageZoomOut=1
    startX=int(startX-(endX-startX)*pixelPecentageZoomOut)
    startY=int(startY-(endY-startY)*pixelPecentageZoomOut)
    endX=int(endX+(endX-startX)*pixelPecentageZoomOut)
    endY=int(endY+(endX-startX)*pixelPecentageZoomOut)
    if(startX<0):
        startX=0
    if(startY<0):
        startY=0
    if(endX>image.shape[1]):
        endX=image.shape[1]-1
    if(endY>image.shape[0]):
        endY=image.shape[0]-1


    mgrsPoints=[]
    if(templateIndex==0):
        cv2.imshow("Give Input on Terminal", image[startY:endY,startX:endX])
        print("Enter grid zone designator and identifier: ")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        gridName=input("Press enter to continue").upper()

        # topLeft=input("Enter top-left point: ").upper()
        # topRight=input("Enter top-right point: ").upper()
        # bottomLeft=input("Enter bottom-left point: ").upper()
        # bottomRight=input("Enter bottom-right point: ").upper()


        tempImage=image.copy()
        #request top line coordinate
        cv2.line(tempImage, (leftTop), (rightTop), (0, 255, 0), thickness=3, lineType=8)


        print("Enter top-line coordinate: ")
        if(edgeCase):
            cv2.imshow("Give Input on Terminal", tempImage)
        else:
            tempImage=tempImage[startY:endY,startX:endX]
            cv2.imshow("Give Input on Terminal", tempImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        topLine=input("Press enter to continue").upper()


        tempImage=image.copy()
        #request bottom line coordinate
        cv2.line(tempImage, (leftBot), (RightBot), (0, 255, 0), thickness=3, lineType=8)

        print("Enter bottom-line coordinate: ")

        if(edgeCase):
            cv2.imshow("Give Input on Terminal", tempImage)
        else:
            tempImage=tempImage[startY:endY,startX:endX]
            cv2.imshow("Give Input on Terminal", tempImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        bottomLine=input("Press enter to continue").upper()


        tempImage=image.copy()
        #request left line coordinate
        cv2.line(tempImage, (leftTop), (leftBot), (0, 255, 0), thickness=3, lineType=8)
        print("Enter left-line coordinate: ")
        if(edgeCase):
            cv2.imshow("Give Input on Terminal", tempImage)
        else:
            tempImage=tempImage[startY:endY,startX:endX]
            cv2.imshow("Give Input on Terminal", tempImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        leftLine=input("Press enter to continue").upper()

        tempImage=image.copy()
        #request right line coordinate
        cv2.line(tempImage, (rightTop), (RightBot), (0, 255, 0), thickness=3, lineType=8)


        print("Enter right-line coordinate: ")
        if(edgeCase):
            cv2.imshow("Give Input on Terminal", tempImage)
        else:
            tempImage=tempImage[startY:endY,startX:endX]
            cv2.imshow("Give Input on Terminal", tempImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rightLine=input("Press enter to continue").upper()

        mgrsPoints.append(gridName+leftLine+topLine)
        mgrsPoints.append(gridName+rightLine+topLine)
        mgrsPoints.append(gridName+leftLine+bottomLine)
        mgrsPoints.append(gridName+rightLine+bottomLine)

        # mgrsPoints.append(gridName+topLeft)
        # mgrsPoints.append(gridName+topRight)
        # mgrsPoints.append(gridName+bottomLeft)
        # mgrsPoints.append(gridName+bottomRight)

    if(templateIndex==1):

        # edgeCase=False
        # pixelPecentageZoomOut=0.15
        # startX=int(startX-(endX-startX)*pixelPecentageZoomOut)
        # startY=int(startY-(endY-startY)*pixelPecentageZoomOut)
        # endX=int(endX+(endX-startX)*pixelPecentageZoomOut)
        # endY=int(endY+(endX-startX)*pixelPecentageZoomOut)
        # if(startX<0):
        #     startX=0
        # if(startY<0):
        #     startY=0
        # if(endX>image.shape[1]):
        #     endX=image.shape[1]-1
        # if(endY>image.shape[0]):
        #     endY=image.shape[0]-1

        cv2.imshow("Give Input on Terminal", image[startY:endY,startX:endX])
        print("Enter grid zone designator and identifier: ")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        gridName=input("Press enter to continue").upper()

        tempImage=image.copy()
        #request top line coordinate
        cv2.line(tempImage, (leftTop), (leftTop[0]+endX-startX,leftTop[1]), (0, 255, 0), thickness=3, lineType=8)
        cv2.line(tempImage, (leftTop), (leftTop[0]-endX-startX,leftTop[1]), (0, 255, 0), thickness=3, lineType=8)



        print("Enter horizontal-line coordinate: ")
        if(edgeCase):
            cv2.imshow("Give Input on Terminal", tempImage)
        else:
            tempImage=tempImage[startY:endY,startX:endX]
            cv2.imshow("Give Input on Terminal", tempImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        horizontalLine=input("Press enter to continue").upper()


        tempImage=image.copy()
        #request bottom line coordinate
        cv2.line(tempImage, (leftTop), (leftTop[0],leftTop[1]+endY-startY), (0, 255, 0), thickness=3, lineType=8)
        cv2.line(tempImage, (leftTop), (leftTop[0],leftTop[1]-endY-startY), (0, 255, 0), thickness=3, lineType=8)

        print("Enter verticle-line coordinate: ")

        if(edgeCase):
            cv2.imshow("Give Input on Terminal", tempImage)
        else:
            tempImage=tempImage[startY:endY,startX:endX]
            cv2.imshow("Give Input on Terminal", tempImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        verticleLine=input("Press enter to continue").upper()



        mgrsPoints.append(gridName+verticleLine+horizontalLine)
    return mgrsPoints
        

def SearchROI(image,template,templateIndex,visualize=False):
    roi=image.copy()
    boundingbox_widget = BoundingBoxWidget(image)

    while(True):
        boundingbox_widget.show_image()
        key = cv2.waitKey(0)

        # end ROI selection with keyboard 'enter/return'
        if key == 13:
            boundingBoxROIPoints=boundingbox_widget.GetBoundingBoxPoints()
            if(len(boundingBoxROIPoints) != 2):
                print("Please select a region of interest.")
                continue
            cv2.destroyAllWindows()
            break
        # end program with keyboard 'esc'
        if key == 27:
            print("Exiting program.")
            exit()

    boundingBoxROIPoints=boundingbox_widget.GetBoundingBoxPoints()
    boundingBoxROIPointsX=[ x for x,y in boundingBoxROIPoints]
    boundingBoxROIPointsY=[ y for x,y in boundingBoxROIPoints]
    roiXmax=max(boundingBoxROIPointsX)
    roiXmin=min(boundingBoxROIPointsX)
    roiYmax=max(boundingBoxROIPointsY)
    roiYmin=min(boundingBoxROIPointsY)
    roi = roi[roiYmin:roiYmax, roiXmin:roiXmax]
    # visualize=True
    # visualize=False
    found=multiScaleMatchingT(roi,template,visualize)
    detectionWindowName="Detection (if Correct press enter if wrong esc)"
    for maxScore,(X,Y), tH,tW in found:
        x,y=(roiXmin+X,roiYmin+Y)
        (startX, startY) = (x, y)
        (endX, endY) = (int((x + tW)), int((y + tH)))
        tempImg=image.copy()
        print(f"Detection Score: {maxScore}")
        cv2.rectangle(tempImg, (startX, startY), (endX, endY), (0, 0, 255), 10)
        cv2.namedWindow(detectionWindowName,cv2.WINDOW_KEEPRATIO)
        # cv2.setWindowProperty(detectionWindowName, cv2.WND_PROP_FULLSCREEN  , cv2.WINDOW_FULLSCREEN)
        cv2.imshow(detectionWindowName, tempImg)

        key = cv2.waitKey(0)

        while(key!=13 and key!=27):
            key = cv2.waitKey(0)
            continue
        cv2.destroyAllWindows()
        
        if key == 13:
            found=maxScore,(X,Y), tH,tW
            cv2.destroyAllWindows()
            break
        # end program with keyboard 'esc'
        elif key == 27:
            print("Next Detection.")
            continue
    else: 
        print("No Match is selected, exiting.")
        exit()
    


    # else:
    #     found=multiScaleMatchingTParallel(roi,template)

    #returning to original image pixels from corner pixels
    matchScore, (detectedX,detectedY),tH,tW=found
    found=matchScore,(roiXmin+detectedX,roiYmin+detectedY),tH,tW

    return found

def main():
    verbose=False
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--template", required=True, help="Path to template image")
    ap.add_argument("-i", "--images", required=True,
        help="Path to images where template will be matched")
    ap.add_argument("-s", "--singleImage", required=False,default=0,type=int,
        help="input 1 if the image path was for a single image instead of a folder with images")
    ap.add_argument("-d", "--detection", required=False,default=None,
        help="input 1 detected symbols to be also placed (only works with single image mode)")
    ap.add_argument("-t2", "--template2", required=True,
        help="Path to images where template will be matched")
    ap.add_argument("-v", "--visualize",
        help="Flag indicating whether or not to visualize each iteration")
    args = vars(ap.parse_args())
    visualize=False
    if args.get("visualize", False):
        visualize=True
    else:
        visualize=False

    # load the image image, convert it to grayscale, and detect edges
    template = cv2.imread(args["template"])

    #initial shape for template 
    (tH, tW) = template.shape[:2]
    if(verbose):
        cv2.imshow("Template", template)


    template2 = cv2.imread(args["template2"])

    #initial shape for template 
    (t2H, t2W) = template2.shape[:2]

    i=0

    # loop over the images to find the template in
    if(not args["singleImage"]):
        files=glob.glob(args["images"] + "/*.jpg")
        files.extend(glob.glob(args["images"] + "/*.jpeg"))
        files.extend(glob.glob(args["images"] + "/*.png"))
    else:
        files=[args["images"]]


    for imagePath in files:

        overallStartTime=time.perf_counter()
        verbose=False
        i+=1
        # load the image, convert it to grayscale, and initialize the
        # bookkeeping variable to keep track of the matched region
        image = cv2.imread(imagePath)

        #! but this to a if statement not wot divide small imiges by 2
        # imgHalfShape=image.shape[1]//2,image.shape[0]//2
        # image =cv2.resize(image,imgHalfShape,interpolation = cv2.INTER_AREA)


        visualize=args.get("visualize", False)

        #?using global threshold for line detection, because some faint lines were dissapearing when the line detection size is small
        minImg=np.min(image,axis=2)

        minImg=cv2.medianBlur(minImg,7)
        thresh_type = cv2.THRESH_BINARY_INV
        lineDetectionThreshold,_ = cv2.threshold(minImg,0,255,thresh_type+cv2.THRESH_OTSU)

        # cv2.imshow("minImg", minImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        nextDetectionImage=image.copy()
        allDetectionsImage=image.copy()

        lineDetectionPercentile=2


        mgrsAllPoints=[]
        allDetectedPoints=np.empty((0,2))

        markerInput=10
        while(markerInput!=-1):
            markerInput=input("Select which marker to search (1 for hashtag, 2 for cross, -1 for not searching): ")

            try:
                markerInput=int(markerInput)
            except ValueError:
                print("Wrong input try again!")

            if(markerInput==1):
                # startTime=time.perf_counter()
                found=SearchROI(image,template,0,visualize)
                # endTime=time.perf_counter()
                # print("First PatternTime: ",endTime-startTime)

                # startTime=time.perf_counter()
                # mgrsAllPoints=mgrsPoints.copy()


                    
                
                
                # # unpack the bookkeeping variable and compute the (x, y) coordinates
                # # of the bounding box based on the resized ratio
                (_, maxLoc, tH,tW) = found
                (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
                (endX, endY) = (int((maxLoc[0] + tW)), int((maxLoc[1] + tH)))

                # draw a bounding box around the detected result and display the image
                hashtag=image[startY:endY,startX:endX].copy()

                minHash=np.min(hashtag,axis=2)

                # minHash=cv2.medianBlur(minHash,7)
                thresh_type = cv2.THRESH_BINARY_INV
                lineDetectionThreshold,testImga = cv2.threshold(minHash,0,255,thresh_type+cv2.THRESH_OTSU)

                # #? for faint lines works better but need to investigate otsu
                # hsvHashtag=cv2.cvtColor(hashtag,cv2.COLOR_BGR2HSV)
                # saturationImage=hsvHashtag[:,:,1]
                # lineDetectionThreshold,testImga = cv2.threshold(saturationImage,0,255,thresh_type+cv2.THRESH_OTSU)


                #? for making the threshold more lenient for faint lines 
                # lineDetectionThreshold=int(lineDetectionThreshold*1.2)
                # print(lineDetectionThreshold)
                # lineDetectionThreshold,testImga = cv2.threshold(minHash,lineDetectionThreshold,255,thresh_type)



                # cv2.imshow("testImga", testImga)
                # cv2.imshow("saturationImage", saturationImage)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()



                #? draws a suqare on the detected area
                cv2.rectangle(allDetectionsImage, (startX, startY), (endX, endY), (0, 0, 255), 10)

                # cv2.imshow("First", allDetectionsImage)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                #? masking the detected hashtag for line detection

                hashtagMask=template.copy()
                hashtagMask=imutils.resize(hashtagMask, width = tW ,height = tH)
                hashtagMask=np.min(hashtagMask,axis=2)

                hashtagMask[hashtagMask>=240]=255
                hashtagMask[hashtagMask<240]=0
                hashtagMaskInverted=cv2.bitwise_not(hashtagMask)


                maskIndexes=np.where(hashtagMaskInverted==0)
                hashtag[maskIndexes]=255


                hashtagHeight = hashtag.shape[0]
                hashtagWidth = hashtag.shape[1]
                hCut = hashtagHeight // 2
                wCut = hashtagWidth // 2


                l1=hashtag[:hCut,:wCut]
                r1=hashtag[:hCut,wCut:]

                l2=hashtag[hCut:,:wCut]
                r2=hashtag[hCut:,wCut:]



                
                # lineDetectionThreshold=None


                verbose=True
                #TODO handle expection and try different aproach
                #! if something fails(exceptions) here try the without giving the threshold value
                # l1Point=PointsByLinesMinImage(l1,2,lineDetectionPercentile,threshold=lineDetectionThreshold,verbose=verbose)
                # r1Point=PointsByLinesMinImage(r1,2,lineDetectionPercentile,threshold=lineDetectionThreshold,verbose=verbose)
                # l2Point=PointsByLinesMinImage(l2,2,lineDetectionPercentile,threshold=lineDetectionThreshold,verbose=verbose)
                # r2Point=PointsByLinesMinImage(r2,2,lineDetectionPercentile,threshold=lineDetectionThreshold,verbose=verbose)

                #? function that detects lines separately
                localPoints=PointsByLinesMinImageHashtag(hashtag,4,lineDetectionPercentile,threshold=lineDetectionThreshold,verbose=verbose)
                # localPoints=PointsByLinesMinImage(hashtag,4,lineDetectionPercentile,threshold=lineDetectionThreshold,verbose=verbose)
                

                #TODO: use this local points returned above it should use now but there might be problems CHECK 
                # l1Point=PointsByLinesMinImage(l1,2,percentile,verbose=verbose)
                # r1Point=PointsByLinesMinImage(r1,2,percentile,verbose=verbose)
                # l2Point=PointsByLinesMinImage(l2,2,percentile,verbose=verbose)
                # r2Point=PointsByLinesMinImage(r2,2,percentile,verbose=verbose)


                # l1Point =l1Point[0]

                # r1Point=r1Point[0]
                # r1Point[0]+=wCut

                # l2Point =l2Point[0]
                # l2Point[1]+=hCut

                # r2Point =r2Point[0]
                # r2Point[1]+=hCut
                # r2Point[0]+=wCut

                # localPoints=[l1Point,r1Point,l2Point,r2Point]


                pointsOnBigImage=[]

                for x,y in localPoints:
                    cv2.circle(allDetectionsImage,(x+startX,y+startY),2,(0,255,0),-1)
                    pointsOnBigImage.append([x+startX,y+startY])
                pointsOnBigImage=np.array(pointsOnBigImage,int)
                
                
                leftTop,rightTop,leftBot,RightBot=ReturnPoints(pointsOnBigImage)
                mgrsPoints=[]
                mgrsPoints=getMGRSPoints(0,image,(startX,startY,endX,endY),leftTop,rightTop,leftBot,RightBot)
                mgrsAllPoints+=mgrsPoints
                # allDetectedPoints+=[leftTop,rightTop,leftBot,RightBot]
                allDetectedPoints=np.concatenate ((allDetectedPoints,[leftTop,rightTop,leftBot,RightBot]),0)


            #TODO test this to see if it works without deleting the previously detected marker
            #? deletes the previous detection from the image for next marker detection
            # cv2.rectangle(nextDetectionImage, (startX, startY), (endX, endY), (255, 255, 255), -1)




            # startTime=time.perf_counter()
            elif(markerInput==2):
                found=SearchROI(image,template2,1,visualize)
                # endTime=time.perf_counter()
                # print("Second PatternTime: ",endTime-startTime)





                # # unpack the bookkeeping variable and compute the (x, y) coordinates
                # # of the bounding box based on the resized ratio
                (_, maxLoc, tH,tW) = found
                (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
                (endX, endY) = (int((maxLoc[0] + tW)), int((maxLoc[1] + tH)))
                # draw a bounding box around the detected result and display the image
                marker=nextDetectionImage[startY:endY,startX:endX].copy()

                #? masking the detected marker for line detection
                markerMask=template2.copy()
                markerMask=imutils.resize(markerMask, width = tW ,height = tH)
                markerMask=np.min(markerMask,axis=2)

                markerMask[markerMask>=240]=255
                markerMask[markerMask<240]=0
                markerMaskInverted=cv2.bitwise_not(markerMask)


                maskIndexes=np.where(markerMaskInverted==0)
                marker[maskIndexes]=255


                #? draws a suqare on the detected area
                cv2.rectangle(allDetectionsImage, (startX, startY), (endX, endY), (0, 0, 255), 10)
                
                #? corner points then center point detection using edge lines
                # verbose=True
                # localPoints=PointsByLinesMinImage(marker,2,percentile,verbose=verbose)
                verbose=True
                localPoints=PointsByLinesMinImage(marker,2,lineDetectionPercentile,threshold=lineDetectionThreshold,verbose=verbose)



                pointsOnBigImage=[]

                for x,y in localPoints:
                    cv2.circle(allDetectionsImage,(x+startX,y+startY),2,(0,255,0),-1)
                    pointsOnBigImage.append([x+startX,y+startY])
                pointsOnBigImage=np.array(pointsOnBigImage,int)
                
                mgrsPoints=[]
                mgrsPoints=getMGRSPoints(1,image,(startX,startY,endX,endY),pointsOnBigImage[0])
                mgrsAllPoints+=mgrsPoints

                allDetectedPoints=np.concatenate ((allDetectedPoints,pointsOnBigImage),0)
                # allDetectedPoints+=pointsOnBigImage
        print(mgrsAllPoints)
        
        if(not np.any(allDetectedPoints)):
            print("No Searches made, exiting.")
            exit()
        elif(allDetectedPoints.shape[0]<3):
            print("Not Enough points found, exiting.")
            exit()



        # allDetectedPoints=np.array(allDetectedPoints)
        #? maybe show detections for every detection and ask if its correct
        cv2.namedWindow("All Detections (with Points)",cv2.WINDOW_KEEPRATIO)
        cv2.imshow("All Detections (with Points)", allDetectionsImage)

        cv2.waitKey(0)

        #?for closing the image window and still continueing the code
        # while(True):
        #     key=cv2.waitKey(500)
        #     if cv2.getWindowProperty("All Detections (with Points)", cv2.WND_PROP_VISIBLE) < 1 or key!=-1:
        #         print("ALL WINDOWS ARE CLOSED")
        #         break
        cv2.destroyAllWindows()
        # continue

        
        imageName=os.path.basename(imagePath)

        # a1="35VMF3088"
        # a2="35VMF3188"
        # a3="35VMF3087"
        # a4="35VMF3187"

        # a5="35VMF3878"

        # stringMGRSPoints=[a1,a2,a3,a4,a5]

        # detectedPoints=[[leftTop[0],leftTop[1]],[rightTop[0],rightTop[1]],[leftBot[0],leftBot[1]],[RightBot[0],RightBot[1]],[pointsOnBigImage[0,0],pointsOnBigImage[0,1]]]

        stringMGRSPoints=mgrsAllPoints
        # stringMGRSPoints=['35VMF3088', '35VMF3188', '35VMF3087', '35VMF3187', '35VMF3878']
        detectedPoints=allDetectedPoints
        if(args["singleImage"] and (args["detection"] is not None)):
            detectionImage = cv2.imread(args["detection"])
            pImg,mapImg,imgShape,detectionImage=MapPreparing(image,stringMGRSPoints,detectedPoints,detectionImage=detectionImage)
        else:
            pImg,mapImg,imgShape,_=MapPreparing(image,stringMGRSPoints,detectedPoints)
            detectionImage=None


        verbose=False
        
        if(verbose):
            print(allDetectedPoints)
            cv2.imshow("All Detections on Image", allDetectionsImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        
        overlay = mapImg




        overlay =cv2.resize(overlay,imgShape,interpolation = cv2.INTER_AREA)


        if(detectionImage is not None):
            originalDetectionImage = cv2.imread(args["detection"])
            detectionImage=image_overlay_second_method(detectionImage, overlay, (0, 0),originalImg=originalDetectionImage,is_transparent=False,verbose=verbose)

            
        newImg=image_overlay_second_method(pImg, overlay, (0, 0),originalImg=image,is_transparent=False,verbose=verbose)

        # imagesSavePath="./allImagesWithMap/"
        imagesSavePath=""

        imageSaveFolder="ScannedFilms/"
        imageSaveFolder="AllFilms/"
        imageSaveFolder="exampleToPlaceOnMap/"
        imageSaveFolder="PlacedOnMap/"
        if(not os.path.exists(imageSaveFolder)):
            os.mkdir(imageSaveFolder)
        # verbose=True
        if(verbose):
            cv2.imshow("transperent", newImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite(f"{imagesSavePath}{imageSaveFolder}{imageName}",newImg)
            if(detectionImage is not None):
                cv2.imwrite(f"{imagesSavePath}{imageSaveFolder}Detection{imageName}",detectionImage)

            print(f"{imagesSavePath}{imageSaveFolder}{imageName}")
            pass

        print("Image: ",i)
        overallEndTime=time.perf_counter()
        print("Overall Time: ",overallEndTime-overallStartTime)


# arguments example -t .\testTemplate2.jpg -t2 ./crossTemplate2.jpg -i .\cameraFilms\
# arguments example -t .\testTemplate5.jpg -t2 ./crossTemplate6.jpg -i .\ScannedFilms\
#TODO: maybe find a different solution for this.
#? the testTepmlate6 is more restrictive for cases the hashtag marker is very close to edge other wise testTemplate5 works perfectly
# arguments example -t .\testTemplate6.jpg -t2 ./crossTemplate7.jpg -i .\testImage 
#! parameters for demo
# D:/Miniconda3.7/envs/symbols3/python.exe .\MarkerDetectionWithROI.py -t .\testTemplate6.jpg -t2 ./crossTemplate7.jpg -i .\exampleToPlaceOnMap

#  D:/Miniconda3.7/envs/symbols3/python.exe .\MarkerDetectionWithROI.py -t .\testTemplate6.jpg -t2 ./crossTemplate7.jpg -i "D:\Workplace\Symbols\YOLO-Detection\yolov5\DEMOTEST"
    
if __name__=="__main__":
    main()



    
    





