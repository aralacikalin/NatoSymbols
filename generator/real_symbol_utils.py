from generator_utils import *

# Adds unit symbol in the middle of screen, cover and guard.
def add_real_symbol(imgClean,imageDirty, scale ):
    #Cut excess white from edges
    # imgClean = cut_excess_white(imgClean)
    # imageDirty = cut_excess_white(imageDirty)


    # Find if there is cap between arrow and letter S/C/G or not

    #Get unit_symbol  


    # unit_symbol = resize_by_scale(unit_symbol, scale)

    # if imgClean.shape[0] < ceil(unit_symbol.shape[0]/2):
    #     shape1 = unit_symbol.shape[0]
    # else:
    #     shape1 = ceil(unit_symbol.shape[0]/2) + imgClean.shape[0]

    #img3 = img3.astype('float32')
    
    # imgClean = np.pad(imgClean, ((600,600),(600,600)), "constant", constant_values=255)
    # imageDirty = np.pad(imageDirty, ((600,600),(600,600)), "constant", constant_values=255)
    
    rotation = randint(0,359)
    imageDirtyRotated = ndimage.rotate(imageDirty, rotation, reshape=False, mode='constant',cval=255)
    imgCleanRotated = ndimage.rotate(imgClean, rotation, reshape=False, mode='constant',cval=255)
    
    top1 = np.argwhere(np.amin(imageDirtyRotated,axis=1) < 110)[0][0]
    bottom1 = np.argwhere(np.amin(imageDirtyRotated,axis=1) < 110)[-1][0]
    left1 = np.argwhere(np.amin(imageDirtyRotated,axis=0) < 110)[0][0]
    right1 = np.argwhere(np.amin(imageDirtyRotated,axis=0) < 110)[-1][0]
    
    top2 = np.argwhere(np.amin(imgCleanRotated,axis=1) < 110)[0][0]
    bottom2 = np.argwhere(np.amin(imgCleanRotated,axis=1) < 110)[-1][0]
    left2 = np.argwhere(np.amin(imgCleanRotated,axis=0) < 110)[0][0]
    right2 = np.argwhere(np.amin(imgCleanRotated,axis=0) < 110)[-1][0]
    
    point1 = (top1-top2,left1-left2)
    point2 = (bottom1-bottom2,right1-right2)
    
    imageDirtyRotated = cut_excess_white(imageDirtyRotated)
    #img3_rotated = cut_excess_white(img3_rotated)

    img2_float = imageDirtyRotated.astype('float32') #OpenCV requires float32 type, cant work with int16
    cleanrotated = imgCleanRotated.astype('float32') #OpenCV requires float32 type, cant work with int16
    
    return img2_float, rotation, point1, point2



if __name__=="__main__":
    dirty=cv2.imread(r"D:\Workplace\Symbols\SymbolExtractor\ExtractedSymbols-RotationResetted\GoodFilmsCropped\attack_by_fire1.png")
    clean=cv2.imread(r"D:\Workplace\Symbols\SymbolExtractor\ExtractedSymbolsCleaned\GoodFilmsCropped\attack_by_fire1.png")

    	
    dirty = cv2.cvtColor(dirty, cv2.COLOR_BGR2GRAY)
    clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    dirty[dirty <= 100] = 0
    dirty[dirty > 100] = 255
    clean[clean <= 100] = 0
    clean[clean > 100] = 255
    cv2.imshow("dirty",dirty)
    cv2.imshow("clean",clean)

    img,rot,p1,p2=add_real_symbol(clean,dirty,1)
    img=img.astype(np.uint8)
    img[img <= 100] = 0
    img[img > 100] = 255

    cv2.imshow("final image",img)
    print(p1,p2)
    cv2.waitKey(0)