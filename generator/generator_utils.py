import numpy as np
import cv2
import re
import os
from math import sin, cos, tan, radians, ceil
import random
from random import randint
from scipy import ndimage
from generate_unit_symbol import *
import copy

# Given top-left coorindate postion, places symbol on the canvas
# Only places black pixels. Doesn't override already existing black pixels to white.
def place_symbol(canvas, symbol, point1, point2):
    canvas[point1:point1+symbol.shape[0],point2:point2+symbol.shape[1]][symbol < 140] = 0
    return canvas

# Finds the empty place on canvas where to put the symbol.
# Takes into account the allowed overlap
def get_points(dim, symbol, locations, locations_units,location_placement):
    is_overlap = True
    counter = 0
    while(is_overlap):
        overlap = False
        #Get the random location on canvas
        point1 = randint(0,dim[0]-symbol.shape[0]-10)
        point2 = randint(0,dim[1]-symbol.shape[1]-10)
        overlap = (check_overlap(point1,point2,locations,symbol.shape) or
                  check_overlap(point1,point2,locations_units,symbol.shape) or
                  check_overlap(point1,point2,location_placement,symbol.shape))

        if not overlap:
            is_overlap = False
        
        counter+=1
        if counter > 50:
            raise Exception("No place found")
    
    return point1, point2

def get_grid_numbers(grid, scale, sample_extras):
    first_number = get_random(str(grid//10), sample_extras)
    first_number = resize_by_scale(first_number, scale)
    second_number = get_random(str(grid%10), sample_extras)
    second_number = resize_by_scale(second_number, scale)
    return first_number, second_number

# Adds the frid number for palcement symbol
def add_grid_number(canvas,point1,point2,offset0,offset1,grid,placement,sample_extras,scale=0.5):
    first_number, second_number = get_grid_numbers(grid, scale, sample_extras)
    point1_1 = offset0+point1-int(first_number.shape[0]/2)
    point2_1 = canvas.shape[1]-offset1-placement.shape[1]-10
    canvas = place_symbol(canvas, first_number, point1_1, point2_1-second_number.shape[1]-first_number.shape[1]-int(15*scale))
    canvas = place_symbol(canvas, second_number, point1_1, point2_1-second_number.shape[1]-5)
    return canvas

def add_grid_number2(canvas,point1,point2,offset0,offset1,grid,placement,sample_extras,scale=0.5):
    first_number, second_number = get_grid_numbers(grid, scale, sample_extras)
    point1_1 = offset0+placement.shape[0]+10
    point2_1 = canvas.shape[1]-offset1-point2
    canvas = place_symbol(canvas, first_number, point1_1, point2_1-first_number.shape[1]-10)
    canvas = place_symbol(canvas, second_number, point1_1, point2_1+10)
    return canvas

# Draws the random (non-straight) line given the center of it.
def draw_line(canvas, point1, point2, rotation, side, img):
    center = [int(point1+img.shape[0]/2), int(point2+img.shape[1]/2)]
    max_lengths = np.zeros((2,2),dtype="i")
    max_lengths[0,1] = -canvas.shape[0]+center[0] #Up
    max_lengths[0,0] = -canvas.shape[1]+center[1] #Right
    max_lengths[1,1] = center[0] #Down
    max_lengths[1,0] = center[1] #Left
    prev_point = center
    rotation = rotation + side*90
    for i in range(0,randint(3,6)):
        line_length = randint(70,150)
        point2_1, point1_1 = point_location(rotation,(line_length,line_length),prev_point,(1,1))
        new_point = np.array((point1_1,point2_1))

        max_lengths[0,:] = max_lengths[0,:] + (prev_point-new_point)
        max_lengths[1,:] = max_lengths[1,:] + (prev_point-new_point)
        if (max_lengths[0,0] > 0 and max_lengths[0,1] > 0 and max_lengths[1,0] < 0 and max_lengths[1,1] < 0):
            break
        cv2.line(canvas, (prev_point[1],prev_point[0]), (new_point[1],new_point[0]), 0, 9)
        prev_point = new_point
        rotation = rotation+randint(-20,20)
    return canvas

# Given the rotation finds the upper left point on canvas where to place the unit symbol for given tactical task.
def point_location(rotation,symbol_shape,center,unit_symbol_shape):
    x = center[1]
    y = center[0]
    
    if rotation < 180:
        y -= int((rotation/180)*unit_symbol_shape[0])
        x -= int(unit_symbol_shape[1]/2 * abs(cos(radians(rotation))))
    else:
        y -= int(((360-rotation)/180)*symbol_shape[0])
        x -= int(unit_symbol_shape[1]/2 * abs(cos(radians(rotation))) + unit_symbol_shape[1]/2 * abs(sin(radians(rotation))))
    
    #Point on ellipse line
    theta = tan(radians(rotation))**2
    if theta == 0:
        theta = 1e-10
    y2 = int(symbol_shape[0]*symbol_shape[1] / (np.sqrt(symbol_shape[0]**2 + symbol_shape[1]**2 * theta)))
    x2 = int(symbol_shape[0]*symbol_shape[1] / (np.sqrt(symbol_shape[1]**2 + symbol_shape[0]**2 / theta)))
    if rotation >= 90 and rotation < 270:
        y -= y2
    else:
        y += y2
    if (rotation >= 0 and rotation < 180):
        x += x2
    else:
        x -= x2
    return x,y

# Checks if there is overlap. If max_overlap = 50 and the overlap between two symbols is 30 pixels in
# vertical or/and horizontal axis then return false.
def check_overlap(point1,point2,locations, symbol_shape = (1,1),max_overlap=50):
    is_overlap = False
    for location in locations:
        if ((point1 < location[1][0]-max_overlap) and
            (point2 < location[1][1]-max_overlap) and
            (point1+symbol_shape[0] > location[0][0]+max_overlap) and
            (point2+symbol_shape[1] > location[0][1]+max_overlap)):
            is_overlap = True
            break
    return is_overlap

# Read the symbol images into dictionary
def read_into_dic(directory, re_in, output_dir = None):
    if output_dir == None:
        output_dir = {}
    for filename in os.listdir(directory+"/"):
        img = cv2.imread(directory+"/" + filename,0)
        img[img <= 100] = 0
        img[img > 100] = 255
        img = cut_excess_white(img)
        key = re.findall(re_in, filename)[0]
        if key in output_dir:
            output_dir[key].append(img)
        else:
            output_dir[key] = [img]
    return output_dir

# Read the symbol images into list
def read_into_list(directory, re_in):
    output_list = []
    output_labels = []
    for filename in os.listdir(directory+'/'):
        img = cv2.imread(directory+'/' + filename,0)
        img[img <= 100] = 0
        img[img > 100] = 255
        #Remove excess rows and columns that appeared after rotation and padding
        img = cut_excess_white(img)
        output_list.append(img)
        output_labels.append(re.findall(re_in, filename)[0])

    return output_list, output_labels

def resize_by_scale(img, scale):

    """
    Resize the image by scale
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    if not isinstance(scale, (int, float)):
        raise TypeError("scale must be int or float")

    img = cv2.resize(img, (int(img.shape[1]*scale),int(img.shape[0]*scale)))
    return img

def cut_excess_white(img, excess_str = 120):

    """
    Remove excess rows and columns from img
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    if not 0 <= excess_str <= 255:
        raise ValueError("excess_str must be between 0 and 255")
    
    img = img[np.argwhere(np.amin(img,axis=1) < excess_str)[0][0]:np.argwhere(np.amin(img,axis=1) < excess_str)[-1][0],:]
    img = img[:,np.argwhere(np.amin(img,axis=0) < excess_str)[0][0]:np.argwhere(np.amin(img,axis=0) < excess_str)[-1][0]]
    return img

def read_in_labels(file):
    labels_to_nr = {}
    i = 0
    with open(file) as f:
        for line in f:
            labels_to_nr[line.strip('\n')] = i
            i += 1
    return labels_to_nr

def get_labels(label, labels_to_nr):

    """
    Return numerical value of the label
    """

    if not isinstance(label, str):
        raise TypeError("label must be str")

    return labels_to_nr[label]

def get_locations(data_locations, image_height, image_width, offset = 0):

    """
    Computes normalized locations and dimensions of objects in an image.

    Args:
        data_locations (list): A list of tuples where each tuple contains two points (top left and bottom right) that define
                              the location of an object in the image.
        image_height (int): The height of the image.
        image_width (int): The width of the image.
        offset (int, optional): The offset to add to the x coordinate of the bounding box. Defaults to 0.

    Returns:
        numpy.ndarray: An array where each row corresponds to an object and contains 4 elements:
                       - The y-coordinate of the center of the object (normalized to [0,1]).
                       - The x-coordinate of the center of the object (normalized to [0,1]).
                       - The height of the object (normalized to [0,1]).
                       - The width of the object (normalized to [0,1]).
    """

    data_locations = np.array(data_locations)
    data_locations2 = np.zeros((data_locations.shape[0],4))
    data_locations2[:,2] = (data_locations[:,1,1] - data_locations[:,0,1])/image_height
    data_locations2[:,3] = (data_locations[:,1,0] - data_locations[:,0,0])/image_width
    data_locations2[:,0] = (data_locations[:,1,1] + data_locations[:,0,1])/(2*image_height)
    data_locations2[:,1] = ((data_locations[:,1,0] + data_locations[:,0,0])/2 + offset) / image_width
    return data_locations2

def find_cap(img, excess_str = 120):
    """
    Find the location of the cap between letters C/G/S in symbol

    Args:
        img (numpy.ndarray): A grayscale image represented as a 2D numpy array.
        excess_str (int): The threshold value for what is considered a "white" pixel.

    Returns:
        Tuple of left and right bounds of the capital letter in the image.
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    if not 0 <= excess_str <= 255:
        raise ValueError("excess_str must be between 0 and 255")
    values = np.argwhere(np.amin(img,axis=0) > excess_str)

    val_prev = values[0][0]
    sizes = []
    cuts = []
    cuts.append(0)
    size = 1
    for i in range(1,len(values)):
        val = values[i][0]
        if val-val_prev == 1:
            val_prev = val
            size += 1
        else:
            val_prev = val
            sizes.append(size)
            cuts.append(i)
            size = 1
    sizes.append(size)
    cuts.append(i)

    loc = np.argmax(sizes) + 1
    left = values[cuts[loc-1]][0]
    right = values[cuts[loc]-1][0]
    return left, right

def rotate_img(img, rotation,padding_value=255):
    """
    Rotate an image by a given angle.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be rotated.
    rotation : float
        The rotation angle in degrees.
    padding_val : int, optional
        The value to use for padding. Default is 255.

    Returns
    -------
    numpy.ndarray
        The rotated image.
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")

    if not isinstance(rotation, (int, float)):
        raise TypeError("rotation must be a number")
    
    img_rotated = ndimage.rotate(img, rotation, mode='constant',cval=padding_value)
    return img_rotated

def add_unit_symbol_in_middle(img, scale, sample_units, manuever_units,
                                            support_units, resizable, resizable_horizontal,
                                            resizable_vertical, unit_sizes):
    
    """
    Add a unit symbol in the middle of the image.

    Parameters:
    image (np.ndarray): The input image.
    scale (float): The scale of the unit symbol.
    sample_units (dir): The dictionary containing unit symbol samples.
    maneuver_units (list): A list of maneuver units.
    support_units (list): A list of support units.
    resizable (list): A lsit of resizable symbols.
    resizable_horizontal (list): A lsit of resizable horizontal symbols.
    resizable_vertical (list): A list of resizable vertical symbols.
    unit_sizes (list): A list of unit sizes.

    Returns:
    tuple: A tuple containing the following elements:
        - The output image with the unit symbol in the middle.
        - The label of the unit.
        - The rotation angle in degrees.
        - The top-left point of the image without symbol.
        - The bottom-right point of the image without symbol.
    """
    
    #Cut excess white from edges
    img = cut_excess_white(img)

    img, rotation = augment(img, apply_rotation=False, apply_transformation=True, apply_boldness=True, scale_to_binary=True)

    # Find the cap in cover/guard/screen symbol
    left, right = find_cap(img)

    #Get unit_symbol  
    unit_symbol, unit_lab = generate_unit(sample_units,"maneuver",
                                manuever_units,support_units,
                                resizable,resizable_horizontal,
                                resizable_vertical,unit_sizes,False)

    unit_symbol = resize_by_scale(unit_symbol, scale*0.7)

    if img.shape[0] < ceil(unit_symbol.shape[0]/2):
        shape1 = unit_symbol.shape[0]
    else:
        shape1 = ceil(unit_symbol.shape[0]/2) + img.shape[0]

    img2 = np.full([int(shape1),left+unit_symbol.shape[1]+(img.shape[1]-right) + 14], 255) #+14 is for the gap between arrows and unit symbol
    
    # Place the unit symbol into middle
    img2 = place_symbol(img2, unit_symbol, 0, left+7)
    img3 = copy.deepcopy(img2)
    # Place the left side of the screen/cover/guard
    img2 = place_symbol(img2, img[:,0:left], ceil(unit_symbol.shape[0]/2),0)
    # Place the right side of the screen/cover/guard
    img2 = place_symbol(img2, img[:,right:], ceil(unit_symbol.shape[0]/2), left+unit_symbol.shape[1]+14)
    #img2 = img2.astype('float32')
    
    img3[img3 < 100] = 150
    img3 = place_symbol(img3, img[:,0:left], ceil(unit_symbol.shape[0]/2),0)
    img3 = place_symbol(img3, img[:,right:], ceil(unit_symbol.shape[0]/2), left+unit_symbol.shape[1]+14)
    #img3 = img3.astype('float32')
    
    img2pad = np.pad(img2, ((600,600),(600,600)), "constant", constant_values=255)
    img3pad = np.pad(img3, ((600,600),(600,600)), "constant", constant_values=255)
    
    rotation = randint(0,359)
    img2_rotated = ndimage.rotate(img2pad, rotation, reshape=False, mode='constant',cval=255)
    img3_rotated = ndimage.rotate(img3pad, rotation, reshape=False, mode='constant',cval=255)
    
    top1 = np.argwhere(np.amin(img3_rotated,axis=1) < 110)[0][0]
    bottom1 = np.argwhere(np.amin(img3_rotated,axis=1) < 110)[-1][0]
    left1 = np.argwhere(np.amin(img3_rotated,axis=0) < 110)[0][0]
    right1 = np.argwhere(np.amin(img3_rotated,axis=0) < 110)[-1][0]
    
    top2 = np.argwhere(np.amin(img2_rotated,axis=1) < 110)[0][0]
    bottom2 = np.argwhere(np.amin(img2_rotated,axis=1) < 110)[-1][0]
    left2 = np.argwhere(np.amin(img2_rotated,axis=0) < 110)[0][0]
    right2 = np.argwhere(np.amin(img2_rotated,axis=0) < 110)[-1][0]
    
    point1 = (top1-top2,left1-left2)
    point2 = (bottom1-bottom2,right1-right2)
    
    img2_rotated = cut_excess_white(img2_rotated)
    #img3_rotated = cut_excess_white(img3_rotated)

    img2_float = img2_rotated.astype('float32') #OpenCV requires float32 type, cant work with int16
    
    return img2_float, unit_lab, rotation, point1, point2

def get_points_after_rotation(point, rotation):
    x = point[1]*cos(radians(rotation))-point[0]*sin(radians(rotation))
    y = point[1]*sin(radians(rotation))+point[0]*cos(radians(rotation))
    return (y,x)

def get_mortar_area_img(number, scale, sample_extras):
    mortar_img = get_random('mortar', sample_extras)
    m_letter = get_random('m', sample_extras)
    m_letter = cv2.resize(m_letter, (int(m_letter.shape[1]*0.5),int(m_letter.shape[0]*0.5)))
    number = get_random(str(number), sample_extras)
    number = cv2.resize(number, (int(number.shape[1]*0.5),int(number.shape[0]*0.5)))
    center = (int(mortar_img.shape[0]/2),int(mortar_img.shape[1]/2))
    mortar_img = place_symbol(mortar_img, m_letter, center[0]-int(m_letter.shape[0]/2),center[1]-10-m_letter.shape[1])
    mortar_img = place_symbol(mortar_img, number, center[0]-int(number.shape[0]/2),center[1]+10)
    mortar_img = cv2.resize(mortar_img, (int(mortar_img.shape[1]*scale),int(mortar_img.shape[0]*scale)))
    return mortar_img

def get_random(label, sample):
    imgs = sample[label]
    return np.copy(imgs[randint(0,len(imgs)-1)])

def get_noise_img(sample_extras):
    noise_img = get_random('noise', sample_extras)
    noise_img = resize_by_scale(noise_img, 0.17)
    return noise_img

def get_exercise_text(scale, sample, language):

    """
    Generates an image of the word 'exercise' in English or Estonian.

    Args:
        scale (float): The scale factor to resize the images.
        sample (dict): A dictionary of images.
        language (str): The language of the text. Either 'en' for English or 'et' for Estonian.

    Returns:
        np.ndarray: The image of the text.
    """

    letters_dic = {}
    max_height = 0
    length = 0
    if language == 'en':
        letters = ['e','x','e','r','c','i','s','e']
    else:
        letters = ['oline','p','p','u','s']
    for letter in set(letters):
        letter_img = get_random(letter, sample)
        letter_img = resize_by_scale(letter_img, scale)
        if letter_img.shape[0] > max_height:
            max_height = letter_img.shape[0]
        length += letter_img.shape[1]
        letters_dic[letter] = letter_img
    
    max_height += 3

    if language == 'en':
        length += int(7*6) #Add 4pixels between symbols
        length += int(letters_dic['e'].shape[1]*2)
    else:
        length += int(4*6)
        length += letters_dic['p'].shape[1]
    
    text_img = np.full((max_height,length),255)
    pointer = 0

    for letter in letters:
        text_img = place_symbol(text_img, letters_dic[letter],randint(0,3),pointer)
        pointer+=letters_dic[letter].shape[1]+6

    return text_img

def place_exercise_text(canvas, scale, sample, language = None):
    """
    Places the text 'exercise' in English or Estonian on an image.

    Args:
        canvas (np.ndarray): The input image to place the text on.
        scale (float): The scale factor to resize the images.
        sample (dict): A dictionary of images.
        language (str, optional): The language of the text. Either 'en' for English or 'et' for Estonian. Defaults to a random language.

    Returns:
        np.ndarray: The image with the placed text.
    """

    if not isinstance(canvas, np.ndarray):
        raise TypeError("img must be a NumPy array")

    if language is None:
        language = ['en', 'et'][randint(0,1)]
    elif not isinstance(language, str):
        raise TypeError("language must be a str")
    
    scale = scale*0.7
    vertical_loc = randint(16,28)
    mid = int(canvas.shape[1]/2)

    for i in range(2):
        ex_texts = [get_exercise_text(scale, sample, language) for _ in range(3)]
        lines = [resize_by_scale(get_random('line', sample), scale) for _ in range(2)]
        lines = [cv2.resize(line, (int(line.shape[1]*0.5), line.shape[0])) for line in lines]

        canvas = place_symbol(canvas, ex_texts[0], vertical_loc, mid-int(ex_texts[1].shape[1]/2)-lines[0].shape[1]-ex_texts[0].shape[1]-8)
        canvas = place_symbol(canvas, lines[0], vertical_loc+int(ex_texts[0].shape[0]/2)-int(lines[0].shape[0]/2), mid-int(ex_texts[1].shape[1]/2)-lines[0].shape[1]-4)
        canvas = place_symbol(canvas, ex_texts[1], vertical_loc, mid-int(ex_texts[1].shape[1]/2))
        canvas = place_symbol(canvas, lines[1], vertical_loc+int(ex_texts[1].shape[0]/2)-int(lines[1].shape[0]/2), mid+int(ex_texts[1].shape[1]/2)+4)
        canvas = place_symbol(canvas, ex_texts[2], vertical_loc, mid+int(ex_texts[1].shape[1]/2)+lines[1].shape[1]+8)

        vertical_loc = canvas.shape[0]-int(1.1*np.maximum(np.maximum(ex_texts[0].shape[0],ex_texts[1].shape[0]),ex_texts[2].shape[0]))

    return canvas

def inverse(img, binary_threshold=110):
    """
    Inverse the black and white pixels of an image.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be rotated.
    rotation : float
        The rotation angle in degrees.
    binary_threshold : int, optional
        The value to from which the threshold is applied Default is 255.

    Returns
    -------
    numpy.ndarray
        The inverse image.
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")
    # Check if the given threshold is in limits
    if not 0 <= binary_threshold <= 255:
        raise ValueError("binary_threshold must be between 0 and 255")
    # Set the background pixels to 0 and symbols pixels to 255
    img = np.where(img <= binary_threshold, 255, 0)
    return img

# Agument the image
def augment(img, remove_excess = True, excess_str = 110, apply_flip = False,
            flip_random = True, apply_rotation = False, rotation = None,
            padding_val = 255, apply_transformation = False, transformation_dir = None,
            apply_boldness = False, boldness_dir = None, boldness_str = None, add_noise = False,
            noise_threshold = 0.999, scale_to_binary = False, binary_threshold = 110, invert = False, normalize = False):
      
    #Select random image subclass
    #Randomize the size of the image
    #Flip the image along vertical axis.
    if apply_flip:
        if flip_random:
            if randint(0,1) == 0:
                img = cv2.flip(img,1)
        else:
            img = cv2.flip(img,1)
    #Randomize the rotation of the image.
    if apply_rotation:
        if rotation == None:
            rotation = randint(0,359)
        img = rotate_img(img,rotation)
    else:
        rotation = 0
    if apply_transformation:
        #Add padding for affine transformation. Otherwise the picture might not be in bounds.
        img = np.pad(img, (1000, 1000), 'constant', constant_values=(padding_val, padding_val))
        #Originial right triangle
        pts1 = np.float32([[3,3],[3,10],[10,3]])
        if transformation_dir == None:
            transformation_dir = randint(0,2)
        if transformation_dir == 0:
            pts2 = np.float32([[randint(1,5),randint(1,5)],[3,10],[10,3]])
        if transformation_dir == 1:
            pts2 = np.float32([[3,3],[randint(1,5),randint(8,12)],[10,3]])
        if transformation_dir == 2:
            pts2 = np.float32([[3,3],[3,10],[randint(8,12),randint(1,5)]])
        #Get transformation
        M = cv2.getAffineTransform(pts1,pts2)
        #Apply transformation
        img  = cv2.warpAffine(img ,M,(img.shape[1],img.shape[0]),borderValue = 255)
    #Remove excess rows and columns that appeared after rotation and padding
    if remove_excess:
        img = cut_excess_white(img)
    
    #Currently with random we do not dialte(that means we do not erode, some symbols disapper with that)
    #dilation and erosion
    if apply_boldness:
    #If randint = 2, then we apply neither.
        kernel = np.ones((3,3),np.uint8) #Kernel 3x3 seemed to work fine.
        if boldness_dir == None:
            boldness_dir = randint(0,1) #If 0 we apply neither.
        if boldness_str == None:
            boldness_str = randint(1,5)
        if boldness_dir == 1: #erode 1 iteration. #OpenCV erodes white to black. In our case function erode actually dilates.
            img = cv2.erode(img,kernel,iterations = boldness_str)
        if boldness_dir == -1:
            img = cv2.dilate(img,kernel,iterations = boldness_str)
    
    #Apply noise by changing random pixels to one.
    if add_noise:
        img[np.random.rand(img.shape[0],img.shape[1]) > noise_threshold] = 0
    
    #Changes all the pixels with drawing to one and all the "empty" pixels to 0.
    if scale_to_binary:
        if invert:
            img = inverse(img)
        else:
            img[img <= binary_threshold] = 0
            img[img > binary_threshold] = 255
            
    if normalize:
        img = img / 255
    
    return img, rotation
