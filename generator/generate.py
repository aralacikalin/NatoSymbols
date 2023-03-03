import glob
import numpy as np
import cv2
from random import randint
import random
import os
import argparse
from tqdm import tqdm #For progressbar
from generate_unit_symbol import *
from generator_utils import *
import real_background_generation_utils as background_utils


def generate_image(sample,
                   sample_real,
                   sample_units,
                   sample_extras,
                   scale,
                   manuever_units,
                   support_units,
                   resizable,
                   resizable_horizontal,
                   resizable_vertical,
                   unit_sizes,
                   dim = (3468, 4624),
                   excess_str = 110,
                   real_symbols_ratio = 0.0):
    canvas = np.full(dim, 255) #Size of final image
    
    location_placement = []
    locations = [] #Store location of the symbols
    locations_units = []
    
    labels = [] #Store labels
    labels_units = []
    
    rotations = []
    
    location_of_placement = randint(0,0)

    sample_labels = list(sample.keys())

    placement = get_random('placement', sample_extras)
    placement = cv2.resize(placement, [int(placement.shape[1]*0.3*scale), int(placement.shape[0]*0.3*scale)])
    if location_of_placement == 0:
        offset0 = randint(50,250)
        offset1 = randint(50,250)
        #Add text in middle of placement
        if randint(0,1) == 0:
            first_letter = get_random(['e','f','l','m','n'][randint(0,4)], sample_extras)
            second_letter = get_random(['e','f'][randint(0,1)], sample_extras)
            first_letter = resize_by_scale(first_letter, scale)
            second_letter = resize_by_scale(second_letter, scale)
            
            placement_mid = (int(placement.shape[0]/2),int(placement.shape[1]/2))

            placement = place_symbol(placement,first_letter,placement_mid[0]-int(first_letter.shape[0]/2),placement_mid[1]-int(first_letter.shape[1])-10)
            placement = place_symbol(placement,second_letter,placement_mid[0]-int(second_letter.shape[0]/2),placement_mid[1]+10)

        canvas = place_symbol(canvas,placement,offset0,canvas.shape[1]-placement.shape[1]-offset1)
        
        location_placement.append(((offset0,canvas.shape[1]-offset1-placement.shape[1]),
                                   (offset0+placement.shape[0],canvas.shape[1]-offset1)))
        
        point_up = np.arange(placement.shape[0])[placement[:,50] < excess_str][1]
        point_down = np.arange(placement.shape[0])[placement[:,50] < excess_str][-2]
        point_left = np.arange(placement.shape[1])[placement[-50,:] < excess_str][1]
        point_right = np.arange(placement.shape[1])[placement[-50,:] < excess_str][-2]
        #Add location numbers to placement symbols
        #Generate the grid number
        grid1 = randint(0,99)
        canvas = add_grid_number(canvas,point_down,point_left,offset0,offset1,grid1,placement,sample_extras, scale = 0.75*scale)
        canvas = add_grid_number(canvas,point_up,point_left,offset0,offset1,(grid1+1)%100,placement,sample_extras, scale = 0.75*scale)
        grid2 = randint(0,99)
        canvas = add_grid_number2(canvas,point_down,point_left,offset0,offset1,grid2,placement,sample_extras, scale = 0.75*scale)
        canvas = add_grid_number2(canvas,point_up,point_right,offset0,offset1,(grid2+1)%100,placement,sample_extras, scale = 0.75*scale)
        length = point_down-point_up
        point1 = offset0+point_down+((canvas.shape[0]-offset0-point_down)//length-1)*length
        point2 = (canvas.shape[1]-offset1-(placement.shape[1]-point_left))%length+length
        
        placement2 = get_random('2placement', sample_extras)
        placement2 = cv2.resize(placement2, (length,length))
        point1_1 = point1-int(placement2.shape[0]/2)
        point2_1 = point2-int(placement2.shape[1]/2)
        
        canvas = place_symbol(canvas, placement2, point1_1, point2_1)
        
        grid1 = (grid1 - ((canvas.shape[0]-offset0-point_down)//length-1))%100
        canvas = add_grid_number(canvas,point1,point2+length,0,canvas.shape[1]-point2-int(placement2.shape[1]/2),grid1,placement2,sample_extras)
        grid2 = (grid2 - int((canvas.shape[1]-offset1-(placement.shape[1]-point_left+length)-point2)/length))%100
        canvas = add_grid_number2(canvas,point1,-point2,point1-int(placement2.shape[0]/2),canvas.shape[1],grid2,placement2,sample_extras)
        location_placement.append(((point1_1,point2_1),(point1_1+placement2.shape[0],point2_1+placement2.shape[1])))
        
    if randint(0,1) == 0:
        canvas = place_exercise_text(canvas, scale, sample_extras)    
    
    for task in range(randint(3,6)): # Nr of symbols on image
        label = sample_labels[randint(0,len(sample_labels)-1)]
        if random.uniform(0, 1) > real_symbols_ratio:
            img = get_random(label, sample)
            from_real_film = False
        else:
            try:
                #There might not be sample from real film
                img = get_random(label, sample_real)
                from_real_film = True
            except:
                img = get_random(label, sample)
                from_real_film = False
        img_scale = random.uniform(0.7,1.0)
        img = resize_by_scale(img, img_scale*0.8)
        
        if not from_real_film:
            #Insert unit symbol to screen, cover and guard.
            if label in ['screen', 'cover', 'guard']:
                img, unit_lab, rotation, point_1, point_2 = add_unit_symbol_in_middle(img, scale, sample_units, manuever_units,
                                                support_units, resizable, resizable_horizontal,
                                                resizable_vertical, unit_sizes)
            else:
                # Augment the image
                img, rotation = augment(img, apply_rotation=True, apply_transformation=True, apply_boldness=True)
        else:
            #TODO Need to predefine rotations for real films.
            rotation = 0
        
        labels.append(label)
        rotations.append(rotation)
        
        #Check if there is overlap with current symbols.
        #If there is overlap the generate new locations and check again.
        point1, point2 = get_points(dim, img, locations, locations_units,location_placement)
        
        #We append upper left corner point and lower right corner point.
        if label in ['screen', 'cover', 'guard']:
            locations.append(((point1+point_1[0],point2+point_1[1]),(point1+img.shape[0]+point_2[0],point2+img.shape[1]+point_2[1])))
        else:
            locations.append(((point1,point2),(point1+img.shape[0],point2+img.shape[1])))
        #If there is overlap we don't want to overwrite black pixels with white background.
        canvas = place_symbol(canvas, img, point1, point2)
        
        #Draw phase line
        if label in ['attack', 'counterattack', 'advance_to_contact']:
            if random.uniform(0, 1) > 0.4:
                canvas = draw_line(canvas, point1, point2, rotations[-1], 1, img)
                canvas = draw_line(canvas, point1, point2, rotations[-1], -1, img)
        
        if label not in ['screen', 'cover', 'guard']:
            if random.uniform(0, 1) > 0.2:
                #Generate unit symbol
                unit_symbol, unit_lab = generate_unit(sample_units,"maneuver",
                                                      manuever_units,support_units,
                                                      resizable,resizable_horizontal,
                                                      resizable_vertical,unit_sizes)
                unit_symbol = cv2.resize(unit_symbol, [int(unit_symbol.shape[1]*scale), int(unit_symbol.shape[0]*scale)])

                #Scale unit symbol
                #scale = img.shape[0] / unit_symbol.shape[0]
                #new_size = (int(int(unit_symbol.shape[1])*scale),int(int(unit_symbol.shape[0])*scale))
                #unit_symbol = cv2.resize(unit_symbol, new_size)

                #Find the angle for unit symbol
                length = np.max(img.shape)
                
                center = [int(point1+img.shape[0]/2), int(point2+img.shape[1]/2)]
                point2_1, point1_1 = point_location(rotations[-1],img.shape,center,unit_symbol.shape)
                #point1_1 = point1+y_dir
                #point2_1 = point2+x_dir

                # We only place symbol if there is no overlap. If there is then unit symbol is not added.
                if ((point1_1+unit_symbol.shape[0] < dim[0]) and (point2_1+unit_symbol.shape[1] < dim[1]) and (point1_1 >= 0) and (point2_1 >= 0)):
                    is_overlap = (check_overlap(point1_1,point2_1,locations,unit_symbol.shape) or
                                 check_overlap(point1_1,point2_1,locations_units,unit_symbol.shape) or
                                 check_overlap(point1_1,point2_1,location_placement,unit_symbol.shape))

                    if not is_overlap:
                        canvas = place_symbol(canvas, unit_symbol, point1_1, point2_1)
                        locations_units.append(((point1_1,point2_1),(point1_1+unit_symbol.shape[0],point2_1+unit_symbol.shape[0])))
                        labels_units.append(unit_lab)
    
    # Add overlapping support_by_fire
    """
    for i in range(randint(0,3)):
        
        overlapping_img, point1_1, point2_1, point1_2, point2_2, shape1, shape2, rot1, rot2 = get_overlapping_support_by_fire(scale, sample)

        point1, point2 = get_points(dim, overlapping_img, locations, locations_units,location_placement)

        canvas = place_symbol(canvas, overlapping_img, point1, point2)

        locations_units.append(((point1_1,point2_1),(point1_1+unit_symbol.shape[0],point2_1+unit_symbol.shape[0])))
        labels_units.append(unit_lab)

        locations_units.append(((point1_2,point2_2),(point1_2+unit_symbol.shape[0],point2_2+unit_symbol.shape[0])))
        labels_units.append(unit_lab)
    """
    #Add mortar unit locations
    for i in range(randint(1,4)):
        mortar_img = get_mortar_area_img(i, scale, sample_extras)
        
        point1, point2 = get_points(dim, mortar_img, locations, locations_units,location_placement)
        
        canvas = place_symbol(canvas, mortar_img, point1, point2)

    #Add random dots
    for i in range(randint(0,10)):
        noise_img = get_noise_img(sample_extras)

        point1 = randint(0,dim[0]-noise_img.shape[0])
        point2 = randint(0,dim[1]-noise_img.shape[1])

        canvas = place_symbol(canvas, noise_img, point1, point2)
    
    return canvas, locations, labels, rotations, locations_units, labels_units

def main(
    dim_h = 3468,
    dim_w = 4624,
    save_dim_h = 578,
    save_dim_w = 770,
    save_as_square = False,
    examples_nr = 1,
    symbols_dir = 'data/symbols',
    real_symbols_dir = 'data/real_symbols',
    unit_symbols_dir = 'data/unit_symbols',
    extras_dir = 'data/extras',
    symbols_regex = '([a-zA-Z_ ]*)\d*.*',
    units_regex = '([1234a-zA-Z_ ]*)\d*.*',
    extras_regex = '([0-9a-zA-Z]*)\d*.*',
    excess_str = 110,
    save_images_dir = '',
    save_labels_dir = '',
    save_rotations_dir = '',
    manuever_units = ['infrantry',
                        'anti_tank',
                        'armour',],
    support_units = ['recce',
                    'medic',
                    'signal',
                    'hq_unit',
                    'supply',
                    'artillery',
                    'mortar',
                    'air_defence'],
    resizable = ['infrantry',
                'anti_tank',
                'recce',
                'medic',
                'signal'],
    resizable_horizontal = ['hq_unit','supply'],
    resizable_vertical = ['motorized', 'cannon'],
    unit_sizes = ['team', 'squad', 'half-platoon', 'platoon', 'company', #Currently repetition because unit sizes are sampled with uniform distirubution
                  'team', 'squad', 'half-platoon', 'platoon', 'company',
                  'squad', 'half-platoon', 'platoon', 'company',
                  'battalion'],  #'brigade', 'regiment', 'division']
    real_backgrounds_ratio=0.0,
    real_backgrounds_dir="data/real_backgrounds",
    real_symbols_ratio=0.0
):  
    
    save_dim = (save_dim_h,save_dim_w)
    dim = (dim_h,dim_w)
    # Read in the tactical symbols
    sample = {}
    for dir in os.listdir(symbols_dir):
        sample = read_into_dic(f'{symbols_dir}/{dir}', symbols_regex, sample)

    sample_real = {}
    for dir in os.listdir(real_symbols_dir):
        sample_real = read_into_dic(f'{real_symbols_dir}/{dir}', symbols_regex, sample_real)
    
    # Read in the unit symbols
    sample_units = read_into_dic(unit_symbols_dir, units_regex)

    # Read in the extra symbols (letters, grid placement, etc)
    sample_extras = read_into_dic(extras_dir, extras_regex)

    # If save directoris do not exists then create these
    if not os.path.exists(save_images_dir):
        os.makedirs(save_images_dir)
    if not os.path.exists(save_labels_dir):
        os.makedirs(save_labels_dir)
    if not os.path.exists(save_rotations_dir):
        os.makedirs(save_rotations_dir)

    data_labels = []
    data_locations = []
    data_rotations = []
    indices = []

    # references for real background generation
    backgroundImageList=background_utils.ProcessBackgrounds(real_backgrounds_dir)
    

    realBackgroundSampleCount=round(examples_nr*real_backgrounds_ratio)
    realBackgroundSample=random.sample(range(examples_nr),realBackgroundSampleCount)

    for i in tqdm(range(examples_nr)):
        not_successful = True
        while not_successful:
            try:
                scale = random.uniform(0.5,1.2)
                if(i in realBackgroundSample):
                    canvas,boundingBoxesToRemove,background_dim=random.choice(backgroundImageList)
                    canvas=canvas.copy()
                    img, loc, lab, rot, loc_units, lab_units  = background_utils.generate_image_with_real_background(
                                                                                        boundingBoxesToRemove,
                                                                                        sample, 
                                                                                        canvas, 
                                                                                        background_dim,dim)
                                        
                else:
                    img, loc, lab, rot, loc_units, lab_units  = generate_image(sample, sample_real, sample_units,
                                                                            sample_extras, scale, manuever_units,
                                                                            support_units,
                                                                            resizable,
                                                                            resizable_horizontal,
                                                                            resizable_vertical,
                                                                            unit_sizes, dim, excess_str, real_symbols_ratio)
                                    #Conversion to float is needed to use resize

                #img[img<110] = 1
                #img[img>=110] = 0
                #Conversion to float is needed to use resize
                img = cv2.resize(img.astype('float32'), (int(save_dim[1]),int(save_dim[0]))).astype('int16')
                offset = 0
                if save_as_square:
                    img2 = np.full((save_dim[1], save_dim[1]), 255)
                    offset = int((save_dim[1]-save_dim[0])/2)
                    img = place_symbol(img2,img,offset,0)
                cv2.imwrite(f'{save_images_dir}/img{i}.jpg',img)
                data_labels.append(lab)
                data_locations.append(loc)
                data_rotations.append(rot)
                indices.append(i)
                not_successful = False
            except:
                continue
    
    labels_to_nr = read_in_labels('data/labels.txt')

    for i in range(len(data_labels)):
        labels = list(map(lambda label : get_labels(label, labels_to_nr), data_labels[i]))
        if save_as_square:
            locations = get_locations(data_locations[i],dim[1],dim[1],offset)
        else:
            locations = get_locations(data_locations[i],dim[1],dim[0],offset)
        with open(f'{save_labels_dir}/img{indices[i]}.txt', 'w') as f:
            for k, lab in enumerate(labels):
                if (k != len(labels)-1):
                    f.write(f'{lab} {locations[k,0]} {locations[k,1]} {locations[k,2]} {locations[k,3]}\n')
                else:
                    f.write(f'{lab} {locations[k,0]} {locations[k,1]} {locations[k,2]} {locations[k,3]}')
    
    for i in range(len(data_rotations)):
        with open(f'{save_rotations_dir}/img{indices[i]}.txt', 'w') as f:
            for k, rot in enumerate(data_rotations[i]):
                if (k != len(data_rotations[i])-1):
                    f.write(f'{rot}\n')
                else:
                    f.write(f'{rot}')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_h', type=int, default = 3468, help='The dimension which is being used during generating, should be same in which the symbols samples are taken')
    parser.add_argument('--dim_w', type=int, default = 4624, help='The dimension which is being used during generating, should be same in which the symbols samples are taken')
    parser.add_argument('--save_dim_h', type=int, default = 578, help='Dimesion in which the generated images are saved')
    parser.add_argument('--save_dim_w', type=int, default = 770, help='Dimesion in which the generated images are saved')
    parser.add_argument('--save_as_square', type=bool, default = False, help='If the saved iamge hieght and width are equal. Uses save_dim[1] as dimension for both')
    parser.add_argument('--examples_nr', type=int, default = 1, help='Number of images to generate')
    parser.add_argument('--symbols_dir', type=str, default='data/symbols', help='Directory in which the sample of tactical tasks are')
    parser.add_argument('--real_symbols_dir', type=str, default='data/real_symbols', help='Directory in which the sample of tactical tasks cut from real films are')
    parser.add_argument('--unit_symbols_dir', type=str, default='data/unit_symbols', help='Directory in which the sample of unit symbols are')
    parser.add_argument('--extras_dir', type=str, default='data/extras', help='Directory in which the sample of extras is')
    parser.add_argument('--save_images_dir', type=str, default='images/train', help='Directory where to store generated images')
    parser.add_argument('--save_labels_dir', type=str, default='labels/train', help='Directory where to store labels')
    parser.add_argument('--save_rotations_dir', type=str, default='rotations/train', help='Directory where to store rotations')
    parser.add_argument('--real_backgrounds_ratio', type=float, default = 0.0, help='Ratio of data with real backgrounds')
    parser.add_argument('--real_backgrounds_dir', type=str, default = "data/real_backgrounds", help='Directory in which the real data backgrounds are')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))