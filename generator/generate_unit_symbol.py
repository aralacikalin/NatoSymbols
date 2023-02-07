import numpy as np
import cv2
from random import randint
import random

def get_random(label, sample):
    imgs = sample[label]
    return np.copy(imgs[randint(0,len(imgs)-1)])

def resize_img(template, unit_img, unit_lab : str, resizable: list, resizable_horizontal: list, resizable_vertical: list):
    unit_img2 = np.full(template.shape, 255)
    if unit_lab in resizable:
        unit_img = cv2.resize(unit_img, [template.shape[1],template.shape[0]])
        unit_img2 = unit_img
    elif unit_lab in resizable_horizontal:
        unit_img = cv2.resize(unit_img, [template.shape[1],unit_img.shape[0]])
        if unit_lab == "hq_unit":
            unit_img2[int(unit_img2.shape[0]*0.2):(int(unit_img2.shape[0]*0.2)+unit_img.shape[0]),:] = unit_img
        else:
            unit_img2[int(unit_img2.shape[0]*0.7):(int(unit_img2.shape[0]*0.7)+unit_img.shape[0]),:] = unit_img
    elif unit_lab in resizable_vertical:
        unit_img = cv2.resize(unit_img, [unit_img.shape[1],template.shape[0]])
        if unit_lab == "motorized":
            unit_img2[:,int(unit_img2.shape[1]/2-unit_img.shape[1]/2):((int(unit_img2.shape[1]/2-unit_img.shape[1]/2)+unit_img.shape[1]))] = unit_img
        else:
            unit_img2[:,int(unit_img2.shape[1]*0.1):(int(unit_img2.shape[1]*0.1)+unit_img.shape[1])] = unit_img
    else:
        if unit_lab == "mortar":
            unit_img = cv2.resize(unit_img, [int(template.shape[0]*0.3),int(template.shape[0]*0.8)])
            unit_img2[int(unit_img2.shape[0]*0.1):(int(unit_img2.shape[0]*0.1)+unit_img.shape[0]),int(unit_img2.shape[1]/2 - unit_img.shape[1]/2):(int(unit_img2.shape[1]/2 - unit_img.shape[1]/2) + unit_img.shape[1])] = unit_img
        if unit_lab == "artillery":
            unit_img = cv2.resize(unit_img, [int(template.shape[0]*0.2),int(template.shape[0]*0.2)])
            unit_img2[int(unit_img2.shape[0]*0.4):(int(unit_img2.shape[0]*0.4)+unit_img.shape[0]),int(unit_img2.shape[1]/2 - unit_img.shape[1]/2):(int(unit_img2.shape[1]/2 - unit_img.shape[1]/2) + unit_img.shape[1])] = unit_img
        if unit_lab == "engineers" or unit_lab == "combat_service":
            unit_img = cv2.resize(unit_img, [int(template.shape[0]*0.6),int(template.shape[0]*0.3)])
            unit_img2[int(unit_img2.shape[0]*0.35):(int(unit_img2.shape[0]*0.35)+unit_img.shape[0]),int(unit_img2.shape[1]/2 - unit_img.shape[1]/2):(int(unit_img2.shape[1]/2 - unit_img.shape[1]/2) + unit_img.shape[1])] = unit_img
        if unit_lab == "armour":
            unit_img = cv2.resize(unit_img, [int(template.shape[0]*0.9),int(template.shape[0]*0.55)])
            unit_img2[int(unit_img2.shape[0]*0.2):(int(unit_img2.shape[0]*0.2)+unit_img.shape[0]),int(unit_img2.shape[1]/2 - unit_img.shape[1]/2):(int(unit_img2.shape[1]/2 - unit_img.shape[1]/2) + unit_img.shape[1])] = unit_img
        if unit_lab == "sniper":
            unit_img = cv2.resize(unit_img, [int(template.shape[0]*0.3),int(template.shape[0]*0.2)])
            unit_img2[int(unit_img2.shape[0]*0.15):(int(unit_img2.shape[0]*0.15)+unit_img.shape[0]),int(unit_img2.shape[1]/2 - unit_img.shape[1]/2):(int(unit_img2.shape[1]/2 - unit_img.shape[1]/2) + unit_img.shape[1])] = unit_img
        if unit_lab == "wheeled":
            unit_img = cv2.resize(unit_img, [int(template.shape[0]*0.4),int(template.shape[0]*0.15)])
            unit_img2[(-unit_img.shape[0]-int(unit_img2.shape[0]*0.1)):-int(unit_img2.shape[0]*0.1),int(unit_img2.shape[1]/2 - unit_img.shape[1]/2):(int(unit_img2.shape[1]/2 - unit_img.shape[1]/2) + unit_img.shape[1])] = unit_img
        if unit_lab == "missile":
            unit_img = cv2.resize(unit_img, [int(template.shape[0]*0.3),int(template.shape[0]*0.6)])
            unit_img2[int(unit_img2.shape[0]*0.05):(int(unit_img2.shape[0]*0.05)+unit_img.shape[0]),int(unit_img2.shape[1]/2 - unit_img.shape[1]/2):(int(unit_img2.shape[1]/2 - unit_img.shape[1]/2) + unit_img.shape[1])] = unit_img
        if unit_lab == "gun_system":
            unit_img = cv2.resize(unit_img, [int(template.shape[0]*0.3),int(template.shape[0]*0.45)])
            unit_img2[int(unit_img2.shape[0]*0.2):(int(unit_img2.shape[0]*0.2)+unit_img.shape[0]),int(unit_img2.shape[1]/2 - unit_img.shape[1]/2):(int(unit_img2.shape[1]/2 - unit_img.shape[1]/2) + unit_img.shape[1])] = unit_img
        if unit_lab == "air_defence":
            unit_img = cv2.resize(unit_img, [template.shape[1],int(unit_img.shape[0]*(template.shape[1]/unit_img.shape[1]))])
            unit_img2[-unit_img.shape[0]:,:] = unit_img
            
    return unit_img2

def add_unit_size(sample, template, unit_size_img, unit_size_label: str):
    unit_size_img2 = np.full((int(template.shape[0]*1.2),template.shape[1]) ,255)
    unit_size_img2[-template.shape[0]:,-template.shape[1]:] = template
    width = 2
    if unit_size_label in ['company', 'battalion', 'regiment']:
        unit_size_img = cv2.resize(unit_size_img, [int(template.shape[0]*0.08),int(template.shape[0]*0.18)])
    else:
        unit_size_img = cv2.resize(unit_size_img, [int(template.shape[0]*0.18),int(template.shape[0]*0.18)])
    if unit_size_label in ['half-platoon','battalion','division']:
        unit_size_img2[int(template.shape[0]*0.01):(int(template.shape[0]*0.01)+unit_size_img.shape[0]),int(template.shape[1]/2 - unit_size_img.shape[1]*1.05):(int(template.shape[1]/2 - unit_size_img.shape[1]*1.05)+unit_size_img.shape[1])] = unit_size_img
        unit_size_img2[int(template.shape[0]*0.01):(int(template.shape[0]*0.01)+unit_size_img.shape[0]),int(template.shape[1]/2 + unit_size_img.shape[1]*0.05):(int(template.shape[1]/2 + unit_size_img.shape[1]*0.05)+unit_size_img.shape[1])] = unit_size_img
        width = 2.5
    elif unit_size_label in ['platoon', 'regiment']:
        unit_size_img2[int(template.shape[0]*0.01):(int(template.shape[0]*0.01)+unit_size_img.shape[0]),int(template.shape[1]/2 - unit_size_img.shape[1]/2):(int(template.shape[1]/2 - unit_size_img.shape[1]/2)+unit_size_img.shape[1])] = unit_size_img
        unit_size_img2[int(template.shape[0]*0.01):(int(template.shape[0]*0.01)+unit_size_img.shape[0]),int(template.shape[1]/2 - unit_size_img.shape[1]*1.65):(int(template.shape[1]/2 - unit_size_img.shape[1]*1.65)+unit_size_img.shape[1])] = unit_size_img
        unit_size_img2[int(template.shape[0]*0.01):(int(template.shape[0]*0.01)+unit_size_img.shape[0]),int(template.shape[1]/2 + unit_size_img.shape[1]/2*1.3):int(template.shape[1]/2 + unit_size_img.shape[1]/2*1.3+unit_size_img.shape[1])] = unit_size_img
        width = 3
    else:
        unit_size_img2[int(template.shape[0]*0.01):(int(template.shape[0]*0.01)+unit_size_img.shape[0]),int(template.shape[1]/2 - unit_size_img.shape[1]/2):(int(template.shape[1]/2 - unit_size_img.shape[1]/2)+unit_size_img.shape[1])] = unit_size_img
    if random.uniform(0, 1) > 0.9 and (unit_size_label in ['company', 'battalion', 'regiment']):
        unit_size_img22 = np.full((int(unit_size_img2.shape[0]*1.05),unit_size_img2.shape[1]) ,255)
        unit_size_img22[-unit_size_img2.shape[0]:,-unit_size_img2.shape[1]:] = unit_size_img2
        unit_size_img = get_random('unit_tactical', sample)
        unit_size_img = cv2.resize(unit_size_img, [int(template.shape[0]*0.2*width),int(template.shape[0]*0.2)])
        unit_size_img22[int(unit_size_img2.shape[0]*0.01):(int(unit_size_img2.shape[0]*0.01)+unit_size_img.shape[0]),int(unit_size_img22.shape[1]/2 - unit_size_img.shape[1]/2):int(unit_size_img22.shape[1]/2 - unit_size_img.shape[1]/2) + unit_size_img.shape[1]][unit_size_img == 0] = 0
        unit_size_img2 = unit_size_img22
    return unit_size_img2

def generate_unit(sample: dict,
                  lab: str,
                  manuever_units = ['infrantry', #Currently used as global variables
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
                              'battalion', 'brigade', 'regiment', 'division'],
                  can_be_hq = False):
    template = get_random('template', sample)
    
    #Get the random label for iamge if one is not provided
    if lab == "maneuver":
        unit_lab = manuever_units[randint(0,len(manuever_units)-1)]
        unit_img = get_random(unit_lab, sample)
    elif lab == "support":
        unit_lab = support_units[randint(0,len(support_units)-1)]
    else:
        unit_lab = lab
    
    #For these two we use line_horizontal from sample
    if 'hq_unit' == unit_lab or 'supply' == unit_lab:
        unit_img = get_random('line_horizontal', sample)
    else:
        unit_img = get_random(unit_lab, sample)
    
    #Returns unit_img with same shape as template and with symbol in appropriate space.
    unit_img = resize_img(template, unit_img, unit_lab, resizable, resizable_horizontal, resizable_vertical)
    #Add symbol to template.
    template[unit_img == 0] = 0
    
    #Add additional information to symbol randomly.
    is_motorized = False
    if unit_lab != "armour":
        if random.uniform(0, 1) > 0.8 and unit_lab != 'air_defence':
            unit_img = resize_img(template, get_random('armour', sample), 'armour', resizable, resizable_horizontal, resizable_vertical)
            template[unit_img == 0] = 0
        elif random.uniform(0, 1) > 0.9:
            unit_img = resize_img(template, get_random('line_vertical', sample), "motorized", resizable, resizable_horizontal, resizable_vertical)
            template[unit_img == 0] = 0
            is_motorized = True
    if random.uniform(0, 1) > 0.9 and not is_motorized:
        unit_img = resize_img(template, get_random('wheeled', sample), "wheeled", resizable, resizable_horizontal, resizable_vertical)
        template[unit_img == 0] = 0
    if random.uniform(0, 1) > 0.9:
        unit_img = resize_img(template, get_random('line_vertical', sample), "cannon", resizable, resizable_horizontal, resizable_vertical)
        template[unit_img == 0] = 0
    if random.uniform(0, 1) > 0.97 and can_be_hq:
        unit_img2 = np.full((int(template.shape[0]*1.6),template.shape[1]), 255)
        unit_img2[:template.shape[0],:template.shape[1]] = template
        unit_img = get_random('line_vertical', sample)
        unit_img = cv2.resize(unit_img, [int(unit_img.shape[1]*(int(template.shape[0])/unit_img.shape[0])),int(template.shape[0]*0.6)])
        unit_img2[template.shape[0]:,:unit_img.shape[1]][unit_img == 0] = 0
        template = unit_img2
        
    #Add unit size
    #Generate random unit size
    unit_size_lab = unit_sizes[randint(0,len(unit_sizes)-1)]
    image_lab = unit_size_lab
    if unit_size_lab in ['half-platoon', 'platoon']:
        image_lab = "squad"
    elif unit_size_lab in ['battalion', 'regiment']:
        image_lab = "company"
    elif unit_size_lab in ['division', 'regiment']:
        image_lab = "brigade"
    
    unit_size_img = get_random('size_' + image_lab, sample)
    
    template = add_unit_size(sample, template, unit_size_img, unit_size_lab)
        
    return template.astype('float32'), unit_lab

