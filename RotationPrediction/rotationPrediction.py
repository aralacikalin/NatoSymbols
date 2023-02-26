from tqdm import tqdm
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Conv2D, MaxPooling2D
import os
import numpy as np
import cv2


def main():
    Dict = {0: 'advance_to_contact', 1: 'ambush', 2: 'attack', 3: 'attack_by_fire', 4: 'block',
            5: 'breach', 6: 'clear', 7: 'contain', 8: 'control', 9: 'counterattack', 10: 'cover',
            11: 'delay', 12: 'deny', 13: 'destroy', 14: 'disrupt', 15: 'fix', 16: 'guard',
            17: 'isolate', 18: 'main_attack', 19: 'neutralize', 20: 'occupy', 21: 'penetrate',
            22: 'retain', 23: 'retire', 24: 'screen', 25: 'secure', 26: 'seize', 27: 'support_by_fire',
            28: 'suppress', 29: 'turn', 30: 'withdraw'}

    model = modelCNN()

    images = '../data/images/'
    labels = '../data/labels/'
    viz_images = '../SymbolVisualizer/images/'
    viz_labels = '../SymbolVisualizer/labels/'
    file_list = os.listdir(images)

    for name in tqdm(file_list):
        tmp_list_class = []
        image = cv2.imread(images+name, cv2.IMREAD_GRAYSCALE)
        imageW = image.shape[1]
        imageH = image.shape[0]
        new_lines = ""
        with open(labels+str(name).split(".")[0]+".txt", "r") as f:
            lines = [line.rstrip('\n') for line in f]
            for line in lines:
                x = int((float(line.split(" ")[1])-(float(line.split(" ")[3])/2)) * imageW)
                y = int((float(line.split(" ")[2])-(float(line.split(" ")[4])/2)) * imageH)
                w = int(float(line.split(" ")[3]) * imageW + 5)
                h = int(float(line.split(" ")[4]) * imageH + 5)

                crop_image = image[y:y+h, x:x+w]
                tmp_list_class.append(int(line.split(" ")[0]))
                dim = (75, 75)
                resized_img = cv2.resize(crop_image, dim, interpolation = cv2.INTER_AREA)

                # Predict
                reshaped_img = resized_img.reshape(1, 75, 75)
                model.load_weights('./rotation_models/'+Dict[int(line.split(" ")[0])]+'_rotation_model_31.h5')
                predicted_rotation = model.predict(reshaped_img, verbose=0)
                line += f" {np.argmax(predicted_rotation)}\n"
                new_lines += line

        with open(viz_labels + str(name).split(".")[0] + ".txt", "w") as f_w:
            f_w.write(new_lines)

def modelCNN():
    classes = 360

    input_ = Input(shape=(75, 75, 1))
    x = Conv2D(64, 3, 3, padding = 'same', activation='relu')(input_)
    x = Conv2D(64, 3, 3, padding = 'same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, 3, padding = 'same', activation='relu')(x)
    x = Conv2D(64, 3, 3, padding = 'same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=input_, outputs=x)

    return model


main()
