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

    model = CNNModel()

    images = './images/'
    labels = './labels/'
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
                x = int((float(line.split(" ")[1])-(float(line.split(" ")[3])/2)) * imageW) - 25
                y = int((float(line.split(" ")[2])-(float(line.split(" ")[4])/2)) * imageH) - 25
                w = int(float(line.split(" ")[3]) * imageW + 25)
                h = int(float(line.split(" ")[4]) * imageH + 25)

                crop_image = image[y:y+h, x:x+w]
                # print(crop_image.shape)
                tmp_list_class.append(int(line.split(" ")[0]))
                dim = (80, 80)
                # cv2.imshow("Test", crop_image)
                # cv2.waitKey()
                resized_img = cv2.resize(crop_image, dim, interpolation = cv2.INTER_AREA)

                # Predict
                reshaped_img = resized_img.reshape(1, 80, 80)
                model.load_weights('./rotation_models/'+Dict[int(line.split(" ")[0])]+'_rotation_model_31_1804.h5')
                predicted_rotation = model.predict(reshaped_img, verbose=0)
                degreePrediction = np.argmax(predicted_rotation)
                line += f" {degreePrediction}\n"
                new_lines += line

                # if line.split(" ")[0] in ["0", "2", "9", "18"]:
                #     # To see the prediction
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     font_scale = 0.5
                #     font_color = (200, 0, 0)
                #     thickness = 2
                #     text_size, _ = cv2.getTextSize(str(degreePrediction), font, font_scale, thickness)
                #     x_ = int((resized_img.shape[1] - text_size[0]) / 2)
                #     y_ = int((resized_img.shape[0] + text_size[1]) / 2)
                #     cv2.putText(resized_img, str(degreePrediction), (x_, y_), font, font_scale, font_color, thickness)
                #     cv2.imshow("Test", resized_img)
                #     cv2.waitKey()

        with open(viz_labels + str(name).split(".")[0] + ".txt", "w") as f_w:
            f_w.write(new_lines)


def CNNModel():
    classes = 360

    input = Input(shape=(80, 80, 1))
    x = Conv2D(32, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu')(input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x)
    x = Conv2D(32, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x)
    x = Conv2D(64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x)
    x = Conv2D(64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x)
    x = Conv2D(64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    # print(model.summary())

    return model


main()
