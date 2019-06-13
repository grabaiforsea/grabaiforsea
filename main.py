import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


if __name__ == '__main__':
        
        try:
                car_classes_mat = sio.loadmat('devkit/cars_annos.mat')
                for class_names in car_classes_mat['class_names']:
                        car_names = []
                        for x in class_names:
                                car_names.append(x[0])

                cat_dataset_mat = sio.loadmat('devkit/cars_train_annos.mat')
                for data in cat_dataset_mat['annotations']:
                        dataset_list = []
                        for x in data:
                                dataset_list.append(x)

                config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
                sess = tf.Session(config=config) 
                keras.backend.set_session(sess)

                #variables
                num_classes = 196
                batch_size = 256
                epochs = 8

                lines = np.array(dataset_list)

                num_of_instances = lines.size

                #initialize trainset and test set
                x_train, y_train = [], []

                #transfer train and test set data
                for i in range(1, num_of_instances):
                        car_class, img, usage = lines[i].split(",")
                        
                        val = img.split(" ")
                        
                        pixels = np.array(val, 'float32')
                        
                        car = keras.utils.to_categorical(car_class, num_classes)
                
                        y_train.append(car)
                        x_train.append(pixels)

                #data transformation for train and test sets
                x_train = np.array(x_train, 'float32')
                y_train = np.array(y_train, 'float32')

                x_train /= 255 #normalize inputs between [0, 1]

                x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
                x_train = x_train.astype('float32')

                #construct CNN structure
                model = Sequential()

                #1st convolution layer
                model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
                model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

                #2nd convolution layer
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

                #3rd convolution layer
                model.add(Conv2D(128, (3, 3), activation='relu'))
                model.add(Conv2D(128, (3, 3), activation='relu'))
                model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

                model.add(Flatten())

                #fully connected neural networks
                model.add(Dense(1024, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(1024, activation='relu'))
                model.add(Dropout(0.2))

                model.add(Dense(num_classes, activation='softmax'))

                #batch process
                gen = ImageDataGenerator()
                train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

                model.compile(loss='categorical_crossentropy'
                        , optimizer=keras.optimizers.Adam()
                        , metrics=['accuracy']
                )

                fit = True

                if fit == True:
                        model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs) #train for randomly selected one
                else:
                        raise Exception('Model cannot be fitted')

                model.save('car_classifier.model')

                def cars_classify(cars):
                        objects = tuple(car_names)
                        y_pos = np.arange(len(objects))
                        
                        plt.bar(y_pos, cars, align='center', alpha=0.5)
                        plt.xticks(y_pos, objects)
                        plt.ylabel('percentage')
                        plt.title('car name')
                        
                        plt.show()

                     
                #make prediction for custom image out of test set

                img = image.load_img("resources/girlwoman.jpg", grayscale=True, target_size=(48, 48))

                x = image.img_to_array(img)
                x = np.expand_dims(x, axis = 0)

                x /= 255

                custom = model.predict(x)
                cars_classify(custom[0])

                x = np.array(x, 'float32')
                x = x.reshape([48, 48])

                plt.gray()
                plt.imshow(x)
                plt.show()

        except Exception as e:
                print(e, end="")