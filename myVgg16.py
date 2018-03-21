from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import model_from_yaml
import sys
import cv2
import numpy

def checkImg(imgname, model):
    img_pred = cv2.imread (imgname)
   
    # forces the image to have the input dimensions equal to those used in the training data (28x28)
    if img_pred.shape != [224,224]:
        img2 = cv2.resize (img_pred, (224,224))
        img_pred = img2.reshape (224,224, -1);
    #else:
        #img_pred = img_pred.reshape (150,150, -1);
    img = img_pred
    # here also we inform the value for the depth = 1, number of rows and columns, which correspond 28x28 of the image.
    
    img_pred = img_pred.reshape (1,224,224,3)
    
    pred = model.predict (img_pred)
    #pred_proba = model.predict_proba (img_pred)
    #pred_proba = "% .2f %%"% (pred_proba [0] [pred] * 100)

    print (pred [0], "with probability of")
    #print image
    cv2.imshow ('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# dimensions of our images.
img_width, img_height = 224, 224

#parameters
train_data_dir = 'data/static/01/train'
validation_data_dir = 'data/static/01/validation'
nb_train_samples = 5000
nb_validation_samples = 4480
epochs = 10
batch_size = 16
input_shape = (224, 224, 3)
"""
model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, 
kernel_initializer='glorot_uniform', padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(21, activation='softmax')
])


model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True, decay=0.000005),
              metrics=['accuracy'])


"""
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same' ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(21, activation='softmax'))
"""
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
"""
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True, decay=0.000005),
              metrics=['accuracy'])



# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model.summary()

if raw_input('load model or trian t/l?') == "l":
    # load YAML and create model
    yaml_file = open('my_model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("my_model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #score = loaded_model.evaluate(X_test, y_test, verbose=0)

    checkImg('1.jpg', model)
    checkImg('2.jpg', model)
    checkImg('3.jpg', model)
    checkImg('4.jpg', model)
    checkImg('5.jpg', model)
    checkImg('6.jpg', model)
    checkImg('13.jpg', model)
    checkImg('14.jpg', model)
    checkImg('7.jpg', model)
    checkImg('8.jpg', model)
    checkImg('9.jpg', model)
    checkImg('10.jpg', model)
    checkImg('11.jpg', model)
    checkImg('12.jpg', model)

else:
    # fine-tune the model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose=2)

    score = model.evaluate_generator(validation_generator, nb_validation_samples // batch_size, workers=1)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if raw_input('test images? y/n:') == "y" :
    checkImg('1.jpg', model)
    checkImg('2.jpg', model)
    checkImg('3.jpg', model)
    checkImg('4.jpg', model)
    checkImg('5.jpg', model)
    checkImg('6.jpg', model)
    checkImg('13.jpg', model)
    checkImg('14.jpg', model)
    checkImg('7.jpg', model)
    checkImg('8.jpg', model)
    checkImg('9.jpg', model)
    checkImg('10.jpg', model)
    checkImg('11.jpg', model)
    checkImg('12.jpg', model)

if raw_input('Save model? y/n:') == "y" :
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("my_model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        # serialize weights to HDF5
    model.save_weights("my_model.h5")
    print("Saved model to disk")
