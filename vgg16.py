
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
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
    pred_proba = pred

    print (numpy.argmax(pred), "with probability of", pred * 100)
    #print image
    cv2.imshow ('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# dimensions of our images.
img_width, img_height = 150, 150

#parameters
train_data_dir = 'data/static/02/train'
validation_data_dir = 'data/static/02/validation'
test_data_dir = 'data/static/02/test'
nb_train_samples = 35000
nb_validation_samples = 4480
epochs = 20
batch_size = 10

# create model
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(21, activation='softmax'))

#complie
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer='Adam',
              metrics=['accuracy'])



# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

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

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


model.summary()

if raw_input('load model or trian t/l?') == "l":
    # load YAML and create model
    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True, decay=0.000005),
                  metrics=['accuracy'])
    score = loaded_model.evaluate_generator(test_generator, nb_validation_samples // batch_size, workers=1)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

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

# fine-tune the model
else:
	history = model.fit_generator(
	    train_generator,
	    steps_per_epoch=nb_train_samples // batch_size,
	    epochs=epochs,
	    validation_data=validation_generator,
	    validation_steps=nb_validation_samples // batch_size,
	    verbose=2)

	score = model.evaluate_generator(test_generator, nb_validation_samples // batch_size, workers=1)
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


	if raw_input('Save model? y/n:') == "y" :
	    # serialize model to YAML
	    model_yaml = model.to_yaml()
	    with open("model.yaml", "w") as yaml_file:
		yaml_file.write(model_yaml)
		# serialize weights to HDF5
	    model.save_weights("model.h5")
	    print("Saved model to disk")

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
