from __future__ import print_function
from scipy.misc import imsave
import time
from keras import backend as K
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
from keras.preprocessing import image
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy


#parameters
img_width, img_height = 150, 150
train_data_dir = 'data/static/02/train'
validation_data_dir = 'data/static/02/validation'
test_data_dir = 'data/static/02/test'

nb_train_samples = 3500
nb_validation_samples = 4480
epochs = 20
batch_size = 10

def getCategory(num):
    categories = ["Public Transport", "Driving", "Walking outdoor", "Walking indoor",
    "Biking","Drinking together", "Drinking/eating alone", "Eating together", "Socializing",
    "Attending a seminar", "Meeting", "Reading", "TV", "Cleaning and chores", "Working",
    "Cooking","Shopping","Talking", "Resting", "Mobile", "Plane"]
    return categories[num]

def checkImg(imgname, model):
    img_pred = cv2.imread (imgname, 0)
    img = img_pred
    test_image = image.load_img(imgname, target_size = (img_width, img_height))
    test_image = image.img_to_array(test_image)
    test_image = numpy.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print ( getCategory(numpy.argmax(result)), "with probability of", (result * 100))
    #print image
    cv2.imshow ('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = numpy.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = numpy.clip(x, 0, 255).astype('uint8')
    return x


#Build model
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(21, activation='softmax'))

#complie
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
model.summary()


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

if __name__ == '__main__':

    print ("1.- Load Model")
    print ("2.- Train Model")
    print ("3.- Visualize Model")
    op = raw_input()
    if  op == 1:
        # load YAML and create model
        yaml_file = open('model.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])
        score = model.evaluate_generator(test_generator, nb_validation_samples // batch_size, workers=1)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

        checkImg('1.jpg', loaded_model)
        checkImg('2.jpg', loaded_model)
        checkImg('3.jpg', loaded_model)
        checkImg('4.jpg', loaded_model)
        checkImg('5.jpg', loaded_model)
        checkImg('6.jpg', loaded_model)
        checkImg('7.jpg', loaded_model)
        checkImg('8.jpg', loaded_model)
        checkImg('9.jpg', loaded_model)
        checkImg('10.jpg', loaded_model)
        checkImg('11.jpg', loaded_model)
        checkImg('12.jpg', loaded_model)
        checkImg('13.jpg', loaded_model)
        checkImg('14.jpg', loaded_model)
        model = loaded_model
    if  op == 2:

        # fine-tune the model
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

        """
        model = KerasClassifier(build_fn = build_model, epochs=100, batch_size=10, verbose=0)
        parameters = {'learn_rate': [0.001, 0.01, 0.1, 0.2, 0.3], 'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]}
        grid_search = GridSearchCV(estimator = model,
                                   param_grid = parameters,
                                   scoring = 'accuracy',
                                   cv = 10,
                                   n_jobs = -1)
        grid_search = grid_search.fit(train_generator)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        """


        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open("model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
            # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

        checkImg('1.jpg', model)
        checkImg('2.jpg', model)
        checkImg('3.jpg', model)
        checkImg('4.jpg', model)
        checkImg('5.jpg', model)
        checkImg('6.jpg', model)
        checkImg('7.jpg', model)
        checkImg('8.jpg', model)
        checkImg('9.jpg', model)
        checkImg('10.jpg', model)
        checkImg('11.jpg', model)
        checkImg('12.jpg', model)
        checkImg('13.jpg', model)

    if op == "3":
        # load YAML and create model
        yaml_file = open('model.yaml', 'r')
        model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(model_yaml)
        # load weights into new model
        model.load_weights("model.h5")
        print("Loaded model from disk")

        # dimensions of the generated pictures for each filter.
        img_width = 150
        img_height = 150

        # the name of the layer we want to visualize
        # (see model definition at keras/applications/vgg16.py)
        layer_name = 'block5_conv1'
        # this is the placeholder for the input images
        input_img = model.input

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


        def normalize(x):
            # utility function to normalize a tensor by its L2 norm
            return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


        kept_filters = []
        for filter_index in range(200):
            # we only scan through the first 200 filters,
            # but there are actually 512 of them
            print('Processing filter %d' % filter_index)
            start_time = time.time()

            # we build a loss function that maximizes the activation
            # of the nth filter of the layer considered
            layer_output = layer_dict[layer_name].output
            if K.image_data_format() == 'channels_first':
                loss = K.mean(layer_output[:, filter_index, :, :])
            else:
                loss = K.mean(layer_output[:, :, :, filter_index])

            # we compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, input_img)[0]

            # normalization trick: we normalize the gradient
            grads = normalize(grads)

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_img], [loss, grads])

            # step size for gradient ascent
            step = 1.

            # we start from a gray image with some random noise
            if K.image_data_format() == 'channels_first':
                input_img_data = numpy.random.random((1, 3, img_width, img_height))
            else:
                input_img_data = numpy.random.random((1, img_width, img_height, 3))
            input_img_data = (input_img_data - 0.5) * 20 + 128

            # we run gradient ascent for 20 steps
            for i in range(20):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                print('Current loss value:', loss_value)
                if loss_value <= 0.:
                    # some filters get stuck to 0, we can skip them
                    break

            # decode the resulting input image
            if loss_value > 0:
                img = deprocess_image(input_img_data[0])
                kept_filters.append((img, loss_value))
            end_time = time.time()
            print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

        # we will stich the best 64 filters on a 8 x 8 grid.
        n = 8

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top 64 filters.
        kept_filters.sort(key=lambda x: x[1], reverse=True)
        kept_filters = kept_filters[:n * n]

        # build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        margin = 5
        width = n * img_width + (n - 1) * margin
        height = n * img_height + (n - 1) * margin
        stitched_filters = numpy.zeros((width, height, 3))

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                img, loss = kept_filters[i * n + j]
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                                 (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

        # save the result to disk
        imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)


        # util function to convert a tensor into a valid image
