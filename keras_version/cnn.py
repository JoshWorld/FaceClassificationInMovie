from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

# graph
model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(80, 80, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

# train_data processing, generator
BATCH_SIZE = 16

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
        '../blackpink',
        target_size=(80, 80),  # All image resize
        batch_size=BATCH_SIZE,
        class_mode='categorical')

validation_gen = validation_data_gen.flow_from_directory(
        '../blackpink',
        target_size=(80, 80),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

test_gen = test_data_gen.flow_from_directory(
        '../test_data/image',
        target_size=(80, 80),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

# training
model.fit_generator(
        train_generator,
        steps_per_epoch=1000//BATCH_SIZE,
        validation_data=validation_gen,
        validation_steps=10,
        epochs=10)

# saving
#model.save_weights('model')

# evaluating
print("-- Evaluate --")
scores = model.evaluate_generator(test_gen, steps=5)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# using model
print("-- Predict --")
output = model.predict_generator(test_gen, steps=1)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_gen.class_indices)
print(output)