from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# graph
model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(80, 80, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

# data processing
BATCH_SIZE = 16

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(80, 80),  # All image resize
        batch_size=BATCH_SIZE,
        class_mode='categorical')

validation_data_gen = validation_data_gen.flow_from_directory(
        'data/train',
        target_size=(80, 80),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

test_data_gen = test_data_gen.flow_from_directory(
        'data/train',
        target_size=(80, 80),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

