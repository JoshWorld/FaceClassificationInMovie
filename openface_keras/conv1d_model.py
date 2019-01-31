# 0. 사용할 패키지 불러오기
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Flatten, Dropout
from keras.layers import Conv1D, GlobalMaxPooling1D
from openface_keras import image_to_embedding

max_features = 20000

x, y = image_to_embedding.image_to_embedding('data')

print(x.shape)


x_val = x[:int(0.2*len(x))]
y_val = y[:int(0.2*len(y))]
x_train = x[int(0.2*len(x)): int(0.8*len(x))]
y_train = y[int(0.2*len(y)): int(0.8*len(x))]
x_test = x[int(0.8*len(x)):]
y_test = y[int(0.8*len(y)):]

print(len(x_test), len(x_val), len(x_train))
print(y_val)
print(x_val)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=128))
model.add(Dropout(0.2))
model.add(Conv1D(256,
                 2,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(1, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=100, batch_size=5, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([-0.2, 1.2])

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylim([-0.2, 1.2])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)