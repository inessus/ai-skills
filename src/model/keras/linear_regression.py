from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np


x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
 
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


model = Sequential()
model.add(Dense(units=1, kernel_initializer="uniform", activation="linear", input_dim=1))


model.compile(optimizer='sgd',
             loss='mse',
             metrics=['accuracy'])

print('Training --------')
model.fit(x_train, y_train, epochs=500, batch_size=8)
