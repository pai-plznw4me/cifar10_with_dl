import numpy as np
from urllib.request import urlretrieve
import tarfile
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# download dataset 
urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "cifar-10-python.tar.gz")

# extract tar.gz 
fname = './cifar-10-python.tar.gz'
if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()


# load dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


xs = unpickle('/content/cifar-10-batches-py/data_batch_1')[b'data'].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
xs = xs.reshape([-1, 3 * 1024])
ys = unpickle('/content/cifar-10-batches-py/data_batch_1')[b'labels']

# train 묶어서 하나의 학습 데이터 셋으로 만들기.
xs_bucket = []
ys_bucket = []
for i in range(1, 6):
    # append image data to list
    xs = unpickle('/content/cifar-10-batches-py/data_batch_{}'.format(i))[b'data'].reshape((10000, 3, 32, 32)). \
        transpose((0, 2, 3, 1))
    xs = xs.reshape([-1, 3 * 1024])
    xs_bucket.append(xs)

    # append label to list
    ys = unpickle('/content/cifar-10-batches-py/data_batch_{}'.format(i))[b'labels']
    ys_bucket.append(ys)

train_xs = np.concatenate(xs_bucket, axis=0).reshape((-1, 32, 32, 3))
train_ys = np.concatenate(ys_bucket, axis=0)

# test 데이터 셋 가져오기
test_xs = unpickle('/content/cifar-10-batches-py/test_batch')[b'data'].reshape((10000, 3, 32, 32)).transpose(
    (0, 2, 3, 1))
test_xs = test_xs
test_ys = unpickle('/content/cifar-10-batches-py/test_batch')[b'labels']

# input layer
input_ = Input(shape=(32, 32, 3))

# block 1
layer = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu',
               kernel_initializer='he_normal')(input_)
layer = MaxPool2D(strides=2)(layer)

# block 2
layer = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
               kernel_initializer='he_normal')(layer)
layer = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu',
               kernel_initializer='he_normal')(layer)
layer = MaxPool2D(strides=2)(layer)

# block 3
layer = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
               kernel_initializer='he_normal')(layer)
layer = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
               kernel_initializer='he_normal')(layer)
layer = MaxPool2D(strides=2)(layer)

# block4
layer = Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
               activation='relu', kernel_initializer='he_normal')(layer)

# Flatten
layer = Flatten()(layer)

# Fullyconntec layer
layer = Dense(units=256, activation='relu', kernel_initializer='he_normal')(layer)
layer = Dense(units=256, activation='relu', kernel_initializer='he_normal')(layer)

# logits
layer = Dense(units=10, activation='softmax', kernel_initializer='he_normal')(layer)

# Model
model = Model(input_, layer)
model.compile(loss='categorical_crossentropy', metrics=['acc'])

# training
onehot_train_ys = to_categorical(train_ys, 10)
onehot_test_ys = to_categorical(test_ys, 10)
model.fit(train_xs / 255., onehot_train_ys, epochs=100, validation_data=(test_xs / 255., onehot_test_ys))
