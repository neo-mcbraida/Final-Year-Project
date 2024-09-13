from matplotlib import pyplot as plt
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import os
from tqdm.auto import trange
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from sklearn.model_selection import train_test_split


def load_images_from_folder(folder_path, image_size=(64, 64), test_size=0.2, random_state=None):
    images = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img)
            images.append(img_array)
    images = np.asarray(images)

    # Split the dataset into train, test, and validation sets
    X_train, X_test = train_test_split(images, test_size=test_size, random_state=random_state)
    return X_train, X_test

# set folder_path the relevant data set
folder_path = ""
X_train, X_test = load_images_from_folder(folder_path, test_size=0.1)
X_train = (X_train / 127.5) - 1.0
X_test = (X_test / 127.5) - 1.0

def step(x_inp, x_time, filter_num):
    # simple convoluton
    x = layers.Conv2D(filter_num, kernel_size=3, padding='same', activation="relu")(x_inp)
    # x_time provides context about how far along the diffusion process it is
    x_time = layers.Dense(filter_num, activation="relu")(x_time)
    # combine
    x *= layers.Reshape((1, 1, filter_num))(x_time)
    # then just standard steps for image processing model
    x += layers.Conv2D(filter_num, kernel_size=3, padding='same')(x_inp)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def downsample(x_inp, x_t, filter_num):
    x = step(x_inp, x_t, filter_num)
    x = layers.MaxPool2D(2)(x)
    return x

def upsample(x_inp, residual, x_t, filter_num):
    x = layers.Concatenate()([x_inp, residual])
    x = step(x, x_t, filter_num)
    x = layers.UpSampling2D(2)(x)
    return x

def make_model(im_shape, t_shape, filter_num= 128, time_param=128):
    x_inp = layers.Input(shape=im_shape)
    t_inp = layers.Input(shape=t_shape)
    x_t = layers.Dense(time_param, activation='relu')(t_inp)
    x_t = layers.LayerNormalization()(x_t)

    # downsample
    # unique case so that the last layer output has a residual connection
    x64 = step(x_inp, x_t, filter_num)
    x32 = downsample(x64, x_t, filter_num)
    x16 = downsample(x32, x_t, filter_num)
    x8 = downsample(x16, x_t, filter_num)
    x4 = downsample(x8, x_t, filter_num)

    # middle
    x = layers.Flatten()(x4)
    x = layers.Concatenate()([x, x_t])
    x = layers.Dense(filter_num)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)
    fact = int(filter_num / 4)
    x = layers.Dense(4 * 4 * fact)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Reshape((4, 4, fact))(x)

    # upsample
    x = upsample(x, x4, x_t, filter_num)
    x = upsample(x, x8, x_t, filter_num)
    x = upsample(x, x16, x_t, filter_num)
    x = upsample(x, x32, x_t, filter_num)
    x = layers.Concatenate()([x, x64])
    x = step(x, x_t, filter_num)

    x = layers.Conv2D(3, kernel_size=1, padding='same')(x)
    model = tf.keras.models.Model([x_inp, t_inp], x)
    return model

BATCH_SIZE = 16
TIME_STEPS = 20
IMAGE_SIZE = 64
EPOCHS = 50

time_bar = 1 - np.linspace(0, 1.0, TIME_STEPS + 1) # linspace for timesteps
gen_steps = time_bar[0:TIME_STEPS]
t_zero = (time_bar[1:TIME_STEPS+1])[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
t_minus_one = (time_bar[0:TIME_STEPS])[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

# create steps
def linear_noise(images):
    # Convert list of images to a NumPy array for efficient processing
    images_array = np.array(images)
    noise = np.random.normal(size=images_array.shape)
    img_t_minus_1 = noise * t_minus_one + images_array * (1 - t_minus_one)
    img_t = noise * t_zero + images_array * (1 - t_zero)
    return img_t_minus_1.reshape(-1, *images_array.shape[1:]), img_t.reshape(-1, *images_array.shape[1:])

# perform one epoch
def train_epoch(R=50):
    R = len(X_train)
    step = BATCH_SIZE
    total_loss = 0.0
    bar = trange(0, R, step)
    for i in bar:
        if i + step >= R:
            continue
        x_imgs = X_train[i:i+step]
        inp_ims, out_ims = linear_noise(x_imgs)
        loss = model.train_on_batch([inp_ims, np.tile(gen_steps, BATCH_SIZE)], out_ims)
        total_loss += loss
        pg = (i / R) * 100
        if i % 5 == 0:
            bar.set_description(f'loss: {loss:.5f}, p: {pg:.2f}%')
    return total_loss / R

def predict(num_ims=16):
    x = []
    for _ in range(num_ims):
        im = np.random.normal(size=(1, IMAGE_SIZE, IMAGE_SIZE, 3))
        for i in range(TIME_STEPS):
            im = model.predict([im, np.asarray([i])], verbose=0)
        x.extend(im)
    return np.asarray(x)

def save_images(epoch, images):
    # set this path to wherever you want to save your images
    image_path = ""
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(image_path + str(epoch) + "/", exist_ok=True)
    for i in range(len(images)):
        save_img(os.path.join(image_path + str(epoch), f"{str(i)}_saved.png"), images[i])

model_path = ""
model = make_model((IMAGE_SIZE,IMAGE_SIZE,3), (1,), time_param=192)
model.summary()
model.compile(optimizer=Adam(lr=0.0005), loss=tf.keras.losses.MeanAbsoluteError(), metrics=['accuracy'])

for epoch in range(EPOCHS):
    train_epoch()
    model.optimizer.learning_rate = max(0.000001, model.optimizer.learning_rate * 0.9)
    model.save(model_path + str(epoch) + ".h5")
    # show result
    ims = predict()
    save_images(epoch, ims)