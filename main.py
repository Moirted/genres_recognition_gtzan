import os

import librosa
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.utils import load_img, img_to_array
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import Normalize
from pydub import AudioSegment
from tensorflow import keras

class_labels = ['blues',
                'classical',
                'country',
                'disco',
                'hiphop',
                'metal',
                'pop',
                'reggae',
                'rock']


def GenreModel(classes=9):
    X_input = keras.Input(shape=(288, 432, 4))

    X = layers.Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(X_input)
    X = layers.BatchNormalization(axis=3)(X)
    X = keras.activations.relu(X)
    X = layers.MaxPooling2D((2, 2))(X)

    X = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = keras.activations.relu(X)
    X = layers.MaxPooling2D((2, 2))(X)

    X = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = keras.activations.relu(X)
    X = layers.MaxPooling2D((2, 2))(X)

    X = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1))(X)
    X = layers.BatchNormalization(axis=-1)(X)
    X = keras.activations.relu(X)
    X = layers.MaxPooling2D((2, 2))(X)

    X = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1))(X)
    X = layers.BatchNormalization(axis=-1)(X)
    X = keras.activations.relu(X)
    X = layers.MaxPooling2D((2, 2))(X)

    X = layers.Flatten()(X)
    X = layers.Dropout(rate=0.3)(X)

    X = layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = keras.Model(inputs=X_input, outputs=X, name='GenreModel')

    return model


model = GenreModel(classes=9)
model.load_weights("genres_model.h5")


def convert_mp3_to_wav(music_file):
    sound = AudioSegment.from_mp3(music_file)
    sound.export("music_file.wav", format="wav")


def extract_relevant(wav_file, t1, t2):
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000 * t1:1000 * t2]
    wav.export("extracted.wav", format='wav')


def create_melspectrogram(wav_file):
    y, sr = librosa.load(wav_file, duration=3)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
    plt.savefig('melspectrogram.png')


def predict(image_data, model):
    image = img_to_array(image_data)
    image = np.reshape(image, (1, 288, 432, 4))
    prediction = model.predict(image / 255)
    prediction = prediction.reshape((9,))
    class_label = np.argmax(prediction)
    return class_label, prediction


def show_output(songfile):
    convert_mp3_to_wav(songfile)
    extract_relevant("music_file.wav", 10, 20)
    create_melspectrogram("extracted.wav")
    image_data = load_img('melspectrogram.png', color_mode='rgba', target_size=(288, 432))

    class_label, prediction = predict(image_data, model)

    print(f"## Жанр песни {songfile}:" + " " + class_labels[class_label])

    prediction = prediction.reshape((9,))

    color_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    my_cmap = cm.get_cmap('jet')
    my_norm = Normalize(vmin=0, vmax=9)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x=class_labels, height=prediction,
           color=my_cmap(my_norm(color_data)))
    plt.xticks(rotation=45)
    ax.set_title("Вероятностное распределение данной песни по разным жанрам")
    plt.show()
    os.remove(os.getcwd() + "/music_file.wav")
    os.remove(os.getcwd() + "/melspectrogram.png")
    os.remove(os.getcwd() + "/extracted.wav")


show_output('nirvana-sltn.mp3')
