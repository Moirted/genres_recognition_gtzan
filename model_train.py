import os

import keras.backend as k
import tensorflow as tf
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras


genres = 'blues classical country disco pop hiphop metal reggae rock'
genres = genres.split()


# Создаём папки для обработки датасета
'''
os.makedirs(os.getcwd() + '/gtzan_dataset/genres3sec')
os.makedirs(os.getcwd() + '/gtzan_dataset/spctograms3sec')
os.makedirs(os.getcwd() + '/gtzan_dataset/spctograms3sec/train')
os.makedirs(os.getcwd() + '/gtzan_dataset/spctograms3sec/test')
os.makedirs(os.getcwd() + '/gtzan_dataset/spctograms3sec/not_separated')
for g in genres:
    os.makedirs(os.getcwd() + '/gtzan_dataset/genres3sec/' + g + "/")
    os.makedirs(os.getcwd() + '/gtzan_dataset/spctograms3sec/not_separated/' + g + "/")
    os.makedirs(os.getcwd() + '/gtzan_dataset/spctograms3sec/train/' + g + "/")
    os.makedirs(os.getcwd() + '/gtzan_dataset/spctograms3sec/test/' + g + "/")
print('Были созданы папки для обработки датасета')
'''

# Разбиваем каждый .wav файл датасета на 3-х секундный фрагмент
'''
i = 0
for g in genres:
    j = 0
    print(f"{g}")
    for filename in os.listdir(os.getcwd() + f'/gtzan_dataset/genres/{g}'):
        song = os.path.join(os.getcwd() + f'/gtzan_dataset/genres/{g}', f'{filename}')
        j = j + 1
        for w in range(0, 10):
            i = i + 1
            t1 = 3 * (w) * 1000
            t2 = 3 * (w + 1) * 1000
            newAudio = AudioSegment.from_wav(song)
            new = newAudio[t1:t2]
            new.export(os.getcwd() + f'/gtzan_dataset/genres3sec/{g}/{g + str(j) + str(w)}.wav', format="wav")
print('Было выполнено разбиение аудифайлов на 3-х секундные фрагменты')
'''

# Генерация спектограмм для 3-х секундных фрагментов
'''
for g in genres:
    j = 0
    print(g)
    for filename in os.listdir(os.getcwd() + f'/gtzan_dataset/genres3sec/{g}/'):
        song = os.path.join(os.getcwd() + f'/gtzan_dataset/genres3sec/{g}/{filename}')
        j = j + 1
        y, sr = librosa.load(song, duration=3)
        mels = librosa.feature.melspectrogram(y=y, sr=sr)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
        plt.savefig(os.getcwd() + f'/gtzan_dataset/spctograms3sec/not_separated/{g}/{g + str(j)}.png')
'''

# Разбиваем данные на тренировочные и проверочные
'''
for g in genres:
    print(g)
    filenames = os.listdir(os.getcwd() + f"/gtzan_dataset/spctograms3sec/not_separated/{g}")
    random.shuffle(filenames)
    test_files = filenames[0:100]
    train_files = filenames[100:1000]
    for f in test_files:
        shutil.move(os.getcwd() + f"/gtzan_dataset/spctograms3sec/not_separated/{g}/" + f, os.getcwd() + f"/gtzan_dataset/spctograms3sec/test/{g}")
    for f in train_files:
        shutil.move(os.getcwd() + f"/gtzan_dataset/spctograms3sec/not_separated/{g}/" + f, os.getcwd() + f"/gtzan_dataset/spctograms3sec/train/{g}")
'''

# Создаём генераторы для обоих наборов
train_dir = os.getcwd() + f"/gtzan_dataset/spctograms3sec/train/"
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(288, 432), color_mode="rgba",
                                                    class_mode='categorical', batch_size=128)

validation_dir = os.getcwd() + f"/gtzan_dataset/spctograms3sec/test/"
vali_datagen = ImageDataGenerator(rescale=1. / 255)
vali_generator = vali_datagen.flow_from_directory(validation_dir, target_size=(288, 432), color_mode='rgba',
                                                  class_mode='categorical', batch_size=128)


# Модель
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


# Для вычисления f1_score
def get_f1(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())
    recall = true_positives / (possible_positives + k.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + k.epsilon())
    return f1_val


# Тренируем
model = GenreModel(classes=9)
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', get_f1])

model.fit(train_generator, epochs=70, validation_data=vali_generator)

model.save("genres22.h5")
