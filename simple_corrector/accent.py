# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %% [markdown]
# ## Idea
# - Giảm thời gian generate data
# - Train nhiều dữ liệu hơn, dùng GPU
# - Dùng drop out
# - Fit dùng validation set cho val_acc
# - Generate data có dấu
#     - Giữ dấu/Bỏ bớt dấu/Xóa dấu
#     - Áp dụng cho cả target character và ngữ cảnh
#
# - Sử dụng data chat (câu ngắn, ít chữ)

# %%
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import unicodedata
import re
import string
import math


# %%
# get_ipython().system('tar -xvf /content/drive/MyDrive/chatbot/Binhvq_News_Corpus/corpus-title.tar.gz')


# %%
# get_ipython().system('wc -l corpus-title.txt')


# %%
# get_ipython().system('head corpus-title.txt')


# %%
class Encoder:

    def __init__(self):
        self.build_alphabet()

    def __len__(self):
        return len(self.alphabet)

    def build_alphabet(self):
        accented_chars = [
            'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
            'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
            'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
            'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
            'í', 'ì', 'ỉ', 'ĩ', 'ị',
            'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
            'đ',
        ]

        accented_chars.extend([c.upper() for c in accented_chars])

        alphabet = list(chr(0)
                        + chr(1)
                        + string.printable
                        + ''.join(accented_chars))

        self.alphabet = alphabet
        self.index_to_char = dict(enumerate(alphabet))
        self.char_to_index = dict((c, i) for i, c in enumerate(alphabet))

    def text_to_sequence(self, text):
        seq = [self.char_to_index.get(c) for c in text]
        seq = [i if i is not None else 1 for i in seq]
        return seq

    def sequence_to_text(self, seq):
        return [self.index_to_char[i] for i in seq]

    def one_hot(self, sequence, one_hot_length):
        seq_length = len(sequence)
        result = np.zeros((seq_length, one_hot_length))
        result[np.arange(seq_length), sequence] = 1
        return result

    def one_hot_scalar(self, value, one_hot_length):
        result = np.zeros((one_hot_length, ))
        result[value] = 1
        return result


encoder = Encoder()


# %%
class AccentStripper:

    def __init__(self):
        self.build_accent_stripped_map()

    def build_accent_stripped_map(self):
        latin_to_accented_char = {
            'a': ['á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', ],
            'o': ['ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', ],
            'e': ['é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', ],
            'u': ['ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', ],
            'i': ['í', 'ì', 'ỉ', 'ĩ', 'ị', ],
            'y': ['ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', ],
            'd': ['đ', ]

        }

        map = dict()
        for k, cs in latin_to_accented_char.items():
            for c in cs:
                map[c] = k

            k = k.upper()
            cs = [c.upper() for c in cs]
            for c in cs:
                map[c] = k

        accented_chars = set()
        accented_chars.update(map.keys())
        accented_chars.update(map.values())

        self.accented_char_map = map
        self.accented_chars = accented_chars

    def is_target_char(self, c):
        return c in self.accented_chars

    def strip_accent(self, text):
        chars = [c if self.accented_char_map.get(
            c) is None else self.accented_char_map.get(c) for c in text]
        return ''.join(chars)


accent_stripper = AccentStripper()


# %%
CONST_before_sequence_length = 15
CONST_after_sequence_length = 15
CONST_sequence_length = CONST_before_sequence_length + \
    CONST_after_sequence_length + 1
CONST_alphabet_length = len(encoder.alphabet)


# %%
def generate_samples(text):
    text = text.numpy().decode('utf-8')
    padding_before = chr(0) * CONST_before_sequence_length
    padding_after = chr(0) * CONST_after_sequence_length

    text = padding_before + text + padding_after
    stripped_text = accent_stripper.strip_accent(text)
    sequence = encoder.text_to_sequence(stripped_text)

    xs = []
    ys = []

    for i, c in enumerate(text):
        if not accent_stripper.is_target_char(c):
            continue

        start = i - CONST_before_sequence_length
        end = i + CONST_after_sequence_length + 1

        x_sequence = sequence[start:end]
        y_sequence = encoder.text_to_sequence(c)[0]

        xs.append(x_sequence)
        ys.append(y_sequence)

    return xs, ys


def create_generator(ds_text):
    for text in ds_text:
        xs, ys = generate_samples(text)
        if len(ys) == 0:
            continue

        xs = tf.constant(xs, dtype=tf.int32)
        ys = tf.constant(ys, dtype=tf.int32)
        yield xs, ys


def create_datasets():
    def create_dataset_from_generator(ds_text):
        ds = tf.data.Dataset.from_generator(
            lambda: create_generator(ds_text),
            output_signature=(
                tf.TensorSpec(shape=(None, CONST_sequence_length),
                              dtype=tf.int32),
                tf.TensorSpec(shape=(None, ), dtype=tf.int32)))

        ds = ds.unbatch()
        ds = ds.batch(256)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    test_size = int(1e4)
    val_size = int(1e3)
    train_size = int(1e6)

    ds = tf.data.TextLineDataset('/content/corpus-title.txt')
    ds = ds.take(test_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    ds_test = create_dataset_from_generator(ds)

    ds = tf.data.TextLineDataset('/content/corpus-title.txt')
    ds = ds.skip(test_size).take(val_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    ds_val = create_dataset_from_generator(ds)

    ds = tf.data.TextLineDataset('/content/corpus-title.txt')
    ds = ds.skip(test_size).skip(val_size).take(train_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    ds_train = create_dataset_from_generator(ds)

    return ds_train, ds_val, ds_test

# %% [markdown]
# ## Model


# %%


# model = keras.Sequential([
#     layers.Embedding(input_dim=CONST_alphabet_length,
#                      output_dim=CONST_alphabet_length,
#                      embeddings_initializer=tf.keras.initializers.Constant(
#                          np.eye(CONST_alphabet_length)),
#                      input_length=CONST_sequence_length,
#                      trainable=False),
#     layers.Flatten(),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(CONST_alphabet_length, activation='softmax')
# ])

# model.summary()


# %%
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


# %%
# ds_train, ds_val, ds_test = create_datasets()

# check_point_cb = tf.keras.callbacks.ModelCheckpoint(
#     filepath='/content/drive/MyDrive/chatbot/TMT_accent_stripped_vietnamese',
#     monitor='val_accuracy',
#     save_best_only=True,
#     verbose=1)

# early_stopping_cb = tf.keras.callbacks.EarlyStopping(
#     monitor='val_accuracy',
#     patience=1,
#     restore_best_weights=True
# )

# model.fit(ds_train, validation_data=ds_val, epochs=3,
#           callbacks=[early_stopping_cb, check_point_cb])


# %%
# result = model.evaluate(ds_test)
# result

# %% [markdown]
# ## Test a specific sentence

# %%
model = keras.models.load_model('./TMT_accent_stripped_vietnamese')


# %%
# text = '"Cô gái đưa tin lên Facebook đó là phóng viên, đúng ra cô này cũng thuộc nhóm đối tượng ưu tiên trong các nhóm theo quy định của Bộ Y tế.'
# print(f'Original: {text}')

# text = unicodedata.normalize('NFC', text)
# text = text.strip()
# text = accent_stripper.strip_accent(text)
# print(f'Stripped: {text}')

# xs, _ = generate_samples(tf.constant(text))
# ys_pred = model.predict(tf.constant(xs))
# ys_pred = np.argmax(ys_pred, axis=-1)
# ys_pred = encoder.sequence_to_text(ys_pred)

# accented_text = []
# i = 0
# for c in text:
#     if accent_stripper.is_target_char(c):
#         accented_text.append(ys_pred[i])
#         i += 1
#     else:
#         accented_text.append(c)

# accented_text = ''.join(accented_text)
# print(f'Toned:    {accented_text}')


def get_accented(text):
    xs, _ = generate_samples(tf.constant(text))
    ys_pred = model.predict(tf.constant(xs))
    ys_pred = np.argmax(ys_pred, axis=-1)
    ys_pred = encoder.sequence_to_text(ys_pred)

    accented_text = []
    i = 0
    for c in text:
        if accent_stripper.is_target_char(c):
            accented_text.append(ys_pred[i])
            i += 1
        else:
            accented_text.append(c)

    accented_text = ''.join(accented_text)
    return accented_text
