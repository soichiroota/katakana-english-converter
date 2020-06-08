from __future__ import print_function

import os
import pickle

import numpy as np
from tensorflow.keras.layers import Input, Convolution1D, Dot, Dense, Activation, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'data/bep-eng.csv'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
df = pd.read_csv(data_path, keep_default_na=False)
df['english'] = df['english'].str.lower()
df['katakana'] = df['katakana'].str.replace('゛', '').str.replace('゜', '')
for input_text, target_text in zip(df.english, df.katakana):
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
"""
for char in sorted(input_characters):
    print(char)
for char in sorted(target_characters):
    print(char)
"""
input_characters = sorted([' '] + list(input_characters))
target_characters = sorted([' '] + list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
print(target_token_index)

cur_dir = os.path.dirname(__file__)
pickle.dump(input_token_index, open(os.path.join(cur_dir, 'pkl_objects',
   'input_token_index.pkl'), 'wb'), protocol=4)
pickle.dump(target_token_index, open(os.path.join(cur_dir, 'pkl_objects',
   'target_token_index.pkl'), 'wb'), protocol=4)

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# Encoder
x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal')(encoder_inputs)
x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=2)(x_encoder)
x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=4)(x_encoder)

decoder_inputs = Input(shape=(None, num_decoder_tokens))
# Decoder
x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal')(decoder_inputs)
x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=2)(x_decoder)
x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=4)(x_decoder)
# Attention
attention = Dot(axes=[2, 2])([x_decoder, x_encoder])
attention = Activation('softmax')(attention)

context = Dot(axes=[2, 1])([attention, x_encoder])
decoder_combined_context = Concatenate(axis=-1)([context, x_decoder])

decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu',
                                padding='causal')(decoder_combined_context)
decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu',
                                padding='causal')(decoder_outputs)
# Output
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

# Run training
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
encoder_x_train, encoder_x_test, decoder_x_train, decoder_x_test, y_train, y_test, indices_train, indices_test = train_test_split(
    encoder_input_data, decoder_input_data, decoder_target_data, np.arange(len(decoder_target_data)), test_size=0.01, random_state=111)
model.fit([encoder_x_train, decoder_x_train], y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.01,
          callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)])
# Save model
model.save('katakana_english_converter/keras/h5_objects/cnn_s2s.h5')

# Next: inference mode (sampling).

# Define sampling models
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


nb_examples = 100
scores = []
in_encoder = encoder_input_data[list(indices_test)]
in_decoder = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

in_decoder[:, 0, target_token_index["\t"]] = 1

predict = np.zeros(
    (len(input_texts), max_decoder_seq_length),
    dtype='float32')

for i in range(max_decoder_seq_length - 1):
    predict = model.predict([in_encoder, in_decoder])
    predict = predict.argmax(axis=-1)
    predict_ = predict[:, i].ravel().tolist()
    for j, x in enumerate(predict_):
        in_decoder[j, i + 1, x] = 1


for i, seq_index in enumerate(indices_test):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    output_seq = predict[i, :].ravel().tolist()
    decoded = []
    for x in output_seq:
        if reverse_target_char_index[x] == "\n":
            break
        else:
            decoded.append(reverse_target_char_index[x])
    decoded_sentence = "".join(decoded)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
    print('True sentence:', target_texts[seq_index])
    reference = [list(target_texts[seq_index])[1:-1]]
    prediction = list(decoded_sentence)
    bleu = sentence_bleu(reference, prediction)
    print(bleu)
    scores.append(bleu)
print('BLEU:', np.mean(scores))
