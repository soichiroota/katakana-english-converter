import os
import json
import re

import numpy as np
from tensorflow.keras.models import Model, load_model
import pandas as pd


data_path = 'data/test.json'

# Vectorize the data.

with open(data_path) as fp:
    input_texts = json.load(fp)

cur_dir = os.path.dirname(__file__)
with open(os.path.join(
    cur_dir, 'json', 'input_token_index.json')
) as fp:
    input_token_index = json.load(fp)
with open(os.path.join(
    cur_dir, 'json', 'target_token_index.json')
) as fp:
    target_token_index = json.load(fp)

"""
# Define the model that will turn
model = load_model('katakana_english_converter/keras/h5_objects/cnn_s2s.h5')
model.summary()

# Next: inference mode (sampling).

reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

in_encoder = encoder_input_data
in_decoder = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

in_decoder[:, 0, target_token_index["\t"]] = 1

def model_predict(
    input_texts_,
    max_decoder_seq_length_,
    model_,
    in_encoder_,
    in_decoder_,
):
    predict = np.zeros(
        (len(input_texts_), max_decoder_seq_length_),
        dtype='float32'
    )
    for i in range(max_decoder_seq_length_ - 1):
        predict = model_.predict([in_encoder_, in_decoder_])
        predict = predict.argmax(axis=-1)
        predict_ = predict[:, i].ravel().tolist()
        for j, x in enumerate(predict_):
            in_decoder_[j, i + 1, x] = 1
    return predict


predict = model_predict(
    input_texts,
    max_decoder_seq_length,
    model,
    in_encoder,
    in_decoder
)


def decode(output_seq_, reverse_target_char_index_,):
    decoded = []
    for x in output_seq_:
        if reverse_target_char_index_[x] == "\n":
            break
        else:
            decoded.append(reverse_target_char_index_[x])
    decoded_sentence = "".join(decoded)
    return decoded_sentence



for i in range(len(encoder_input_data)):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    output_seq = predict[i, :].ravel().tolist()
    decoded_sentence = decode(
        output_seq, reverse_target_char_index
    )
    print('-')
    print('Input sentence:', input_texts[i])
    print('Decoded sentence:', decoded_sentence)
"""


class Transliterator:
    def __init__(
        self,
        model,
        input_token_index,
        target_token_index,
        max_encoder_seq_length=22,
        max_decoder_seq_length=21
    ):
        self.model = model
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        self.num_encoder_tokens = len(input_token_index)
        self.num_decoder_tokens = len(target_token_index)
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.reverse_input_char_index = dict(
            (i, char) for char, i in input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())

    def accent(self, text):
        text_a = re.sub(r'à|â|ä|á|ã', "a", text)
        text_i = re.sub(r'ï|î', "i", text_a)
        text_u = re.sub(r'û|ù', "u", text_i)
        text_e = re.sub(r'è|é|ê|ë', "e", text_u)
        text_o = re.sub(r'ô|ö|ó|ó', "o", text_e)
        text_A = re.sub(r'À|Â|Ä|Á|Ã', "A", text_o)
        text_I = re.sub(r'Ï|Î', "I", text_A)
        text_U = re.sub(r'Û|Ù', "U", text_I)
        text_E = re.sub(r'È|É|Ê|Ë', "E", text_U)
        text_O = re.sub(r'Ô|Ö|Ó|Ó', "O", text_E)
        text_c = re.sub(r'ç', "c", text_O)
        text_C = re.sub(r'Ç', "C", text_c)
        text_n = re.sub(r'ñ', "n", text_C)
        text_N = re.sub(r'Ñ', "N", text_n)
        return text_N

    def tokenize_word(self, word):
        if re.findall(r'\d|\.', word):
            return [word]
        tokens = list()
        temp_str = ''
        for char in word:
            idx = self.input_token_index.get(char)
            if (char == ' ' or idx is None) and temp_str:
                tokens.append(temp_str)
                temp_str = ''
            if char == ' ' or idx is None:
                tokens.append(char)
            else:
                temp_str = temp_str + char
        if temp_str:
            tokens.append(temp_str)
        return tokens

    def tokenize_text(self, text):
        tokens = list()
        for word in text.split(' '):
            tokens.extend(self.tokenize_word(word))
        return tokens

    def convert_token2ids_for_encoder(self, token):
        encoder_input_data = np.zeros(
            (
                1,
                self.max_encoder_seq_length,
                self.num_encoder_tokens
            ),
            dtype='float32'
        )
        for t, char in enumerate(token):
            encoder_input_data[0, t, self.input_token_index[char]] = 1.
        encoder_input_data[0, t + 1:, self.input_token_index[' ']] = 1.
        return encoder_input_data

    def init_ids_for_decoder(self):
        decode_input_data = np.zeros(
            (
                1,
                    self.max_decoder_seq_length,
                    self.num_decoder_tokens
                ),
            dtype='float32'
        )
        decode_input_data[:, 0, self.target_token_index["\t"]] = 1
        return decode_input_data

    def predict(self, in_encoder, in_decoder):
        in_decoder_ = np.copy(in_decoder)
        predict = np.zeros(
            (1, self.max_decoder_seq_length),
            dtype='float32'
        )
        for i in range(self.max_decoder_seq_length - 1):
            predict = self.model.predict([in_encoder, in_decoder_])
            predict = predict.argmax(axis=-1)
            predict_ = predict[:, i].ravel().tolist()
            for j, x in enumerate(predict_):
                in_decoder_[j, i + 1, x] = 1
        return predict[0, :].ravel().tolist()

    def decode(self, output_seq):
        decoded = []
        for x in output_seq:
            if self.reverse_target_char_index[x] == "\n":
                break
            else:
                decoded.append(self.reverse_target_char_index[x])
        decoded_token = "".join(decoded)
        return decoded_token.replace('ゥ', '')

    def transliterate(self, text):
        preprocessed_text = self.accent(
            text
        ).lower().replace(
            '’', "'"
        ).replace(
            '´', "'"
        ).replace(
            'no. ', 'no.'
        )
        tokens = self.tokenize_text(preprocessed_text)
        print(text, tokens)

        decoded_tokens = list()
        for token in tokens:
            if re.findall(r'\d|\.', token):
                decoded_tokens.append(token.upper())
            elif len(token) == 1 and not token in self.input_token_index:
                decoded_tokens.append(token)
            else:
                in_encoder = self.convert_token2ids_for_encoder(token)
                in_decoder = self.init_ids_for_decoder()
                output_seq = self.predict(in_encoder, in_decoder)
                decoded_token = self.decode(output_seq)
                decoded_tokens.append(decoded_token)

        return ''.join(decoded_tokens)



model = load_model(
    'katakana_english_converter/keras/h5_objects/cnn_s2s.h5'
)
model.summary()

transliterator = Transliterator(
    model,
    input_token_index,
    target_token_index,
)

test_result = dict()
for text in input_texts:
    decoded_text = transliterator.transliterate(text)
    print(decoded_text)
    test_result[text] = decoded_text

with open('data/test-result.json', 'w') as fp:
    json.dump(test_result, fp, ensure_ascii=False, indent=2)
    