from collections import Counter
import os

import numpy as np
import pandas as pd


def pad_1d(array, max_len):
    array = array[:max_len]
    length = len(array)
    padded = array + [0]*(max_len - len(array))
    return padded, length


def make_word_idx(product_names):
    words = [word for name in product_names for word in name.split()]
    word_counts = Counter(words)

    max_id = 1
    word_idx = {}
    for word, count in word_counts.items():
        if count < 10:
            word_idx[word] = 0
        else:
            word_idx[word] = max_id
            max_id += 1

    return word_idx


def encode_text(text, word_idx):
    return ' '.join([str(word_idx[i]) for i in text.split()]) if text else '0'


if __name__ == '__main__':
    product_data = pd.read_csv('../../data/processed/product_data.csv')
    product_data['product_name'] = product_data['product_name'].map(lambda x: x.lower())

    product_df = pd.read_csv('../../data/raw/products.csv')
    product_df['product_name'] = product_df['product_name'].map(lambda x: x.lower())

    word_idx = make_word_idx(product_df['product_name'].tolist())
    product_data['product_name_encoded'] = product_data['product_name'].map(lambda x: encode_text(x, word_idx))

    num_rows = len(product_data)

    user_id = np.zeros(shape=[num_rows], dtype=np.int32)
    product_id = np.zeros(shape=[num_rows], dtype=np.int32)
    aisle_id = np.zeros(shape=[num_rows], dtype=np.int16)
    department_id = np.zeros(shape=[num_rows], dtype=np.int8)
    eval_set = np.zeros(shape=[num_rows], dtype='S5')
    label = np.zeros(shape=[num_rows], dtype=np.int8)

    is_ordered_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    index_in_order_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_dow_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_hour_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    days_since_prior_order_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_size_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    reorder_size_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_number_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    product_name = np.zeros(shape=[num_rows, 30], dtype=np.int32)
    product_name_length = np.zeros(shape=[num_rows], dtype=np.int8)
    history_length = np.zeros(shape=[num_rows], dtype=np.int8)

    