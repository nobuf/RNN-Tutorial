# -*- coding: utf-8 -*-

import numpy as np
import unicodedata
import codecs
import re

import tensorflow as tf

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
#FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
# reserve the first index for space
#JAPANESE_CHARACTERS = list(u" いろはにほへとちりぬるをわかよたれそつねならむうゐのおくやまけふこえてあさきゆめみしゑひもせすん")
JAPANESE_CHARACTERS = list(u" 要一ゃ措内ょ最安済事再希わんじ外超会 党環予昨カ優0儲4茶方費示談るタ労員ッ必いデ兆えニ半重がハ県金2ご盗策願開討針鼻ミ除だ代面日つとル資置ばン言年表へ任選固企む丁垂や円よ程れ小公排引難気挙続朝府速無加#関遇出循責グ3割張ゴ7営相手建理ダ掲蓄ぇ積か配く迎バこ寒捕ピし取留高野ボたっ発来で部に自業道は共声びヒ極直求後違見合好ー現め省。約率分ら月残上ろ明税抑間誕備検実的財交帰繋訪キ逮ケ進サ制ス伸去下チあ計組投うド午おネ経総過パげ氏受ざ務ず身拡向臣院ュ寧て全なロ6の議互調執ぶ査社国池ま、擁体も者ゆ消り倍賃意頑民を懇対北望中生想両枠産定春曇大度温伝猫衆設障1新課5男先運画人立政更校元久球働今き連問行け当黒さ量敗ブす闘応潟回ちモ玉どリね諮保述歳具べ席%前埼使数")

def chr_japanese(i):
    return JAPANESE_CHARACTERS[i]


def ord_japanese(c):
    return JAPANESE_CHARACTERS.index(c)


def normalize_txt_file(txt_file, remove_apostrophe=True):
    """
    Given a path to a text file, return contents with unsupported characters removed.
    """
    with codecs.open(txt_file, encoding="utf-8") as open_txt_file:
        return normalize_text(open_txt_file.read(), remove_apostrophe=remove_apostrophe)


def normalize_text(original, remove_apostrophe=True):
    """
    Given a Python string ``original``, remove unsupported characters.

    The only supported characters are letters and apostrophes.
    """
    # convert any unicode characters to ASCII equivalent
    # then ignore anything else and decode to a string
    # result = unicodedata.normalize("NFKD", original).encode("ascii", "ignore").decode()
    # if remove_apostrophe:
    #     # remove apostrophes to keep contractions together
    #     result = result.replace("'", "")
    # # return lowercase alphabetic characters and apostrophes (if still present)
    # return re.sub("[^a-zA-Z']+", ' ', result).strip().lower()
    result = original.replace('\r', ' ').replace('\n', ' ')
    return result.strip().lower()


def text_to_char_array(original):
    """
    Given a Python string ``original``, map characters
    to integers and return a numpy array representing the processed string.

    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/text.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """

    # Create list of sentence's words w/spaces replaced by ''
    result = original.replace(' ', '  ')
    result = result.split(' ')

    # Tokenize words into letters adding in SPACE_TOKEN where required
    result = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in result])

    # Return characters mapped into indicies
    return np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord_japanese(xt) for xt in result])


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of ``sequences``.

    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)

    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/text.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape


def sparse_tensor_value_to_texts(value):
    """
    Given a :class:`tf.SparseTensor` ``value``, return an array of Python strings
    representing its values.

    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/text.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """
    return sparse_tuple_to_texts((value.indices, value.values, value.dense_shape))


def sparse_tuple_to_texts(tuple):
    '''
    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/text.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    '''
    indices = tuple[0]
    values = tuple[1]
    results = [''] * tuple[2][0]
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        c = ' ' if c == SPACE_INDEX else chr_japanese(c)
        results[index] = results[index] + c
    # List of strings
    return results


def ndarray_to_text(value):
    '''
    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/text.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    '''
    results = ''
    for i in range(len(value)):
        results += chr_japanese(value[i])
    return results.replace('`', ' ')


def gather_nd(params, indices, shape):
    '''
    # Function aken from https://github.com/tensorflow/tensorflow/issues/206#issuecomment-229678962

    '''
    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x * y, shape[i + 1:], 1) for i in range(0, rank)]
    indices_unpacked = tf.unstack(tf.transpose(indices, [rank - 1] + range(0, rank - 1)))
    flat_indices = sum([a * b for a, b in zip(multipliers, indices_unpacked)])
    return tf.gather(flat_params, flat_indices)


def ctc_label_dense_to_sparse(labels, label_lengths, batch_size):
    '''
    The CTC implementation in TensorFlow needs labels in a sparse representation,
    but sparse data and queues don't mix well, so we store padded tensors in the
    queue and convert to a sparse representation after dequeuing a batch.

    Taken from https://github.com/tensorflow/tensorflow/issues/1742#issuecomment-205291527
    '''

    # The second dimension of labels must be equal to the longest label length in the batch
    correct_shape_assert = tf.assert_equal(tf.shape(labels)[1], tf.reduce_max(label_lengths))
    with tf.control_dependencies([correct_shape_assert]):
        labels = tf.identity(labels)

    label_shape = tf.shape(labels)
    num_batches_tns = tf.stack([label_shape[0]])
    max_num_labels_tns = tf.stack([label_shape[1]])

    def range_less_than(previous_state, current_input):
        return tf.expand_dims(tf.range(label_shape[1]), 0) < current_input

    init = tf.cast(tf.fill(max_num_labels_tns, 0), tf.bool)
    init = tf.expand_dims(init, 0)
    dense_mask = tf.scan(range_less_than, label_lengths, initializer=init, parallel_iterations=1)
    dense_mask = dense_mask[:, 0, :]

    label_array = tf.reshape(tf.tile(tf.range(0, label_shape[1]), num_batches_tns), label_shape)

    label_ind = tf.boolean_mask(label_array, dense_mask)

    batch_array = tf.transpose(tf.reshape(tf.tile(tf.range(0, label_shape[0]), max_num_labels_tns),
                                          tf.reverse(label_shape, [0]))
                               )
    batch_ind = tf.boolean_mask(batch_array, dense_mask)
    batch_label = tf.concat([batch_ind, label_ind], 0)
    indices = tf.transpose(tf.reshape(batch_label, [2, -1]))
    shape = [batch_size, tf.reduce_max(label_lengths)]
    vals_sparse = gather_nd(labels, indices, shape)

    return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape))
