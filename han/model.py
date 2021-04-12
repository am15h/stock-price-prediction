import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Embedding, Dropout, Dense, GRU, CuDNNGRU, Bidirectional
import tensorflow.nn as nn


class HAN(tf.keras.Model):
    def __init__(self, wordvec, params):
        
        super(HAN, self).__init__()
        self.params = params

        # wordvec: ndarray
        vocab_size = wordvec.shape[0]
        self.embedding_dim = wordvec.shape[1]
        self.embedding =  Embedding(vocab_size, self.embedding_dim, weights=[wordvec],mask_zero=True,trainable=True,)

        self.dropout =  Dropout(rate=self.params['dr'])  # StockNet

        # Word-level attention
        self.word_att =  Dense(1, activation=nn.sigmoid)

        # News-level attention
        self.news_att =  Dense(1, activation=nn.sigmoid)

        # Sequence modeling
        self.bi_gru = self.make_bi_gru(self.embedding_dim)

        # Temporal attention
        self.temp_att =  Dense(1, activation=nn.sigmoid)

        # Discriminative Network (MLP)
        self.fc0 =  Dense(self.params['hidden_size'], activation = nn.elu)
        self.fc1 =  Dense(self.params['hidden_size'], activation = nn.elu)

        self.fc_out =  Dense(2)

    def call(self, x, day_len, news_len, training=False):

        max_dlen = tf.keras.backend.max(day_len).numpy()
        max_nlen = tf.keras.backend.max(news_len).numpy()
        
        max_dlen = np.max(day_len)
        max_nlen = np.max(news_len)

        x = x[:, :, :max_dlen, :max_nlen]
        news_len = news_len[:, :, :max_dlen]

        # Averaged daily news corpus
        # (batch_size, days, max_daily_news, max_news_words
        # -> (batch_size, days, max_daily_news, max_news_words, embedding_dim)
        x = self.embedding(x)

        # handle variable-length news word sequences
        mask = tf.sequence_mask(news_len, maxlen=max_nlen, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=4)
        x *= mask

        # Word-level attention
        # x: (batch_size, days, max_daily_news, max_news_words, embedding_dim)
        # t: (batch_size, days, max_daily_news, max_news_words, 1)
        # n: (batch_size, days, max_daily_news, embedding_dim)
        word_att = self.word_att(x)
        n = nn.softmax(word_att, axis=3) * x
        n = tf.reduce_sum(n, axis=3)

        # handle variable-length day news sequences
        mask = tf.sequence_mask(day_len, maxlen=max_dlen, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=3)
        n *= mask

        # News-level attention
        news_att = self.news_att(n)
        d = nn.softmax(news_att, axis=2) * n
        d = tf.reduce_sum(d, axis=2)

        # Sequence modeling
        gru = self.bi_gru(d, training=training)

        # Temporal attention
        temp_att = self.temp_att(gru)
        v = nn.softmax(temp_att, axis=2) * gru
        v = tf.reduce_sum(v, axis=1)

        # Discriminative Network (MLP)
        v = self.fc0(v)
        v = self.dropout(v) if training else v
        v = self.fc1(v)
        v = self.dropout(v) if training else v
        return self.fc_out(v)

    def make_bi_gru(self, units):
        if tf.test.is_gpu_available() and not self.params['no_gpu']:
            return  Bidirectional(CuDNNGRU(units, return_sequences=True), merge_mode='concat')
        else:
            return  Bidirectional(GRU(units, return_sequences=True), merge_mode='concat')
