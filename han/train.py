import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from datetime import datetime
import numpy as np
import os
import pickle
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import clip_ops
import time
from dataset import SnP500Dataset
from model import HAN

from tensorflow.keras.optimizers import Adam, SGD
import keras

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)

import logging
logging.getLogger('tensorflow').disabled = True

def train(params):

    tf.enable_eager_execution()
    (device, data_format) = ('/cpu:0', 'channels_last')

    batch_size = params['batch_size']
    print("Loading Dataset...")
    dataset = pickle.load(open(params['data_path'], 'rb'))

    train_set, dev_set, test_set = dataset.get_dataset(batch_size,params['max_date_len'], params['max_news_len'])
    
    model = HAN(dataset.wordvec, params)

#   TODO: Improve how this step is implemented. Wastes 60-70 seconds.

    for step, ((days, day_lens, news_lens), labels) in enumerate(train_set):
        logits = model(days, day_lens, news_lens, training=True)

    if params['optimizer'] == 'adam':
        opt = Adam(lr=params['learning_rate'], decay=params['decay_rate'],beta_1=params['betas'][0], beta_2=params['betas'][1], epsilon=1e-6)
    else:
        opt = SGD(lr=params['learning_rate'], decay=params['decay_rate'], momentum=0.99, nesterov=True)

    model.compile(loss=params['loss_function'], optimizer=opt, metrics=['accuracy'])
   
#    print(dataset.wordvec.shape)

    model._name = "Hierarchical Attention Network"
    model.summary()


if __name__ == '__main__':
    params = {}
    params['optimizer'] = 'adam'
    params['decay_rate'] = 0.0
    params['batch_size'] = 32
    params['learning_rate'] = 1e-3
    params['loss_function'] = 'mean_squared_error'
    params['betas'] = [0.9, 0.999]
    params['dr'] = 0.3
    params['l2'] = 1e-6
    params['clip_norm'] = 5.0
    params['hidden_size'] = 50
    params['train_epochs'] = 50
    params['patience'] = 1
    params['log_interval'] = 50
    params['vocab_size'] = 33000
    params['days'] = 5
    params['max_date_len'] = 40
    params['max_news_len'] = 30
    params['seed'] = 2019
    params['data_path'] = 'data/sp500glove.pkl'
    params['no_gpu'] = 0

    train(params)
