from datetime import datetime
import numpy as np
import os
import pickle
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.python.ops import clip_ops
import time
from dataset import SnP500Dataset
from model import HAN
from bert.optimization import AdamWeightDecayOptimizer
import keras

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tfe = tf.contrib.eager

from train import *

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
    params['train_epochs'] = 2
    params['patience'] = 1
    params['log_interval'] = 50
    params['vocab_size'] = 33000
    params['days'] = 5
    params['max_date_len'] = 40
    params['max_news_len'] = 30
    params['seed'] = 2019
    params['data_path'] = 'data/sp500glove.pkl'
    params['no_gpu'] = 0
    params['output_dir'] = 'summaries/'
    params['model_dir'] = 'checkpoints'
    params['train_losses'] = []
    params['val_losses'] = []


    tf.enable_eager_execution()
    tf.logging.set_verbosity(tf.logging.ERROR)

    print('Load dataset..', params['data_path'])
    dataset = pickle.load(open(params['data_path'], 'rb'))
    train_ds, val_ds, test_ds = dataset.get_dataset(
        params['batch_size'], params['max_date_len'], params['max_news_len'])

    train(params, train_ds, val_ds, dataset.wordvec, dataset.class_weights)

    model = HAN(dataset.wordvec, params)
    
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=params['learning_rate'],
        weight_decay_rate=0.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    timestamp = datetime.now().strftime(' %d%m%y %H%M%S')
    
    checkpoint_prefix = os.path.join(params['model_dir'], 'ckpt')
    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)

    latest_checkpoint = tf.train.latest_checkpoint(params['model_dir'])
    print('Load the last checkpoint..', latest_checkpoint)
    checkpoint.restore(latest_checkpoint)

    test_acc, test_loss = test(model, test_ds, dataset.class_weights, show_classification_report=True)
