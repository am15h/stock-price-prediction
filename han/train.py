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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from tensorflow.keras.optimizers import Adam, SGD
from tensorflow import keras

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)

import logging
logging.getLogger('tensorflow').disabled = True


class PlotLosses(keras.callbacks.Callback):
    def __init__(self):
        super(PlotLosses, self).__init__()

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        plt.ylabel('Train and Validation Loss')
        plt.xlabel('Number Of Epochs')

        plt.plot(self.x, self.losses, label = "Train Loss")
        plt.plot(self.x, self.val_losses, label = "Validation Loss")

        plt.legend()
        plt.show();

        plt.savefig('training_curve.png')

        plt.gcf().clear()


def gen(dataset, batch_size):
    cnt = 0

    while True:
        day = []
        day_len = []
        news_len = []
        label = []

        for x in range (batch_size):
            for (days, day_lens, news_lens), labels in dataset:
                cnt += 1
                if(cnt % 500 == 0):
                    print(cnt)
                day.append(days)
                day_len.append(day_lens)
                news_len.append(news_lens)
                label.append(labels)
            
        yield ((day, day_len, news_len), label)


def train(params):

    tf.enable_eager_execution()
    (device, data_format) = ('/gpu:0', 'channels_first')
    if params['no_gpu'] > 0 or not tf.test.is_gpu_available():
        (device, data_format) = ('/cpu:0', 'channels_last')

    print('Using device %s, and data format %s.' % (device, data_format))

    batch_size = params['batch_size']
    print("Loading Dataset...")
    dataset = pickle.load(open(params['data_path'], 'rb'))

    train_set, dev_set, test_set = dataset.get_dataset(batch_size,params['max_date_len'], params['max_news_len'])
    
    model = HAN(dataset.wordvec, params)

#   TODO: Improve how this step is implemented. Takes up time.

    tick = time.time()
    
    train_steps = 0

    days_train = []
    day_lens_train = []
    news_lens_train = []

    for step, ((days, day_lens, news_lens), labels) in enumerate(train_set):
#        logits = model(days, day_lens, news_lens, training=True)
        days_train.append(days)
        day_lens_train.append(day_lens)
        news_lens_train.append(news_lens)
        train_steps += 1

    valid_steps = 0

    days_val = []
    day_lens_val = []
    news_lens_val = []

    for step, ((days, day_lens, news_lens), labels) in enumerate(dev_set):
#        logits = model(days, day_lens, news_lens, training=True)
        days_val.append(days)
        day_lens_val.append(day_lens)
        news_lens_val.append(news_lens)
        valid_steps += 1
    
    tock = time.time()


    """
    days_train = np.array(days_train, dtype=object)
    day_lens_train = np.array(day_lens_train, dtype=object)
    news_lens_train = np.array(news_lens_train, dtype=object)

    days_val = np.array(days_val, dtype=object)
    day_lens_val = np.array(day_lens_val, dtype=object)
    news_lens_val = np.array(news_lens_val, dtype=object)
    
    print(tock - tick)
    print("Training : ")
    print(type(days_train), days_train.shape)
    print(type(day_lens_train), day_lens_train.shape)
    print(type(news_lens_train), news_lens_train.shape)
    
    print("Validation : ")
    print(type(days_val), days_val.shape)
    print(type(day_lens_val), day_lens_val.shape)
    print(type(news_lens_val), news_lens_val.shape)
    """

    print("Train Steps = ", train_steps)
    print("Validation Steps = ", valid_steps)

    if params['optimizer'] == 'adam':
        opt = Adam(lr=params['learning_rate'], decay=params['decay_rate'],beta_1=params['betas'][0], beta_2=params['betas'][1], epsilon=1e-6)
    else:
        opt = SGD(lr=params['learning_rate'], decay=params['decay_rate'], momentum=0.99, nesterov=True)

    model.compile(loss=params['loss_function'], optimizer=opt, metrics=['accuracy'])
   
#    print(dataset.wordvec.shape)

    callbacks = []

    best_acc = keras.callbacks.ModelCheckpoint("best_acc.h5", monitor='val_acc', verbose=0,save_best_only=True, save_weights_only=True, mode='auto', period=1)
    callbacks.append(best_acc)

    best_loss = keras.callbacks.ModelCheckpoint('best_loss.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True,mode='auto', period=1)
    callbacks.append(best_loss)

    all_models = keras.callbacks.ModelCheckpoint('all_models' + '-{epoch:03d}.h5',verbose=0, save_best_only=False, save_weights_only=True, period=5)
    callbacks.append(all_models)

    plot_loss = PlotLosses()
    callbacks.append(plot_loss)

    train_merged = np.stack([days_train, day_lens_train, news_lens_train], axis = 1)
    val_merged = np.stack([days_val, day_lens_val, news_lens_val], axis = 1)

#    logits = model(days_train, day_lens_train, news_lens_train, training=True)

#    hist = model.fit(train_set, steps_per_epoch = (train_steps // batch_size), epochs = params['train_epochs'], verbose = 2, class_weight = dataset.class_weights,callbacks=callbacks, validation_data=dev_set, validation_steps=(valid_steps //batch_size))

#    hist = model.fit(train_merged, steps_per_epoch = (train_steps // batch_size), epochs = params['train_epochs'], verbose = 2, class_weight = dataset.class_weights,callbacks=callbacks, validation_data=val_merged, validation_steps=(valid_steps //batch_size))

    hist = model.fit_generator(gen(train_set, params['batch_size']), steps_per_epoch = (train_steps // batch_size), epochs = params['train_epochs'], verbose = 2,class_weight = dataset.class_weights, callbacks=callbacks,validation_data=gen(dev_set, params['batch_size']), validation_steps=(valid_steps // batch_size))

    model_arch = model.to_json()
    open('model_architecture.json', 'w').write(model_arch)       

    model.save_weights('trained_model.h5')

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

    train(params)
