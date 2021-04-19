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
from bert.optimization import AdamWeightDecayOptimizer
import keras

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tfe = tf.contrib.eager


def loss(logits, labels, weights):
	weighted_labels = tf.reduce_sum(
		tf.constant(weights, dtype=tf.float32) * tf.one_hot(labels, 2), axis=1)
	unweighted_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits)
	return tf.reduce_mean(unweighted_losses * weighted_labels)


def compute_accuracy(logits, labels):
	predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
	labels = tf.cast(labels, tf.int64)
	batch_size = int(logits.shape[0])
	return tf.reduce_sum(
		tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def train_step(model, optimizer, dataset, step_counter, ep, class_weights, params, log_interval=None):

	start = time.time()
	steps = 0
	total_loss = 0
	# params
	for step, ((days, prices, day_lens, news_lens), labels) in enumerate(dataset):
		steps += 1
		with tf.GradientTape() as tape:
			logits = model(days, prices, day_lens, news_lens, training=True)
			loss_value = loss(logits, labels, class_weights)
			total_loss += loss_value

		grads = tape.gradient(loss_value, model.trainable_weights)
		grads, _ = clip_ops.clip_by_global_norm(grads, params['clip_norm'])

		optimizer.apply_gradients(zip(grads, model.trainable_weights), global_step=step_counter)

	params['train_losses'].append(total_loss/steps)


def test(model, dataset, class_weights, show_classification_report=False, ds_name='Test'):

	start = time.time()
	avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
	accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)

	y_true = list()
	y_pred = list()
	for (days, prices, day_lens, news_lens), labels in dataset:
		logits = model(days, prices, day_lens, news_lens, training=False)
		avg_loss(loss(logits, labels, class_weights))
		pred = tf.argmax(logits, axis=1, output_type=tf.int64)
		accuracy(pred, tf.cast(labels, tf.int64))

		if show_classification_report:
			y_true.extend(labels.numpy().tolist())
			y_pred.extend(pred.numpy().tolist())
	end = time.time()
	print('%s set: Average loss: %.6f, Accuracy: %.3f%% (%.3f sec)' %
		  (ds_name, avg_loss.result(), 100 * accuracy.result(), end - start))


	if show_classification_report:
		print(classification_report(y_true, y_pred, target_names=['DOWN', 'UP']))

	return accuracy.result(), avg_loss.result()


def train(parameters, train_ds, val_ds, wordvec, class_weights):
	tf.enable_eager_execution()
	tf.logging.set_verbosity(tf.logging.ERROR)

	random_seed.set_random_seed(parameters['seed'])

	(device, data_format) = ('/gpu:0', 'channels_first')
	if parameters['no_gpu'] > 0 or not tf.test.is_gpu_available():
		(device, data_format) = ('/cpu:0', 'channels_last')
	print('Using device %s, and data format %s.' % (device, data_format))

	model = HAN(wordvec, parameters)

	optimizer = AdamWeightDecayOptimizer(
		learning_rate=parameters['learning_rate'],
		weight_decay_rate=0.0,
		beta_1=0.9,
		beta_2=0.999,
		epsilon=1e-6,
		exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

	timestamp = datetime.now().strftime(' %d%m%y %H%M%S')

	# Create and restore checkpoint (if one exists on the path)
	checkpoint_prefix = os.path.join(parameters['model_dir'], 'ckpt')
	step_counter = tf.train.get_or_create_global_step()
	checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)

	best_acc_ep = (0.0, -1, float('inf'))  # acc, epoch, loss
	patience = 0

	with tf.device(device):
		for ep in range(parameters['train_epochs']):
			start = time.time()
			train_step(model, optimizer, train_ds, step_counter, ep, class_weights, parameters, parameters['log_interval'])

			val_acc, val_loss = test(model, val_ds, class_weights, ds_name='Val')

			end = time.time()
			print('\n Epoch: {} \tTime: {:.6f}'.format(ep + 1, end - start))

			parameters['val_losses'].append(val_loss)

			if val_loss.numpy() < best_acc_ep[2]:
				best_acc_ep = (val_acc.numpy(), ep, val_loss.numpy())
				print('Save checkpoint', checkpoint_prefix)
				checkpoint.save(checkpoint_prefix)
#            else:
#                if patience == parameters['patience']:
#                    print('Apply early stopping')
#                    break

#                patience += 1
#                print('patience {}/{}'.format(patience, parameters['patience']))

		print('Min loss {:.6f}, dev acc. {:.3f}%, ep {} \n'.format(
				best_acc_ep[2], best_acc_ep[0] * 100., best_acc_ep[1] + 1))


	model._name = "Hybrid Attention Network"
	model.summary()


	plt.ylabel('Training/Validation Loss')
	plt.xlabel('Number of Epochs')
	plt.plot(parameters['train_losses'], label="Train Loss")
	plt.plot(parameters['val_losses'], label="Validation Loss")
	plt.legend()
	plt.show();
	plt.savefig('han_training_curve.png')
	plt.gcf().clear()

"""
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

	train(params)

"""
