"""
    Copyright (C) 2022 Marco Cominelli <marco.cominelli@unibs.it>
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from random import shuffle
import sys
import h5py
import pickle
import argparse
import numpy as np
from network_utility import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import CSI_dataset

if __name__ != '__main__':
    raise Exception('This module cannot be imported')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset_filename', help='Dataset filename (with relative path)')
parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int, default=100)
parser.add_argument('sample_length', help='Length along the time dimension (width)', type=int, default=300)
parser.add_argument('channels', help='Number of channels', type=int, default=1)
parser.add_argument('num_antennas', help='Number of antennas', type=int)
parser.add_argument('basename', help='basename for the files')
parser.add_argument('activities', help='activities to consider')
args = parser.parse_args()

dataset_filename = args.dataset_filename

basename = args.basename

sample_length = args.sample_length
feature_length = args.feature_length
channels = args.channels
num_antennas = args.num_antennas
input_network = (sample_length, feature_length, channels)
activities = args.activities.split(',')
output_shape = len(activities)

''' Load dataset '''
dataset_train, dataset_val, dataset_test = CSI_dataset.loadDataGenerators(dataset_filename, activities, channels)

''' Create model '''
model_name = f'{basename}_{channels}ants_{args.activities}.h5'
optimiz = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='True')
cb_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
cb_save = tf.keras.callbacks.ModelCheckpoint(model_name, save_freq='epoch', save_best_only=True, monitor='val_sparse_categorical_accuracy')

model = csi_network_inc_res(input_network, output_shape)
model.compile(optimizer=optimiz, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.summary()

results = model.fit(dataset_train, epochs=25, validation_data=dataset_val, callbacks=[cb_save, cb_stop], shuffle=True)

model.save(model_name)

