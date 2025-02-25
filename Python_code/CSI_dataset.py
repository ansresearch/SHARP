"""
    Copyright (C) 2022 Marco Cominelli <marco.cominelli@unibs.it>
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
from locale import windows_locale
import h5py
import string
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
  '''Generate data on-the-fly for training/testing'''

  def __init__(self, data, window_length, batch_size, numAntennas):
    '''
    data[0] -> # activities
    data[1] -> # samples
    data[2] -> Doppler vector size
    data[3] -> # antennas
    '''
    self.data = data
    self.window_length = window_length
    self.batch_size = batch_size

    self.num_activities = int(self.data.shape[0])
    self.num_samples_per_activity = int(self.data.shape[1] - window_length)
    self.num_doppler_bins = int(self.data.shape[2])
    self.num_antennas = min([numAntennas, int(self.data.shape[3])])

    self.batches_per_activity = int(self.num_samples_per_activity / self.batch_size)

    self.output = np.ones((self.num_activities, self.num_samples_per_activity))
    for idx in range(self.num_activities):
      self.output[idx,...] = idx * np.ones(self.num_samples_per_activity)

  def __len__(self):
    return self.num_activities * self.batches_per_activity

  def __getitem__(self, batch_idx):
    activity = int(np.floor_divide(batch_idx, self.batches_per_activity))
    start_idx = int(self.batch_size * np.mod(batch_idx, self.batches_per_activity))
    stop_idx = start_idx + self.batch_size

    x = np.array([
        self.data[activity, idx:idx+self.window_length, ..., 0:self.num_antennas] for idx in range(start_idx, stop_idx)
    ])

    y = self.output[activity, start_idx:stop_idx]

    return x, y


def loadDataGenerators(filename, activities, numAntennas=1, batch_size=10):
  ''' Prepare the data generators for training/testing/validation samples '''
  train_samples_per_activity = 8000
  test_samples_per_activity = 500
  val_samples_per_activity = 500
  window_length = 300


  fp = h5py.File(filename, 'r')
  data = np.array([fp[a] for a in activities])

  print(data.shape)

  offset = 0
  data_train = data[:, offset:offset + train_samples_per_activity + window_length, ...]
  train = DataGenerator(data_train, window_length, batch_size, numAntennas)

  offset += (train_samples_per_activity + window_length)
  data_test = data[:, offset:offset + test_samples_per_activity + window_length, ...]
  test = DataGenerator(data_test, window_length, batch_size, numAntennas)
  
  offset += (test_samples_per_activity + window_length)
  data_val = data[:, offset:offset + val_samples_per_activity + window_length, ...]
  val = DataGenerator(data_val, window_length, batch_size, numAntennas)

  return train, test, val
