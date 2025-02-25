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

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset_filename', help='Dataset filename (with relative path)')
parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int, default=100)
parser.add_argument('sample_length', help='Length along the time dimension (width)', type=int, default=300)
parser.add_argument('channels', help='Number of channels', type=int, default=1)
parser.add_argument('num_antennas', help='Number of antennas', type=int)
parser.add_argument('model_name', help='name of the model')
parser.add_argument('activities', help='activities to consider')
parser.add_argument('outname', help='how to name results')
args = parser.parse_args()

dataset_filename = args.dataset_filename

outname = args.outname

sample_length = args.sample_length
feature_length = args.feature_length
channels = args.channels
num_antennas = args.num_antennas
input_network = (sample_length, feature_length, channels)
activities = args.activities.split(',')
output_shape = len(activities)

model_name = f"{args.model_name}_{channels}ants_{args.activities}.h5"

''' Load dataset '''
dataset_train, dataset_val, dataset_test = CSI_dataset.loadDataGenerators(dataset_filename, activities, channels)

model = tf.keras.models.load_model('models/' + model_name)
model.summary()

''' Evaluate on test set '''
model.evaluate(dataset_test)

predictions = model.predict(dataset_test)

print(predictions.shape)

trainscene = model_name.split('.')[0].split("_")[1]
wifi = model_name.split('.')[0].split("_")[2]

output_fname = f"results_{trainscene}_{wifi}_{outname}_{channels}ants_{args.activities}"

np.savetxt(f'results/{output_fname}.csv', predictions, delimiter=',')

sys.exit()









train_labels_true = np.array(labels_train_selected_expanded)

name_cache_train_test = basename + '_' + str(csi_act) + '_cache_train_test'
dataset_csi_train_test = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                                stream_ant_train, input_network, batch_size,
                                                shuffle=False, cache_file=name_cache_train_test, prefetch=False)
train_prediction_list = model.predict(dataset_csi_train_test,
                                            steps=train_steps_per_epoch)[:train_labels_true.shape[0]]

train_labels_pred = np.argmax(train_prediction_list, axis=1)

conf_matrix_train = confusion_matrix(train_labels_true, train_labels_pred)

# val
val_labels_true = np.array(labels_val_selected_expanded)
val_prediction_list = model.predict(dataset_csi_val, steps=val_steps_per_epoch)[:val_labels_true.shape[0]]

val_labels_pred = np.argmax(val_prediction_list, axis=1)

conf_matrix_val = confusion_matrix(val_labels_true, val_labels_pred)

# test
test_labels_true = np.array(labels_test_selected_expanded)

test_prediction_list = model.predict(dataset_csi_test, steps=test_steps_per_epoch)[
                        :test_labels_true.shape[0]]

test_labels_pred = np.argmax(test_prediction_list, axis=1)

conf_matrix = confusion_matrix(test_labels_true, test_labels_pred)
precision, recall, fscore, _ = precision_recall_fscore_support(test_labels_true,
                                                                test_labels_pred,
                                                                labels=labels_considered)
accuracy = accuracy_score(test_labels_true, test_labels_pred)

# merge antennas test
labels_true_merge = np.array(labels_test_selected)
pred_max_merge = np.zeros_like(labels_test_selected)
for i_lab in range(len(labels_test_selected)):
    pred_antennas = test_prediction_list[i_lab * num_antennas:(i_lab + 1) * num_antennas, :]
    lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

    pred_max_antennas = test_labels_pred[i_lab * num_antennas:(i_lab + 1) * num_antennas]
    lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
    lab_max_merge = -1
    if lab_unique.shape[0] > 1:
        count_argsort = np.flip(np.argsort(count))
        count_sort = count[count_argsort]
        lab_unique_sort = lab_unique[count_argsort]
        if count_sort[0] == count_sort[1] or lab_unique.shape[0] > 2:  # ex aequo between two labels
            lab_max_merge = lab_merge_max
        else:
            lab_max_merge = lab_unique_sort[0]
    else:
        lab_max_merge = lab_unique[0]
    pred_max_merge[i_lab] = lab_max_merge

conf_matrix_max_merge = confusion_matrix(labels_true_merge, pred_max_merge, labels=labels_considered)
precision_max_merge, recall_max_merge, fscore_max_merge, _ = \
    precision_recall_fscore_support(labels_true_merge, pred_max_merge, labels=labels_considered)
accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)

metrics_matrix_dict = {'conf_matrix': conf_matrix,
                        'accuracy_single': accuracy,
                        'precision_single': precision,
                        'recall_single': recall,
                        'fscore_single': fscore,
                        'conf_matrix_max_merge': conf_matrix_max_merge,
                        'accuracy_max_merge': accuracy_max_merge,
                        'precision_max_merge': precision_max_merge,
                        'recall_max_merge': recall_max_merge,
                        'fscore_max_merge': fscore_max_merge}

name_file = './outputs/test_' + str(csi_act) + '_' + subdirs_training + '_band_' + str(bandwidth) + '_subband_' + str(sub_band) + '.txt'
with open(name_file, "wb") as fp:  # Pickling
    pickle.dump(metrics_matrix_dict, fp)

# impact of the number of antennas
one_antenna = [[0], [1], [2], [3]]
two_antennas = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
three_antennas = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
four_antennas = [[0, 1, 2, 3]]
seq_ant_list = [one_antenna, two_antennas, three_antennas, four_antennas]
average_accuracy_change_num_ant = np.zeros((num_antennas,))
average_fscore_change_num_ant = np.zeros((num_antennas,))
labels_true_merge = np.array(labels_test_selected)
for ant_n in range(num_antennas):
    seq_ant = seq_ant_list[ant_n]
    num_seq = len(seq_ant)
    for seq_n in range(num_seq):
        pred_max_merge = np.zeros((len(labels_test_selected),))
        ants_selected = seq_ant[seq_n]
        for i_lab in range(len(labels_test_selected)):
            pred_antennas = test_prediction_list[i_lab * num_antennas:(i_lab + 1) * num_antennas, :]
            pred_antennas = pred_antennas[ants_selected, :]

            lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

            pred_max_antennas = test_labels_pred[i_lab * num_antennas:(i_lab + 1) * num_antennas]
            pred_max_antennas = pred_max_antennas[ants_selected]
            lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
            lab_max_merge = -1
            if lab_unique.shape[0] > 1:
                count_argsort = np.flip(np.argsort(count))
                count_sort = count[count_argsort]
                lab_unique_sort = lab_unique[count_argsort]
                if count_sort[0] == count_sort[1] or lab_unique.shape[0] > ant_n - 1:  # ex aequo between two labels
                    lab_max_merge = lab_merge_max
                else:
                    lab_max_merge = lab_unique_sort[0]
            else:
                lab_max_merge = lab_unique[0]
            pred_max_merge[i_lab] = lab_max_merge

        _, _, fscore_max_merge, _ = precision_recall_fscore_support(labels_true_merge, pred_max_merge,
                                                                    labels=[0, 1, 2, 3, 4])
        accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)

        average_accuracy_change_num_ant[ant_n] += accuracy_max_merge
        average_fscore_change_num_ant[ant_n] += np.mean(fscore_max_merge)

    average_accuracy_change_num_ant[ant_n] = average_accuracy_change_num_ant[ant_n] / num_seq
    average_fscore_change_num_ant[ant_n] = average_fscore_change_num_ant[ant_n] / num_seq

metrics_matrix_dict = {'average_accuracy_change_num_ant': average_accuracy_change_num_ant,
                        'average_fscore_change_num_ant': average_fscore_change_num_ant}

name_file = './outputs/change_number_antennas_test_' + str(csi_act) + '_' + subdirs_training + '_band_' + \
            str(bandwidth) + '_subband_' + str(sub_band) + '.txt'
with open(name_file, "wb") as fp:  # Pickling
    pickle.dump(metrics_matrix_dict, fp)
