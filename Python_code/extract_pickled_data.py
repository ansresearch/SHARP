import sys
import h5py
import pickle
import string
import numpy as np

if len(sys.argv) < 4:
    print(f'usage: python {sys.argv[0]} <scene> <bw> <wifi>')
    sys.exit()

scene = sys.argv[1]
bw = sys.argv[2]
wifi = sys.argv[3]


hf = h5py.File(f'datasets/{scene}_doppler_{bw}{wifi}.h5', 'w')

for activity in ['A', 'B', 'C', 'D', 'E', 'F', 'H', 'J', 'K', 'M', 'N', 'O']:
    data = []
    for stream in range(4):
        filename = f'doppler_traces/{bw}{wifi}/{scene}/{scene}_{activity}_stream_{stream}.txt'
        with open(filename, 'rb') as fp:
            new_data = pickle.load(fp)

        new_data = np.expand_dims(new_data, axis=2)

        if len(data) == 0:
            data = new_data
        else:
            data = np.concatenate((data, new_data), axis=2)

    hf.create_dataset(activity, data=data)

    print(data.shape)

hf.close()

