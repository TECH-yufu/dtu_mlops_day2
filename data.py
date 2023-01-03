import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def mnist():
    path = r"C:\Users\Yucheng\OneDrive - Danmarks Tekniske Universitet\DTU\Menneskeorienteret kunstig intelligens\7. semester\Machine learning operations\dtu_mlops\data\corruptmnist"
    
    # exchange with the corrupted mnist dataset
    d1 = np.load(os.path.join(path, 'train_0.npz'))

    train = {'images': d1['images'], 'labels': d1['labels']}

    for i in range(1,5):
        d2 = np.load(os.path.join(path, 'train_{}.npz'.format(i)))
        train['images'] = np.concatenate((train['images'], d2['images']))
        train['labels'] = np.concatenate((train['labels'], d2['labels']))


    test_ = np.load(os.path.join(path, 'test.npz'))

    test = {'images': test_['images'], 'labels': test_['labels']}
    
    return train, test

# train, test = mnist()

# images = train['images']
# labels = train['labels']

# plt.imshow(images[4,:,:], cmap='gray')
# plt.show()