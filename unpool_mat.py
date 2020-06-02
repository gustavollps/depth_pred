import torch
import numpy as np
import pickle

sizes = [[1024, 8, 10], [512, 16, 20], [256, 32, 40], [128, 64, 80]]
# sizes = [[8, 10], [16, 20], [32, 40], [64, 80], [128, 160]]

for id in range(len(sizes)):
    tmp = np.zeros(sizes[id], dtype=np.int32)
    for layer in range(sizes[id][0]):
        id_counter = 0
        for i in range(sizes[id][1]):
            for j in range(sizes[id][2]):
                tmp[layer][i][j] = id_counter
                id_counter += 2
            id_counter += sizes[id][2]*2
    tensor = torch.tensor(tmp, dtype=torch.int64)
    pickle.dump(tensor, open('tensors/' + str(sizes[id][0]) + 'tensor', 'wb'))
