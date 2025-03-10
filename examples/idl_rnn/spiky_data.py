import numpy as np
import torch
from .utils import set_seed

set_seed(0)

# Spiky region function
def spiky_function(x, scale=5, freq=[2, 23, 78, 100]):
    return scale * sum(np.sin(f * x) for f in freq)

# Non-spiky region function with added noise
def non_spiky_function(x, noise_mean=0, noise_std=0.25):
    return np.sin(x) + np.random.normal(noise_mean, noise_std, size=x.shape)

level = 7
total_data_points = 10000
num_spiky_regions = 20 + level
spiky_region_size = 100 + level
spiky_region_magnitude = level/2

# Generate data
x_values = np.linspace(0, 100, total_data_points)
y_values = np.zeros(total_data_points)

# Assign spiky regions
spiky_regions = torch.randint(0, total_data_points - spiky_region_size, (num_spiky_regions, ))
for start_idx in spiky_regions:
    end_idx = start_idx + spiky_region_size
    x_spiky = x_values[start_idx:end_idx]
    y_values[start_idx:end_idx] = spiky_function(x_spiky, spiky_region_magnitude)

# Assign non-spiky regions
mask = y_values == 0
y_values[mask] = non_spiky_function(x_values[mask], noise_mean=0, noise_std=level*0.05)

# Save and reload data
np.savetxt(f'./data/spiky_data_{level}.csv', y_values, delimiter=',', fmt='%.17g')
y_values = np.loadtxt(f'./data/spiky_data_{level}.csv', delimiter=',')


def load_data(stock, look_back):
    data_raw = stock 
    data = []
    
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index:index + look_back])
    data = np.array(data)

    test_set_size = int(np.round(0.4*data.shape[0])) # 30% for test
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return x_train, y_train, x_test, y_test

def spiky_synthetic_dataset(look_back):
    data = y_values.reshape(-1,1)
    x_train, y_train, x_test, y_test = load_data(data, look_back)
    return x_train, x_test, y_train, y_test