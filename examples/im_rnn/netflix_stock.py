import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./data/NetflixDataset.csv')
df[["year", "month", "dates"]] = df["Date"].str.split("-", expand = True)
df = df.tail(2500)

# Calculate moving averages
df['MA30'] = df['Close'].rolling(window=30).mean()
df['MA90'] = df['Close'].rolling(window=90).mean()

# Calculate rolling standard deviation (volatility)
df['Volatility'] = df['Close'].rolling(window=20).std()

# Predict Adj Close
fea_name = "Volatility"
# fea_name = "Close"

df = df.tail(2470)
rm_name = ['Open', 'Close', 'High', 'Low', 'Volume','MA30' ,'MA90', 'Volatility', 'year', 'month', 'dates', "Adj Close"]
rm_name.remove(fea_name)
df.drop(columns=rm_name, inplace=True)

scaler = MinMaxScaler(feature_range=(-1, 1))
model=scaler.fit(df[fea_name].values.reshape(-1,1))
df[fea_name]=model.transform(df[fea_name].values.reshape(-1,1))

def load_data(stock, look_back):
    data_raw = stock.values
    data = []
        
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index:index + look_back])

    data = np.array(data)

    test_set_size = int(np.round(0.4*data.shape[0])) # 30% for test
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]


def netflix_dataset(look_back):
    
    x_train, y_train, x_test, y_test = load_data(df[[fea_name]], look_back)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    
    
    print('x_train.shape = ',x_train.shape)
    print('y_train.shape = ',y_train.shape)
    print('x_test.shape = ',x_test.shape)
    print('y_test.shape = ',y_test.shape)

    return x_train, x_test, y_train, y_test
