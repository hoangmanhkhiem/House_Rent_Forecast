import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy import stats
import torch.optim as optim

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def Import_Normalize_Data(path):
    data = pd.read_csv(path)
    data['Rent'] = np.log1p(data['Rent'])
    data['Size'] = np.log1p(data['Size'])
    return data


def Plot_Rent(data):
    fig, ax = plt.subplots()
    res = stats.probplot(data['Rent'], plot=ax)
    ax.ticklabel_format(style='plain')
    ax.set_title("Probability Plot - Rent")

    plt.show()


def Plot_Size(data):
    fig, ax = plt.subplots()
    res = stats.probplot(data['Size'], plot=ax)
    ax.ticklabel_format(style='plain')
    ax.set_title("Probability Plot - Size")

    plt.show()


# Label : Rent
# Features : Size, Floor

def Filter(data):
    Size = np.array(data['Size'])
    Rent = np.array(data['Rent'])
    _size = [vl for idx, vl in enumerate(Size)]
    _rent = [vl for idx, vl in enumerate(Rent)]
    data = np.array([_size, _rent])
    label = _rent
    features = np.array(_size)
    data = np.transpose(data)
    return label, features


def main():
    data = Import_Normalize_Data("House_Rent_Dataset.csv")
    y, x = Filter(data)

    feature = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    label = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = LinearRegressionModel()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 4700
    for epoch in range(num_epochs):
        outputs = model(feature)
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    X_ = data['Size']
    _size = [vl for idx, vl in enumerate(X_)]
    X_ = np.array(_size)
    print(X_)
    Y_ = data['Rent']
    _rent = [vl for idx, vl in enumerate(Y_)]
    Y_ = np.array(_rent)
    #
    predicted_prices = model(feature).detach().numpy() * np.std(y) + np.mean(y)
    plt.scatter(X_, Y_, label='Thực tế')
    plt.scatter(x, predicted_prices, color='red', label='Dự đoán')
    plt.xlabel('Diện tích (m2)')
    plt.ylabel('Giá nhà')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
