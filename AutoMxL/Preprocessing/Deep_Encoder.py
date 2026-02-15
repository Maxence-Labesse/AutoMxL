import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from AutoMxL.Utils.Display import color_print

class Torch_Dataset(Dataset):

    def __init__(self, data, cat_cols=None, output_col=None):
        self.nrow = data.shape[0]
        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.nrow, 1))
        self.cat_cols = cat_cols if cat_cols else []
        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.nrow, 1))
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + [output_col]]
        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.nrow, 1))

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def __len__(self):
        return self.nrow

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def __getitem__(self, idx):
        return [self.y[idx], self.cat_X[idx]]

"""
------------------------------------------------------------------------------------------------------
"""

class Deep_Cat_Encoder(nn.Module):

    def __init__(self, l_levels, layer_sizes, output_size, layer_dropout=0.1):
        super(Deep_Cat_Encoder, self).__init__()

        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in l_levels])
        self.emb_layer_size = sum([y for x, y in l_levels])

        self.linear1 = nn.Linear(self.emb_layer_size, layer_sizes[0])
        nn.init.kaiming_uniform_(self.linear1.weight.data)

        self.linear2 = nn.Linear(layer_sizes[0], layer_sizes[-1])
        nn.init.kaiming_uniform_(self.linear2.weight.data)

        self.output_layer = nn.Linear(layer_sizes[-1], output_size)
        nn.init.kaiming_uniform_(self.output_layer.weight.data)

        if layer_dropout:
            self.dropout = nn.Dropout(p=layer_dropout)

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def forward(self, cat_data):

        if self.emb_layer_size > 0:
            x = [emb_layer(cat_data[:, i])
                 for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)

        x = F.relu(self.linear1(x))
        if self.dropout: x = self.dropout(x)

        x = F.relu(self.linear2(x))
        if self.dropout: x = self.dropout(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x

"""
------------------------------------------------------------------------------------------------------
"""

def train_deep_encoder(torch_dataset, model, optimizer, criterion, lr, n_epochs,
                       batchsize, verbose=False):
    assert criterion == 'MSE', 'invalid criterion : select MSE'
    assert optimizer == 'Adam', "invalid optimizer : select 'Adam'"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if criterion == 'MSE':
        criterion = nn.MSELoss()

    dataloader = DataLoader(torch_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    l_loss, l_accuracy = [], []
    ep = 0

    for epoch in range(n_epochs):
        ep += 1
        for y, cat_x in dataloader:
            cat_x = cat_x.to(device)
            y = y.to(device)

            preds = model(cat_x)
            loss = criterion(preds, y)
            output = (preds > 0.5).float()
            accuracy = (output == y).float().sum()
            l_loss.append(loss.item())
            l_accuracy.append(accuracy.item() / y.shape[0])

        if verbose:
            color_print("Epoch " + str(ep))
            print("loss : ", loss.item())
            print("accuracy : ", accuracy.item() / y.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if verbose:
        plt.plot(l_loss)
        plt.title('Loss')
        plt.show()
        plt.plot(l_accuracy)
        plt.title('Accuracy')
        plt.show()

    return model, l_loss[-1], l_accuracy[-1]
