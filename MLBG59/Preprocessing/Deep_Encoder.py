""" Label encoder for categorical features :

- Torch_Dataset (class) : Prepare dataset for encoder
- Categorical encoder (class) : Build categorical encoder
- train_label_encoder : Train encoder for categorical features

https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from MLBG59.Utils.Display import color_print


# parameters config
# from MLBG59.config import n_epoch, learning_rate, batch_size, crit, optim


class Torch_Dataset(Dataset):
    """Prepare Dataset for categorical encoder

    - creates batches from the dataset
    - shuffles the data
    - loads the data in parallel

    Parameters
    ----------
    data : DataFrame
        input dataset, containes numerical/categorical/output features
    cat_cols : list of strings
        Names of the categorical columns
    output_col : String
        Name of the output variable
    """

    def __init__(self, data, cat_cols=None, output_col=None):
        # data rows number
        self.nrow = data.shape[0]
        # target
        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.nrow, 1))
        # categorical features
        self.cat_cols = cat_cols if cat_cols else []
        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.nrow, 1))
        # (real) numerical features
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + [output_col]]
        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.nrow, 1))

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def __len__(self):
        """
        Return total number of samples
        """
        return self.nrow

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def __getitem__(self, idx):
        """
        Generates one sample of data
        """
        return [self.y[idx], self.cat_X[idx]]


"""
------------------------------------------------------------------------------------------------------
"""


class Deep_Cat_Encoder(nn.Module):
    """Build categorical encoder

        Parameters
        ----------
        l_levels : list of tuples (int, int)]
            for each col (#unique values, embedding dim)
        layer_sizes : list of int
            encoder layers sizes
        output_size : int
            output dim (binary : 1, multi-class : n)
        """

    def __init__(self, l_levels, layer_sizes, output_size, layer_dropout=0.1):
        super(Deep_Cat_Encoder, self).__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in l_levels])
        self.emb_layer_size = sum([y for x, y in l_levels])

        # layer 1
        self.linear1 = nn.Linear(self.emb_layer_size, layer_sizes[0])
        nn.init.kaiming_uniform_(self.linear1.weight.data)

        # layer 2
        self.linear2 = nn.Linear(layer_sizes[0], layer_sizes[-1])
        nn.init.kaiming_uniform_(self.linear2.weight.data)

        # output_layer
        self.output_layer = nn.Linear(layer_sizes[-1], output_size)
        nn.init.kaiming_uniform_(self.output_layer.weight.data)

        # dropout
        if layer_dropout:
            self.dropout = nn.Dropout(p=layer_dropout)

    """
    -------------------------------------------------------------------------------------------------------------
    """

    def forward(self, cat_data):
        """Execute the forward propagation fed with categorical data input

        Parameters
        ----------
        cat_data : list
            categorical features sample

        Returns
        -------
         model : updated model
        """

        # embedding layer
        if self.emb_layer_size > 0:
            x = [emb_layer(cat_data[:, i])
                 for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)

        # layer 1 (+ dropout)
        x = F.relu(self.linear1(x))
        if self.dropout: x = self.dropout(x)

        # layer 2 (+ dropout)
        x = F.relu(self.linear2(x))
        if self.dropout: x = self.dropout(x)

        # output layer
        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x


"""
------------------------------------------------------------------------------------------------------
"""


def train_deep_encoder(torch_dataset, model, optimizer, criterion, lr, n_epochs,
                       batchsize, verbose=False):
    """Train label encoder for categorical features

    Parameters
    ----------
    torch_dataset : Torch_Dataset
        Dataset to feed the NN containing categorical features and target
    model : Deep_Cat_Encoder
        encoder
    crit : string
        model loss to optimize
    optimizer : string
        NN optimizer
    n_epochs : int (default = 20)
    batchsize : int
        batch size
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
        Categorical_encoder : fitted model
        float : loss
        float : accuracy
    """
    assert criterion == 'MSE', 'invalid criterion : select MSE'
    assert optimizer == 'Adam', "invalid optimizer : select 'Adam'"

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # optimizer and criterion
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if criterion == 'MSE':
        criterion = nn.MSELoss()

    # create DataLoader object to feed the model
    dataloader = DataLoader(torch_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    ####################
    # encoder training #
    ####################
    l_loss, l_accuracy = [], []
    ep = 0

    for epoch in range(n_epochs):
        ep += 1
        # batch training
        for y, cat_x in dataloader:
            # to device
            cat_x = cat_x.to(device)
            y = y.to(device)

            # outputs
            preds = model(cat_x)
            loss = criterion(preds, y)
            output = (preds > 0.5).float()
            accuracy = (output == y).float().sum()
            l_loss.append(loss.item())
            l_accuracy.append(accuracy.item() / y.shape[0])

        if verbose:
            color_print("Epoch " + str(ep))
            print("loss : ", loss.item())
            print("accurdy : ", accuracy.item() / y.shape[0])

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # plot
    if verbose:
        plt.plot(l_loss)
        plt.title('Loss')
        plt.show()
        plt.plot(l_accuracy)
        plt.title('Accuracy')
        plt.show()

    return model, loss, accuracy
