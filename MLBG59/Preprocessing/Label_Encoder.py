"""
- train_label_encoder : Train a label encoder for categorical features
- Torch_Dataset : Prepare dataset for encoder

https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from MLBG59.Utils.Display import color_print
#
from MLBG59.config import n_epoch, learning_rate, batch_size, crit, optim


def train_label_encoder(torch_dataset, model, criterion=crit, lr=learning_rate, optimizer=optim, n_epochs=n_epoch,
                        batchsize=batch_size, verbose=False):
    """Train label encoder for categorical features

    Parameters
    ----------
    torch_dataset : Torch_Dataset
        Dataset to feed the NN containing categorical features and target
    model : Categorical_Encoder
        encoder
    crit : torch.nn criterion
        model loss to optimize
    optimizer : torch.optim
        NN optimizer
    n_epochs : int (default = 20)
    batchsize : int
        batch size
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
        Categorical_encoder : fitted model
    """
    assert criterion == 'MSE', 'invalid criterion : select lr'
    assert optimizer == 'Adam', "invalid optimizer : select 'Adam'"

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if criterion == 'MSE':
        criterion = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create DataLoader object to feed the model
    dataloader = DataLoader(torch_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    # Training
    l_loss, l_accuracy = [], []
    ep = 0
    for epoch in range(n_epochs):
        ep += 1
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

    if verbose:
        plt.plot(l_loss)
        plt.title('Loss')
        plt.show()
        plt.plot(l_accuracy)
        plt.title('Accuracy')
        plt.show()

    return model, loss, accuracy


"""
------------------------------------------------------------------------------------------------------
"""


class Torch_Dataset(Dataset):

    def __init__(self, data, cat_cols=None, output_col=None):
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
        self.nrow = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.nrow, 1))

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + [output_col]]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.nrow, 1))

        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.nrow, 1))

    def __len__(self):
        """
        Return total number of samples
        """
        return self.nrow

    def __getitem__(self, idx):
        """
        Generates one sample of data
        """
        return [self.y[idx], self.cat_X[idx]]


"""
------------------------------------------------------------------------------------------------------
"""


class Categorical_Encoder(nn.Module):
    """Build categorical encoder

    Parameters
    ----------

    """

    def __init__(self, l_levels, layer_sizes, output_size):
        super(Categorical_Encoder, self).__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in l_levels])

        self.n_embs = sum([y for x, y in l_levels])

        # layer 1
        self.linear1 = nn.Linear(self.n_embs, layer_sizes[0])
        nn.init.kaiming_uniform_(self.linear1.weight.data)

        # layer 2
        self.linear2 = nn.Linear(layer_sizes[0], layer_sizes[-1])
        nn.init.kaiming_uniform_(self.linear2.weight.data)

        # output_layer
        self.output_layer = nn.Linear(layer_sizes[-1], output_size)
        nn.init.kaiming_uniform_(self.output_layer.weight.data)

        # dropout
        self.dropout = nn.Dropout(p=0.1)

        # batchnorm ?

    def forward(self, cat_data):
        """
        Execute the forward propagation fed with categorical data input

        input
        -----
        inputs : Tensor
            input fed in the NN

        return
        ------
        logs_probs : Tensor
            softmax output of the model
        """
        if self.n_embs > 0:
            x = [emb_layer(cat_data[:, i])
                 for i, emb_layer in enumerate(self.emb_layers)]

            x = torch.cat(x, 1)

        # ² x = self.dropout(x)

        x = F.relu(self.linear1(x))
        x = self.dropout(x)

        x = F.relu(self.linear2(x))
        x = self.dropout(x)

        x = self.output_layer(x)

        x = torch.sigmoid(x)

        return x
