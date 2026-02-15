import pandas as pd
from AutoMxL.Preprocessing.Deep_Encoder import *
from sklearn.preprocessing import LabelEncoder
from AutoMxL.param_config import batch_size, n_epoch, learning_rate
from AutoMxL.Explore.Features_Type import is_categorical, is_boolean

class CategoricalEncoder(object):

    def __init__(self,
                 method='deep_encoder'
                 ):

        assert method in ['deep_encoder', 'one_hot'], 'invalid method : select deep_encoder / one_hot'

        self.method = method
        self.is_fitted = False
        self.l_var2encode = []
        self.l_var_other = []
        self.target = None
        self.d_embeddings = {}
        self.d_int_encoders = {}
        self.d_metrics = {}

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit(self, df, l_var=None, target=None, verbose=False):
        if self.method == 'deep_encoder':
            assert target is not None, 'fill target parameter to use deep encoder'

        l_cat = [col for col in df.columns.tolist() if
                 (is_categorical(df, col) or is_boolean(df, col)) and col != target]

        if l_var is None:
            self.l_var2encode = l_cat
        else:
            self.l_var2encode = [col for col in l_var if col in l_cat]

        df_local = df.copy()

        self.target = target

        if verbose:
            print(" **method : " + self.method)
            if (self.method == 'deep_encoder') and (len(self.l_var2encode) > 20):
                color_print('  might take a little while, make coffee', 32)
            print("  >", len(self.l_var2encode), "features to encode")
            if len(self.l_var2encode) > 0:
                print(" ", self.l_var2encode)

        if len(self.l_var2encode) > 0:
            if self.method == 'deep_encoder':
                self.d_int_encoders, self.d_embeddings, self.d_metrics = \
                    get_embedded_cat(df_local, self.l_var2encode, target, batch_size, n_epoch, learning_rate,
                                     verbose=False)

        self.is_fitted = True

        if verbose:
            if (self.method == "deep_encoder") and len(self.l_var2encode) > 0:
                print("  NN Loss:", round(self.d_metrics['loss'], 4), "/ Accuracy:",
                      round(self.d_metrics['accuracy'], 4))
                print("  Epoch:", n_epoch, "/ batch:", batch_size, "/ l_rate:", learning_rate)

    """
    ----------------------------------------------------------------------------------------------
    """

    def transform(self, df, verbose=False):
        assert self.is_fitted, 'fit the encoding first using .fit method'

        df_local = df.copy()

        if len(self.l_var2encode) > 0:
            if self.method == 'one_hot':

                return dummy_all_var(df_local, var_list=self.l_var2encode, prefix_list=None, keep=False,
                                     verbose=verbose)

            elif self.method == 'deep_encoder':
                self.l_var_other = [col for col in df_local.columns.tolist() if col not in self.l_var2encode]

                for col in self.l_var2encode:
                    df_local[col] = self.d_int_encoders[col].fit_transform(df_local[col].astype('str'))

                if verbose:
                    print(' Deep Encoder Embedding dim:')

                df_embedded = df_local[self.l_var2encode].copy()

                for col, d_level in self.d_embeddings.items():
                    for i in range(len(d_level[0])):
                        df_embedded[col + '_' + str(i)] = df_embedded[col].replace(
                            {k: v[i] for k, v in d_level.items()})

                    df_embedded = df_embedded.drop(col, axis=1)

                    if verbose:
                        print("  > " + col + ":", len(d_level[0]))

                return pd.concat([df[self.l_var_other], df_embedded], axis=1)

        else:
            print(" No variable to encode")

            return df_local

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit_transform(self, df, l_var=None, target=None, verbose=False):
        df_local = df.copy()
        self.fit(df_local, l_var, target, verbose)
        df_local = self.transform(df_local, verbose)

        return df_local

"""
----------------------------------------------------------------------------------------------
"""

def dummy_all_var(df, var_list=None, prefix_list=None, keep=False, verbose=False):
    df_local = df.copy()

    for col in var_list:
        if prefix_list is None:
            pref = col
        else:
            pref = prefix_list[var_list.index(col)]

        df_cat = pd.get_dummies(df_local[col], prefix=pref, drop_first=True)
        df_local = pd.concat((df_local, df_cat), axis=1)

        if not keep:
            df_local = df_local.drop(col, axis=1)
        if verbose:
            print('  > ' + col + ' ->', df_cat.columns.tolist())

    return df_local

"""
----------------------------------------------------------------------------------------------
"""

def get_embedded_cat(df, var_list, target, batchsize, n_epochs, lr, verbose=False):
    df_local = df[var_list + [target]].copy()

    d_int_encoders = {}
    for cat_col in var_list:
        d_int_encoders[cat_col] = LabelEncoder()
        df_local[cat_col] = d_int_encoders[cat_col].fit_transform(df_local[cat_col].astype('str'))

    d_exp = {col: np.exp(-df_local[col].nunique() * 0.05) for col in var_list}
    d_tmp = {col: np.int(5 * (1 - exp) + 1) for col, exp in d_exp.items()}

    sum_ = sum([1. * np.log(k) for k in d_tmp.values()])

    A, B = 10, 5
    nlayer1 = min(1000, int(A * (len(d_tmp) ** 0.5) * sum_ + 1))
    nlayer2 = int(nlayer1 / B) + 2

    emb_dims = [(df_local[col].nunique(), d_tmp[col]) for col in var_list]

    df_to_encoder = Torch_Dataset(data=df_local, cat_cols=var_list, output_col=target)

    model = Deep_Cat_Encoder(emb_dims, layer_sizes=[nlayer1, nlayer2], output_size=1)

    fit_model, loss, accuracy = train_deep_encoder(df_to_encoder, model=model, optimizer='Adam', criterion='MSE',
                                                   lr=lr, n_epochs=n_epochs, batchsize=batchsize,
                                                   verbose=verbose)

    d_metrics = {'loss': loss, 'accuracy': accuracy}

    i = 0
    d_embeddings = {}
    for param in fit_model.emb_layers.parameters():
        d_embeddings[var_list[i]] = dict(zip(list(range(len(param.data[:, 0]))), param.data.tolist()))
        i += 1

    return d_int_encoders, d_embeddings, d_metrics
