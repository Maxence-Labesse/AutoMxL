from MLBG59.Explore.Features_Type import is_categorical, is_boolean

""" Categorical features processing

 - CategoricalEncoder (class) : Encode categorical features
 - dummy_all_var (func) : get one hot encoded vector for each category of a categorical features list
 - get_embedded_cat (func) : get embedding representation with NN
 - mca (func) : TODO

"""
import pandas as pd
from MLBG59.Preprocessing.Deep_Encoder import *
from sklearn.preprocessing import LabelEncoder
from MLBG59.param_config import batch_size, n_epoch, learning_rate


class CategoricalEncoder(object):
    """Encode categorical features

    Available methods :

    - one hot encoding
    - Build and train a Neural Network for the creation of embeddings for categorical variables.
    Default model parameters are stored in param_config.py file

    Parameters
    ----------
    method : string (Default : deep_encoder)
        method used to get categorical encoding
        Available methods : "one_hot", "deep_encoder"
    """

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

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit(self, df, l_var=None, target=None, verbose=False):
        """ Fit encoder on dataset following method

        Parameters
        ----------
        df : DataFrame
            input dataset
        l_var : list (Default None)
            names of the variables to encode.
            If None, all the categorical and boolean features
        target : string (Default None)
            name of the target for deep_encoder method
        verbose : boolean (Default False)
            Get logging information
        """
        # features to encode
        l_cat = [col for col in df.columns.tolist() if
                 (is_categorical(df, col) or is_boolean(df, col)) and col != target]

        if l_var is None:
            self.l_var2encode = l_cat
        else:
            self.l_var2encode = [col for col in l_var if col in l_cat]

        df_local = df.copy()

        # features not to encode
        self.l_var_other = [col for col in df_local.columns.tolist() if col not in self.l_var2encode]

        # target
        self.target = target

        # one hot encoding method
        if self.method == 'one_hot':
            if len(self.l_var2encode) > 0:
                self.is_fitted = True

        # deep learning embedded representation method
        elif self.method == 'deep_encoder':
            self.d_int_encoders, self.d_embeddings = \
                get_embedded_cat(df_local, self.l_var2encode, target, batch_size, n_epoch, learning_rate,
                                 verbose=verbose)
            self.is_fitted = True

        return self

    """
    ----------------------------------------------------------------------------------------------
    """

    def transform(self, df, verbose=False):
        """ transform dataset categorical features using the encoder.
        Can be done only if encoder has been fitted

        Parameters
        ----------
        df : DataFrame
            dataset to transform
        verbose : boolean (Default False)
            Get logging information
        """
        assert self.is_fitted, 'fit the encoding first using .fit method'

        df_local = df.copy()

        if self.method == 'one_hot':
            return dummy_all_var(df_local, var_list=self.l_var2encode, prefix_list=None, keep=False, verbose=verbose)

        elif self.method == 'deep_encoder':
            for col in self.l_var2encode:
                df_local[col] = self.d_int_encoders[col].fit_transform(df_local[col])

            # int labes to embedding
            df_embedded = df_local[self.l_var2encode].copy()
            for col, d_level in self.d_embeddings.items():
                for i in range(len(d_level[0])):
                    # print(d_level)
                    df_embedded[col + '_' + str(i)] = df_embedded[col].replace({k: v[i] for k, v in d_level.items()})
                    # (df_embedded[[col, col + '_' + str(i)]].sample(20))
                df_embedded = df_embedded.drop(col, axis=1)

        return pd.concat([df[self.l_var_other], df_embedded], axis=1)

    """
    ----------------------------------------------------------------------------------------------
    """

    def fit_transform(self, df, l_var, target, verbose=False):
        """fit and transform dataset categorical features

        Parameters
        ----------
        df : DataFrame
            input dataset
        l_var : list (Default None)
            names of the variables to encode.
            If None, all the categorical and boolean features
        target : string (Default None)
            name of the target for deep_encoder method
        verbose : boolean (Default False)
            Get logging information
        """
        self.fit(df, l_var, target, verbose)
        df = self.transform(df, verbose)

        return df


"""
----------------------------------------------------------------------------------------------
"""


def dummy_all_var(df, var_list=None, prefix_list=None, keep=False, verbose=False):
    """Get one hot encoded vector for selected/all categorical features

    Parameters
    ----------
     df : DatraFrame
        Input dataset
     var_list : list (Default : None)
        Names of the features to dummify
        If None, all the num features
     prefix_list : list (default : None)
        Prefix to add before new features name (prefix+'_'+cat).
        It None, prefix=variable name
     keep : boolean (Default = False)
        If True, delete the original feature
     verbose : boolean (Default False)
        Get logging information

    Returns
    -------
    DataFrame
          Modified dataset
    """
    df_local = df.copy()

    for col in var_list:
        # if prefix_list == None, add column name as prefix, else add prefix_list
        if prefix_list is None:
            pref = col
        else:
            pref = prefix_list[var_list.index(col)]

        # dummify
        df_cat = pd.get_dummies(df_local[col], prefix=pref)
        # concat source DataFrame and new features
        df_local = pd.concat((df_local, df_cat), axis=1)

        # if keep = False, remove original features
        if not keep:
            df_local = df_local.drop(col, axis=1)
        if verbose:
            print('  > ' + col + ' ->', df_cat.columns.tolist())

    return df_local


"""
----------------------------------------------------------------------------------------------
"""


def get_embedded_cat(df, var_list, target, batchsize, n_epochs, lr, verbose=False):
    """Get embedded representation for categorical features using NN encoder

    Parameters
    ----------
    df : DataFrame
        input Dataset
    var_list : list of strings
        features names
    target : string
        target name
    batchsize : int
        batch size for encoder training
    n_epochs : int
        number of epoch for encoder training
    lr : float
        encoder learning rate
    verbose : boolean (Default False)
        Get logging information

    Returns
    -------
        DataFrame : modified dataset
    """
    ######################
    # Get list to encode #
    ######################
    df_local = df[var_list + [target]].copy()

    ############################
    # Categories to int labels #
    ############################
    d_int_encoders = {}
    for cat_col in var_list:
        d_int_encoders[cat_col] = LabelEncoder()
        df_local[cat_col] = d_int_encoders[cat_col].fit_transform(df_local[cat_col])

    ###################
    # Get layer sizes #
    ###################
    d_exp = {col: np.exp(-df_local[col].nunique() * 0.05) for col in var_list}
    d_tmp = {col: np.int(5 * (1 - exp) + 1) for col, exp in d_exp.items()}

    sum_ = sum([1. * np.log(k) for k in d_tmp.values()])

    A, B = 10, 5
    nlayer1 = min(1000, int(A * (len(d_tmp) ** 0.5) * sum_ + 1))
    nlayer2 = int(nlayer1 / B) + 2

    emb_dims = [(df_local[col].nunique(), d_tmp[col]) for col in var_list]

    #####################
    # Train the encoder #
    #####################
    # Create Torch_Dataset
    df_to_encoder = Torch_Dataset(data=df_local, cat_cols=var_list, output_col=target)

    model = Deep_Cat_Encoder(emb_dims, layer_sizes=[nlayer1, nlayer2], output_size=1)

    fit_model, loss, accuracy = train_deep_encoder(df_to_encoder, model=model, optimizer='Adam', criterion='MSE',
                                                   lr=lr, n_epochs=n_epochs, batchsize=batchsize,
                                                   verbose=verbose)

    ############################################
    # Store embedding and get output DataFrame #
    ############################################
    i = 0
    d_embeddings = {}
    for param in fit_model.emb_layers.parameters():
        d_embeddings[var_list[i]] = dict(zip(list(range(len(param.data[:, 0]))), param.data.tolist()))
        i += 1

    return d_int_encoders, d_embeddings


"""
----------------------------------------------------------------------------------------------
"""

"""
def mca(df, var_list=None, sample_size=100000, n_iter=30, verbose=False):
     Why MCA ?

    Parameters
    ----------
    df : Dataframe
        input dataset
    var_list : list (Default : None)
        Names of the features to dummify
        If None, all the cat features
    sample_size : int (Default : 100 000)
        size of the sample to fit the mca
        if None, all the dataset.
    n_iter : int (Default 30)
        number of iterations
     verbose : boolean (Default False)
        Get logging information

    Returns
    -------
        !
    
    #
    # if var_list = None, get all categorical features
    # else, exclude features from var_list whose type is not categorical
    l_cat = [col for col in df.columns.tolist() if df[col].dtype == 'object']

    if var_list is None:
        var_list = l_cat
    else:
        var_list = [col for col in var_list if col in l_cat]

    l_other = [col for col in df.columns.tolist() if col not in var_list]

    # dataset to apply mca
    df_to_mca = df[var_list].copy()

    # compute max components
    max_components = np.sum([df_to_mca[col].nunique() - 1 for col in df_to_mca.columns])

    # get variables dummies to feed mca algo
    #df_to_acm_dum = dummy_all_var(df_to_mca, var_list=None, prefix_list=None, keep=False, verbose=False)

    # create sample to speed up mca fitting
    sample_size = None
    if sample_size is not None:
        df_sample = df_to_mca.sample(min(sample_size, df_to_mca.shape[0])).copy()
    else:
        df_sample = df_to_mca

    print(df_sample)

    # init mca object
    mca = prince.MCA(n_components=max_components, n_iter=n_iter)

    # fit mca
    mca.fit(df_sample)

    explained_variance_ratio_ = mca.explained_inertia_ / np.sum(mca.explained_inertia_)

    # transform whole dataset
    df_transf = mca.transform(df_to_mca)

    # find argmin to get 90% of variance
    n_dim = np.argwhere(np.cumsum(explained_variance_ratio_) > 0.90)[0][0]

    if len(l_other) > 0:
        df_mca = pd.concat((df[l_other], df_transf.iloc[:, :n_dim + 1]), axis=1)
    else:
        df_mca = df_transf.iloc[:, :n_dim + 1]

    if verbose:
        print("Numerical Dimensions reduction : " + str(max_components) + " - > " + str(n_dim + 1))
        print("explained inertia : " + str(round(np.cumsum(explained_variance_ratio_)[n_dim], 4)))
        print(df_mca.shape)

    return df_mca, explained_variance_ratio_


df = pd.read_csv('..\..\data\\bank-additional-full.csv', sep=";")

df_mca, inertia = mca(df, var_list=None, sample_size=100000, n_iter=30, verbose=True)
"""
