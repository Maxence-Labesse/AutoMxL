import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MLKit.Utils.Display import *


def get_cat_outliers(df, var_list=None, threshold=0.05, verbose=1) :
    """
    outliers detection for categorical features

    input
    -----
     > df : DataFrame
          dataset
     > var_list : list (Default : None)
          list of the features to analyze
          if None, contains all the categorical features
     > threshold : float (Default : 0.05)
          minimum modality frequency
     > verbose : int (0/1) (Default : 1)
          get more operations information

    return
    ------
     > outlier_dict : dict
          key : feature
          value : list of modalities considered as outliers

    """
    # if var_list = None, get all categorical features
    if var_list is None :
        var_list = df.select_dtypes(include=object).columns
    # else, exclude features from var_list whose type is not categorical
    else :
        var_list = [i for i in var_list if i in  df.select_dtypes(include=object).columns]

    df_bis = df[var_list].copy()

    if verbose>0 :
        print_title1('cat features outliers identification (frequency<'+str(threshold)+')')
        print('features : ',var_list,'\n')

    # initialize output dict
    outlier_dict = {}

    # value count (frequency as number and percent for each modality) for features in var_list
    for col in df_bis.columns :
        # percent
        freq_perc = pd.value_counts(df[col],dropna=False) / len(df[col])
        # number
        freq_nb = pd.value_counts(df[col],dropna=False)

        # if feature contain modalities with frequency < trehshold, store in outlier_dict
        if len(freq_perc.loc[freq_perc<threshold])>0 :
            outlier_dict[col] = freq_perc.loc[freq_perc<threshold].index.tolist()

            if verbose > 0 :
                bold_print("-> "+col)
                print("nb outliers : ",freq_nb.loc[freq_perc<threshold].sum())
                print("modalités outliers : ", outlier_dict[col])

                if verbose>1 :
                    df_bis[col].value_counts(-1).plot.bar()
                    plt.axhline(y=threshold,linewidth=2, color='red')
                    plt.show()

    return  outlier_dict

"""
-------------------------------------------------------------------------------------------------------------------------
"""
def get_num_outliers(df, var_list=None, xstd=3, verbose=0):
    """
    outliers detection for num features
    
    input
    -----
     > df : DataFrame
         dataset
     > var_list : list (Default : None)
         list of the features to analyze
         if None, contains all the num features
     > xstd : int (Default : 3)
         coefficient ... ?
     > verbose : int (0/1) (Default : 1)
          get more operations information
            
    return
    ------
     > outlier_dict : dict
         key : feature
         value : index of outliers
    """       
    # if var_list = None, get all num features
    if var_list is None :
        var_list = df._get_numeric_data().columns
    # else, exclude features from var_list whose type is not num
    else :
        var_list = [i for i in var_list if i in  df._get_numeric_data().columns]
    
    df_bis = df[var_list].copy()

    if verbose>0 :
        print_title1('Identification des outliers des features numériques')
        print('features : ',var_list,'\n')
    
    # initialize output dict
    outlier_dict = {}
        
    # compute features upper and lower limit (deviation from the mean > x*std dev (x=3 by default))
    data_std = np.std(df_bis)
    data_mean = np.mean(df_bis)
    anomaly_cut_off = data_std * xstd
    lower_limit  = data_mean - anomaly_cut_off 
    upper_limit = data_mean + anomaly_cut_off
    
    
    df_outliers = pd.DataFrame()
    
    # mask (1 if outlier, else 0)
    for col in df_bis.columns :
        df_outliers[col] = np.where((df_bis[col]<lower_limit[col]) | (df_bis[col]>upper_limit[col]),1,0)
    
    # for features containing outliers
    for col in df_outliers.sum().loc[df_outliers.sum()>0].index.tolist() :
        # store features and outliers index in outlierèdict
        outlier_dict[col] = [lower_limit[col],upper_limit[col]]
    
        if verbose>0 :
            bold_print("-> "+col)
            print('mean : ',round(data_mean[col],2), " / variance : ",round(data_std[col],2))
            print('Nb outliers : ',df_outliers.sum()[col],' (si >',round(upper_limit[col],2),' ou <',round(lower_limit[col],2),')')
            
            # display scatter plots
            if verbose>1 :
                s = df_bis[col]
                supper = s.loc[s>upper_limit[col]]
                slower = s.loc[s<lower_limit[col]]
                smiddle = s.loc[(s >= lower_limit[col]) & (s<=upper_limit[col])]

                fig, ax = plt.subplots()
                ax.scatter(smiddle.index, smiddle)
                ax.scatter(supper.index, supper)
                ax.scatter(slower.index, slower)
                plt.show()

    return outlier_dict
