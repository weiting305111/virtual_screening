import pandas as pd
import numpy as np
from rdkit import Chem
from numpy import dot
from numpy.linalg import norm
from sklearn.decomposition import PCA

def check_isna(name):
    return df[name].isna().any()
def check_isame(name):
    return len(df[name].drop_duplicates()) == 1
def check_for_infinity_or_large_values(name):
    return np.isinf(df[name]).any() or (df[name] > np.finfo(np.float64).max).any()

def intersection_cols(df):
    intersection_columns = []
    for row_name in df.columns:

            if (
                (not check_isna(row_name)) and \
                (not check_isame(row_name)) and \
                (not check_for_infinity_or_large_values(row_name))
            ):
                intersection_columns.append(row_name)
    df.loc[:, intersection_columns]
    return df


def add_missing(df, col, nums):
    df_duplictaed = df.copy()
    sample_df = df_duplictaed[col].sample(n=nums).index
    df_duplictaed.loc[sample_df, col] = \
        df_duplictaed.loc[sample_df, col].map(lambda x: np.nan)
    return df_duplictaed


def euclidian_dist(x, y):
    return np.linalg.norm(x - y, axis=0)
def cosin_dist(x, y):
    return dot(x, y)/(norm(x)*norm(y))

class PCA_dist_matrix(PCA):
    def __init__(self, df, threshold):
        super().__init__()
        self.df = df
        self.threshold = threshold
    
    def fit_transform(self, *args, **kwargs):
        pca = PCA(*args, **kwargs)
        pca.fit(self.df)
        df_idx = np.where(pca.explained_variance_ratio_>self.threshold)
        return self.df.iloc[:, df_idx[0]]

class KNN_DataFill:
    def __init__(
        self, 
        df, 
        label,
        set_PCA_dist=False,
        **kwargs
    ):

        self.df = df
        self.label = label
        self.y = self.df[self.label]
        self.x = self.df.drop(columns=self.label)

        if set_PCA_dist==True:
            pca_transformer = PCA_dist_matrix(self.x, kwargs["threshold"])
            self.x = pca_transformer.fit_transform(kwargs["n_components"])
    
    def fit(self, k, distance='euclidian'):

        '''
        Calculate the distance matrix and fit the model with given parameters.

        Parameters:
        k (int): 
        The number of nearest neighbors to consider for clustering.
        This represents the number of neighbors to use for each data point during clustering.
        distance (str, optional): The distance metric to be used for clustering. Defaults to 'euclidean'.

        Returns:
        None
        '''

        self.k = k
        dist_matrix = np.zeros((len(self.x), len(self.x)))

        print('model fitting...')

        from tqdm import tqdm
        for i in tqdm(range(len(self.x))):
            for g in range(len(self.x)):

                if distance == 'euclidian':
                    dist_matrix[i, g] = euclidian_dist(
                        self.x.iloc[i, :].values, 
                        self.x.iloc[g, :].values
                    )
                if distance == 'cosine':
                    dist_matrix[i, g] = cosin_dist(
                        self.x.iloc[i, :].values, 
                        self.x.iloc[g, :].values
                    )
        self.dist_matrix = dist_matrix
    
    def predict(self):

        nan_y = self.y[self.y == np.nan]
        nan_y_idx = nan_y.index
        fill_value_list = []
        for cpd in nan_y_idx:
            fill_value = np.sort(self.dist_matrix[cpd, :])[:self.k].mean()
            fill_value_list.append(fill_value)
        self.df[self.label][self.df[self.label].isna()] = fill_value_list
        self.fill_value = fill_value_list
        return self.df
    
    def evaluation(self):

        def rmse(x_dim, y_dim):
            return np.sqrt((((x_dim - y_dim)**2).sum()).mean())
        return rmse(np.array(self.fill_value), self.y.values)

df = pd.read_csv("./CNS_output.csv")
df = df.set_index('Name')
missing_value_per_col = round(len(df.columns)*0.05)

col_dict= {}
for col in df.columns:
    inter_df = intersection_cols(df.drop(columns=[col]))
    inter_df[col] = df[col]
    add_missing_df = add_missing(
        inter_df, 
        col, 
        missing_value_per_col
    )
    predictor = KNN_DataFill(
        add_missing_df, 
        col, 
        set_PCA_dist=True, 
        threshold=0.6,
        n_components=500
    )
    predictor.fit(k=10)
    filled_df = predictor.predict()
    rmse = predictor.evaluation()
    col_dict[col] = rmse
