import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA


def standard_preprocessing(data):
    #標準化 Standardization (Z-score Normalization)
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def maxmin_preprocessing(data):
    #最大最小縮放 Min-Max Scaling
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


def correlation_feature_selection(data, r_threshold=0.2):
    #相關分析（只看與output相關的部分）
    correlations = data.corrwith(original.set_index("name")['output'])
    #特徵選擇
    selected_features = correlations[correlations.abs() > r_threshold].index
    data = df[selected_features]
    print(selected_features)

    return data


def decision_tree_feature_selection(data, threshold=.05):
    #建立決策樹
    tree = DecisionTreeClassifier()
    tree.fit(data, original.set_index("name")['output'])
    #從決策樹的信息熵來選擇
    sfm = SelectFromModel(tree, threshold=threshold)
    return sfm.fit_transform(data, original.set_index("name")['output'])


def PCA_preprocessing(data, com_number=1):
    #創建主成份分析
    #請指定主成分個數（須小於特徵數量）
    pca = PCA(n_components=com_number)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    print("Explained variance ratio:", explained_variance)
    print("Cumulative variance ratio:", cumulative_variance)
    return pca.fit_transform(data)


def RFE_feature_selection(data, n_features_to_select=5):
    rfe = RFE(
        estimator=DecisionTreeClassifier(), 
        n_features_to_select=n_features_to_select
    )
    return rfe.fit_transform(data, original.set_index("name")['output'])

def preprocessing_pipeline(
    data:pd.DataFrame, 
    pipeline:list, 
    **kwargs
):

    () = PCA_preprocessing(data, kwargs['com_number'])
    preprocessing_dict = {
        'standardization':standard_preprocessing(data),
        'maxmin':maxmin_preprocessing(data),
        'correlation':correlation_feature_selection(data, r_threshold=kwrags['r_threshold']),
        'decision_tree':decision_tree_feature_selection(data, r_threshold=kwrags['threshold']),
        'PCA':PCA_preprocessing(data, kwargs['com_number']),
        'RFE':RFE_feature_selection(data, n_features_to_select=kwargs['n_features_to_select'])
    }


    for p in pipeline:
        data = preprocessing_dict[p]
    
    return data


df = pd.read_csv("./CNS_output.csv")
df = df.set_index('Name')

original = pd.read_excel("./CNSData(940).xlsx")
data = df.dropna(axis=1)

preprocessing_pipeline(
    data, 
    pipeline=['standardization', 'correlation'], 
    r_threshold=0.3
)








