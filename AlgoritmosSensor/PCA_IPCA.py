import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dt_heart = pd.read_csv('./datos/Datos_Agos08_Diciem12.csv')
    dt_features = dt_heart.drop(['INCIDENCIA'], axis=1)
    dt_target = dt_heart['INCIDENCIA']

    # Eliminar características constantes
    dt_features = dt_features.loc[:, dt_features.nunique() != 1]

    # Discretizar los datos
    #discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    #dt_features = discretizer.fit_transform(dt_features)

    # Normalizar los datos
    #dt_features = StandardScaler().fit_transform(dt_features)

    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.30, random_state=42)

    pca = PCA(n_components=3)
    pca.fit(X_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)

    logistic = LogisticRegression(solver='lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE PCA:", logistic.score(dt_test, y_test))

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE IPCA:", logistic.score(dt_test, y_test))
