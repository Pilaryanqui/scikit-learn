import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dt_heart = pd.read_csv('./datos/Datos_Agos08_Diciem12.csv')
    print("Datos originales:")
    print(dt_heart.head(7))

    dt_features = dt_heart.drop(['INCIDENCIA'], axis=1)
    dt_target = dt_heart['INCIDENCIA']
    dt_features = StandardScaler().fit_transform(dt_features)
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.30, random_state=42)

    print("Resultados originales:")
    pca = PCA(n_components=3)
    pca.fit(X_train)
    logistic = LogisticRegression(solver='lbfgs')
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE PCA: ", logistic.score(dt_test, y_test))

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))

    print("Datos normalizados:")
    print(X_train[:7])

    print("Resultados con datos normalizados:")
    pca = PCA(n_components=3)
    pca.fit(X_train)
    logistic = LogisticRegression(solver='lbfgs')
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE PCA: ", logistic.score(dt_test, y_test))

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))

    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    dt_features_discretized = discretizer.fit_transform(dt_features)

    X_train, X_test, y_train, y_test = train_test_split(dt_features_discretized, dt_target, test_size=0.30, random_state=42)

    print("Datos discretizados:")
    print(X_train[:7])

    print("Resultados con datos discretizados:")
    pca = PCA(n_components=3)
    pca.fit(X_train)
    logistic = LogisticRegression(solver='lbfgs')
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE PCA: ", logistic.score(dt_test, y_test))

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))

    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componente Principal')
    plt.show()
