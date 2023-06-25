import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import DataConversionWarning

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", message="Feature 9 is constant and will be replaced with 0.")
    warnings.filterwarnings("ignore", message="Feature 23 is constant and will be replaced with 0.")
    warnings.filterwarnings("ignore", message="Feature 43 is constant and will be replaced with 0.")
    warnings.filterwarnings("ignore", message="Feature 45 is constant and will be replaced with 0.")

    dt_heart = pd.read_csv('./datos/DataDeber.csv')

    #print("Datos originales:")
    #print(dt_heart.head(7))

    dt_features = dt_heart.drop(['Toxicos'], axis=1)
    dt_target = dt_heart['Toxicos'].apply(lambda x: 1 if x >= 0.5 else 0)  # Discretización de las etiquetas
    dt_features = StandardScaler().fit_transform(dt_features)
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.30, random_state=42)

    print("-----------------------------")
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

    #print("Datos normalizados:")
    
    #print(X_train[:7])

    print("-----------------------------")
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

   # print("Datos discretizados:")
    #print(X_train[:7])
    print("-----------------------------")
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
