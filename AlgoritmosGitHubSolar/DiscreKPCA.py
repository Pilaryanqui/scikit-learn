import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Cargamos los datos del dataframe de pandas
    dt_heart = pd.read_csv('./datos/Data.csv')

    # Imprimimos un encabezado con los primeros 5 registros
    print(dt_heart.head(5))

    # Guardamos nuestro dataset sin la columna de target
    dt_features = dt_heart.drop(['Toxicos'], axis=1)
    dt_target = dt_heart['Toxicos']

    # DISCRETIZAR
    dt_features = KBinsDiscretizer(strategy='uniform').fit_transform(dt_features)

    # Partimos el conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        dt_features, 
        dt_target, 
        test_size=0.3, 
        random_state=42
    )

    kernel = ['linear', 'poly', 'rbf']

    # Aplicamos la función de kernel de tipo polinomial
    for k in kernel:
        kpca = KernelPCA(n_components=4, kernel=k)
        kpca.fit(X_train)

        dt_train = kpca.transform(X_train)
        dt_test = kpca.transform(X_test)

        logistic = LogisticRegression(solver='lbfgs')
        logistic.fit(dt_train, y_train)

        # Imprimimos los resultados
        print("SCORE KPCA " + k + ":", logistic.score(dt_test, y_test))
