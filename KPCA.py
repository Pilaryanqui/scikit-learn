import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Cargamos los datos del dataframe de pandas
    dt_heart = pd.read_csv('./datos/Datos_Agos08_Diciem12.csv')
    # Imprimimos un encabezado con los primeros 5 registros
    print(dt_heart.head(5))
    # Guardamos nuestro dataset sin la columna de target
    dt_features = dt_heart.drop(['INCIDENCIA'], axis=1)
    # Este será nuestro dataset, pero sin la columna
    dt_target = dt_heart['INCIDENCIA']
    # Normalizamos los datos
    dt_features_normalized = StandardScaler().fit_transform(dt_features)
    # Partimos el conjunto de entrenamiento y para añadir replicabilidad usamos el random state
    X_train, X_test, y_train, y_test = train_test_split(
        dt_features_normalized, dt_target, test_size=0.3, random_state=42)
    kernel = ['linear', 'poly', 'rbf']

    print("Resultado Original:")
    # Aplicamos la función de kernel de tipo polinomial
    for k in kernel:
        
        kpca = KernelPCA(n_components=4, kernel=k)
        # Ajustamos los datos
        kpca.fit(X_train)
        # Aplicamos el algoritmo a nuestros datos de prueba y entrenamiento
        dt_train = kpca.transform(X_train)
        dt_test = kpca.transform(X_test)
        # Aplicamos la regresión logística una vez que reducimos su dimensionalidad
        logistic = LogisticRegression(solver='lbfgs')
        # Entrenamos los datos
        logistic.fit(dt_train, y_train)
        # Imprimimos los resultados
        print("SCORE KPCA " + k + ":", logistic.score(dt_test, y_test))

    print()
    print("Resultado Normalizado:")
    for k in kernel:
        # Utilizamos los datos normalizados en lugar de los originales
        kpca_normalized = KernelPCA(n_components=4, kernel=k)
        kpca_normalized.fit(dt_features_normalized)
        dt_train_normalized = kpca_normalized.transform(X_train)
        dt_test_normalized = kpca_normalized.transform(X_test)
        logistic.fit(dt_train_normalized, y_train)
        print("SCORE KPCA " + k + ":", logistic.score(dt_test_normalized, y_test))

    print()
    print("Resultado Discretizado:")
    for k in kernel:
        
        # Utilizamos los datos discretizados en lugar de los originales
        # (Aquí debes agregar tu código para discretizar los datos según tus necesidades)
        dt_train_discretized = dt_train
        dt_test_discretized = dt_test
        logistic.fit(dt_train_discretized, y_train)
        print("SCORE KPCA " + k + ":", logistic.score(dt_test_discretized, y_test))
