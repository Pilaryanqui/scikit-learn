# Importamos las bibliotecas
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # Importamos el dataset del 2017
    dataset = pd.read_csv('./datos/Datos_Agos08_Diciem12.csv')

    # Elegimos los features que vamos a usar
    X = dataset[['Rain', 'Temperature', 'RH', 'DewPoint', 'WindSpeed',
                 'GustSpeed', 'WindDirection', 'PLANTA', 'FRUTO', 'SEVERIDAD (%)']]

    # Definimos nuestro objetivo
    y = dataset[['INCIDENCIA']]

    # Dividimos los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Creamos los modelos y realizamos el ajuste
    modelLinear = LinearRegression().fit(X_train, y_train)
    modelLasso = Lasso(alpha=0.2).fit(X_train, y_train)
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    modelElasticNet = ElasticNet(random_state=0).fit(X_train, y_train)

    # Realizamos las predicciones
    y_predict_linear = modelLinear.predict(X_test)
    y_predict_lasso = modelLasso.predict(X_test)
    y_predict_ridge = modelRidge.predict(X_test)
    y_pred_elastic = modelElasticNet.predict(X_test)

    # Calculamos el error medio cuadrado para cada modelo
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    elastic_loss = mean_squared_error(y_test, y_pred_elastic)

    # Imprimimos los valores de pérdida
    print("Linear Loss: ", linear_loss)
    print("Lasso Loss: ", lasso_loss)
    print("Ridge Loss: ", ridge_loss)
    print("ElasticNet Loss: ", elastic_loss)

    # Imprimimos los coeficientes
    print("Coeficientes linear: ", modelLinear.coef_)
    print("Coeficientes lasso: ", modelLasso.coef_)
    print("Coeficientes ridge: ", modelRidge.coef_)
    print("Coeficientes elastic net: ", modelElasticNet.coef_)

    # Calculamos la exactitud de cada modelo
    score_linear = modelLinear.score(X_test, y_test)
    score_lasso = modelLasso.score(X_test, y_test)
    score_ridge = modelRidge.score(X_test, y_test)
    score_elastic = modelElasticNet.score(X_test, y_test)

    # Imprimimos los puntajes (scores)
    print("Score Lineal: ", score_linear)
    print("Score Lasso: ", score_lasso)
    print("Score Ridge: ", score_ridge)
    print("Score ElasticNet: ", score_elastic)

    # Creamos una gráfica de barras para comparar los puntajes de los modelos
    models = ['Linear', 'Lasso', 'Ridge', 'ElasticNet']
    scores = [score_linear, score_lasso, score_ridge, score_elastic]

    plt.bar(models, scores)
    plt.xlabel('Modelos')
    plt.ylabel('Puntaje (Score)')
    plt.title('Comparación de puntajes entre modelos')

    # Agregamos las cantidades en la parte superior de cada barra
    for i, score in enumerate(scores):
        plt.text(i, score, str(round(score, 2)), ha='center', va='bottom')

    plt.show()

    # Identificamos el mejor modelo
    best_model = models[scores.index(max(scores))]
    print("El mejor modelo es:", best_model)

    # Creamos una gráfica de dispersión para comparar las predicciones de cada modelo
    plt.scatter(y_test, y_predict_linear, label='Linear Regression')
    plt.scatter(y_test, y_predict_lasso, label='Lasso')
    plt.scatter(y_test, y_predict_ridge, label='Ridge')
    plt.scatter(y_test, y_pred_elastic, label='ElasticNet')
    plt.xlabel('Valores reales')
    plt.ylabel('Valores predichos')
    plt.title('Comparación de valores reales vs. valores predichos')
    plt.legend()
    plt.show()
