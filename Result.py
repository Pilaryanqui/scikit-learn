import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dt_heart = pd.read_csv('./datos/Datos_Agos08_Diciem12.csv')
    dt_features = dt_heart.drop(['INCIDENCIA'], axis=1)
    dt_target = dt_heart['INCIDENCIA']
    dt_features = StandardScaler().fit_transform(dt_features)
    X_train, X_test, y_train, y_test = train_test_split(
        dt_features, dt_target, test_size=0.30, random_state=42)

    # PCA
    pca = PCA(n_components=3)
    pca.fit(X_train)
    dt_train_pca = pca.transform(X_train)
    dt_test_pca = pca.transform(X_test)
    logistic_pca = LogisticRegression(solver='lbfgs')
    logistic_pca.fit(dt_train_pca, y_train)
    score_pca = logistic_pca.score(dt_test_pca, y_test)

    print("Resultados del PCA:")
    print("Score PCA:", score_pca)

    # IPCA
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    dt_train_ipca = ipca.transform(X_train)
    dt_test_ipca = ipca.transform(X_test)
    logistic_ipca = LogisticRegression(solver='lbfgs')
    logistic_ipca.fit(dt_train_ipca, y_train)
    score_ipca = logistic_ipca.score(dt_test_ipca, y_test)

    print("\nResultados del IPCA:")
    print("Score IPCA:", score_ipca)

    # KPCA
    kernels = ['linear', 'poly', 'rbf']
    scores_kpca = []
    explained_variance_kpca = []
    best_score = 0.0
    best_algorithm = ""
    best_kernel = ""
    for kernel in kernels:
        kpca = KernelPCA(n_components=3, kernel=kernel)
        dt_train_kpca = kpca.fit_transform(X_train)
        dt_test_kpca = kpca.transform(X_test)
        logistic_kpca = LogisticRegression(solver='lbfgs')
        logistic_kpca.fit(dt_train_kpca, y_train)
        score_kpca = logistic_kpca.score(dt_test_kpca, y_test)
        scores_kpca.append(score_kpca)

        # Estimación de varianza explicada
        eigenvalues = kpca.eigenvalues_
        explained_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        explained_variance_kpca.append(explained_variance)

        print("\nResultados de KPCA con kernel", kernel)
        print("Score KPCA:", score_kpca)

        # Actualizar el mejor algoritmo
        if score_kpca > best_score:
            best_score = score_kpca
            best_algorithm = "KPCA"
            best_kernel = kernel

    # Comparación de resultados
    results = pd.DataFrame({'Algoritmo': ['PCA', 'IPCA'] + ['KPCA_' + kernel for kernel in kernels],
                            'Score': [score_pca, score_ipca] + scores_kpca})
    print("\nCuadro comparativo:")
    print(results)

    # Gráfica de varianza explicada
    plt.plot(range(len(pca.explained_variance_ratio_)),
             pca.explained_variance_ratio_, label='PCA')
    plt.plot(range(len(ipca.explained_variance_ratio_)),
             ipca.explained_variance_ratio_, label='IPCA')
    for i, kernel in enumerate(kernels):
        plt.plot(range(len(
            explained_variance_kpca[i])), explained_variance_kpca[i], label='KPCA_' + kernel)

    # Agregar línea para indicar el mejor algoritmo
    plt.axhline(y=best_score, color='r', linestyle='--',
                label=f'Mejor: {best_algorithm} ({best_kernel})')
    plt.annotate(f'Mejor: {best_algorithm} ({best_kernel})', xy=(2, best_score), xytext=(2, best_score-0.05),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componente Principal')
    plt.legend()
    plt.show()
