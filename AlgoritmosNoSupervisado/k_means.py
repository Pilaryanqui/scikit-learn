import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
if __name__ == "__main__":
    dataset = pd.read_csv('./datos/candy.csv')
    #print(dataset.head(10))
    X = dataset.drop('competitorname', axis=1)
    #Selecionamos 4 grupos
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print("Total de centros: " , len(kmeans.cluster_centers_))
    print("="*64)
    print(kmeans.predict(X))
    dataset['group'] = kmeans.predict(X)
    print(dataset)
    sns.pairplot(dataset[['sugarpercent','pricepercent','winpercent','group']], hue =
    'group')
    #sns.pairplot(dataset[['temperature','rh','dew_point','wind_speed','gust_speed','wind_direction','group']], hue = 'group')
    plt.show()
    #implementacion_k_means