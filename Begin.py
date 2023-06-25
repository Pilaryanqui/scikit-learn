import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./datos/Datos_Agos08_Diciem12.csv')
    x = dt_heart.drop(['INCIDENCIA'], axis=1)
    y = dt_heart['INCIDENCIA']
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)
    
    estimators = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'LinearSVC': LinearSVC(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
        'KNN': KNeighborsClassifier(),
        'DecisionTreeClf': DecisionTreeClassifier(),
        'RandomTreeForest': RandomForestClassifier(random_state=0)
    }
    print('--------------------------') 
    print('Resultados')
    print('--------------------------')   
    for name, estimator in estimators.items():
        bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=50).fit(X_train, y_train)
        bag_predict = bag_class.predict(X_test)
        print('{}: {}'.format(name, accuracy_score(bag_predict, y_test)))
    print('--------------------------')   
    print('Resultados normalizados:')
    print('--------------------------')    
    for name, estimator in estimators.items():
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)
        bag_class_norm = BaggingClassifier(base_estimator=estimator, n_estimators=50).fit(X_train_norm, y_train)
        bag_predict_norm = bag_class_norm.predict(X_test_norm)
        print('{}: {}'.format(name, accuracy_score(bag_predict_norm, y_test)))
    print('--------------------------')      
    print('Resultados discretizados:')
    print('--------------------------')   
    for name, estimator in estimators.items():
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        X_train_disc = discretizer.fit_transform(X_train)
        X_test_disc = discretizer.transform(X_test)
        bag_class_disc = BaggingClassifier(base_estimator=estimator, n_estimators=50).fit(X_train_disc, y_train)
        bag_predict_disc = bag_class_disc.predict(X_test_disc)
        print('{}: {}'.format(name, accuracy_score(bag_predict_disc, y_test)))
