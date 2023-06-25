import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./datos/DataDeber.csv')
    x = dt_heart.drop(['Toxicos'], axis=1)
    y = dt_heart['Toxicos']
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)
    
    estimators = {
        'LinearRegression': LinearRegression(),
        'SVR': SVR(),
        'DecisionTreeReg': DecisionTreeRegressor(),
        'RandomForestReg': RandomForestRegressor(random_state=0)
    }
    
    print('--------------------------') 
    print('Resultados')
    print('--------------------------') 

    for name, estimator in estimators.items():
        bag_reg = BaggingRegressor(base_estimator=estimator, n_estimators=50).fit(X_train, y_train)
        bag_predict = bag_reg.predict(X_test)
        print('{}: {}'.format(name, mean_squared_error(bag_predict, y_test)))
    print('--------------------------')   
    print('Resultados normalizados:')
    print('--------------------------')    
    for name, estimator in estimators.items():
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)
        bag_reg_norm = BaggingRegressor(base_estimator=estimator, n_estimators=50).fit(X_train_norm, y_train)
        bag_predict_norm = bag_reg_norm.predict(X_test_norm)
        print('{}: {}'.format(name, mean_squared_error(bag_predict_norm, y_test)))
    print('--------------------------')      
    print('Resultados discretizados:')
    print('--------------------------')   
    for name, estimator in estimators.items():
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        X_train_disc = discretizer.fit_transform(X_train)
        X_test_disc = discretizer.transform(X_test)
        bag_reg_disc = BaggingRegressor(base_estimator=estimator, n_estimators=50).fit(X_train_disc, y_train)
        bag_predict_disc = bag_reg_disc.predict(X_test_disc)
        print('{}: {}'.format(name, mean_squared_error(bag_predict_disc, y_test)))
