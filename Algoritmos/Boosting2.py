import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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
    
    print('Resultados originales:')
    boosting_orig = GradientBoostingRegressor(n_estimators=188).fit(X_train, y_train)
    boosting_pred_orig = boosting_orig.predict(X_test)
    mse_orig = mean_squared_error(boosting_pred_orig, y_test)
    print('MSE: {}'.format(mse_orig))
    
    print('Resultados con datos normalizados:')
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    boosting_norm = GradientBoostingRegressor(n_estimators=188).fit(X_train_norm, y_train)
    boosting_pred_norm = boosting_norm.predict(X_test_norm)
    mse_norm = mean_squared_error(boosting_pred_norm, y_test)
    print('MSE: {}'.format(mse_norm))
    
    print('Resultados con datos discretizados:')
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_train_disc = discretizer.fit_transform(X_train)
    X_test_disc = discretizer.transform(X_test)
    boosting_disc = GradientBoostingRegressor(n_estimators=188).fit(X_train_disc, y_train)
    boosting_pred_disc = boosting_disc.predict(X_test_disc)
    mse_disc = mean_squared_error(boosting_pred_disc, y_test)
    print('MSE: {}'.format(mse_disc))
