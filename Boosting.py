import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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
    
    print('Resultados originales:')
    boosting_orig = GradientBoostingClassifier(n_estimators=188).fit(X_train, y_train)
    boosting_pred_orig = boosting_orig.predict(X_test)
    accuracy_orig = accuracy_score(boosting_pred_orig, y_test)
    print('Accuracy: {}'.format(accuracy_orig))
    
    print('Resultados con datos normalizados:')
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    boosting_norm = GradientBoostingClassifier(n_estimators=188).fit(X_train_norm, y_train)
    boosting_pred_norm = boosting_norm.predict(X_test_norm)
    accuracy_norm = accuracy_score(boosting_pred_norm, y_test)
    print('Accuracy: {}'.format(accuracy_norm))
    
    print('Resultados con datos discretizados:')
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_train_disc = discretizer.fit_transform(X_train)
    X_test_disc = discretizer.transform(X_test)
    boosting_disc = GradientBoostingClassifier(n_estimators=188).fit(X_train_disc, y_train)
    boosting_pred_disc = boosting_disc.predict(X_test_disc)
    accuracy_disc = accuracy_score(boosting_pred_disc, y_test)
    print('Accuracy: {}'.format(accuracy_disc))

