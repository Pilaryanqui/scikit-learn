import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer

if __name__ == "__main__":
    dataset = pd.read_csv('./datos/DataDeber.csv')
    print(dataset.head(5))
    X = dataset[['Total commits', 'Total commits per day', 'Accumulated commits', 'Committers', 'Committers Weight', 'classes', 'ncloc',
            'functions', 'duplicated_lines', 'test_errors', 'skipped_tests', 'coverage', 'complexity', 'comment_lines',
            'comment_lines_density', 'duplicated_lines_density', 'files', 'directories', 'file_complexity', 'violations',
            'duplicated_blocks', 'duplicated_files', 'lines', 'public_api', 'statements', 'blocker_violations', 'critical_violations',
            'major_violations', 'minor_violations', 'info_violations', 'lines_to_cover', 'line_coverage', 'conditions_to_cover',
            'branch_coverage', 'sqale_index', 'sqale_rating', 'false_positive_issues', 'open_issues', 'reopened_issues',
            'confirmed_issues', 'sqale_debt_ratio', 'new_sqale_debt_ratio', 'code_smells', 'new_code_smells', 'bugs',
            'effort_to_reach_maintainability_rating_a', 'reliability_remediation_effort', 'reliability_rating',
            'security_remediation_effort', 'security_rating', 'cognitive_complexity', 'new_development_cost',
            'security_hotspots', 'security_review_rating']]
    y = dataset[['Toxicos']]
    
    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }
    
    warnings.simplefilter("ignore")
    
    for name, estimator in estimadores.items():
        print("=" * 64)
        print(name)
        
        # Datos originales
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print("Datos Originales")
        print("MSE: " + "%.10f" % float(mse))
        
        # Datos normalizados
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)
        estimator.fit(X_train_norm, y_train)
        predictions_norm = estimator.predict(X_test_norm)
        mse_norm = mean_squared_error(y_test, predictions_norm)
        print("Datos Normalizados")
        print("MSE: " + "%.10f" % float(mse_norm))
        
        # Datos discretizados
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        X_train_disc = discretizer.fit_transform(X_train)
        X_test_disc = discretizer.transform(X_test)
        estimator.fit(X_train_disc, y_train)
        predictions_disc = estimator.predict(X_test_disc)
        mse_disc = mean_squared_error(y_test, predictions_disc)
        print("Datos Discretizados")
        print("MSE: " + "%.10f" % float(mse_disc))




