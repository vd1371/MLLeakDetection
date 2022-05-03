import numpy as np
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

def R2(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]**2

def CorCoef(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]

def report_accuracy_of_solutions(df, **params):
    logger = params.get('logger')
    report_directory = params.get('report_directory')
    model_name = params.get('model_name')
    n_sections = params.get("n_sections")
    max_n_leaks = params.get("max_n_leaks")

    y_true = []
    y_pred = []
    for i in range(n_sections):

    	y_true.append(df[f'Actual-{i}'])
    	y_pred.append(df[f'Predicted-{i}'])

    y_true = list(itertools.chain(*y_true))
    y_pred = list(itertools.chain(*y_pred))

    acc = round(accuracy_score(y_true, y_pred),4)

    logger.info(f"----------Classification Report for {model_name}------------\n" + \
                    str(classification_report(y_true, y_pred))+"\n")
    logger.info(f"----------Confusion Matrix for {model_name}------------\n" + \
                str(confusion_matrix(y_true, y_pred))+"\n")
    logger.info(f'----------Accurcay for {model_name}------------\n' + \
                    str(round(accuracy_score(y_true, y_pred),4)))

    print (classification_report(y_true, y_pred))
    print (f'Accuracy score for {model_name}', round(accuracy_score(y_true, y_pred),4))
    print ("------------------------------------------------")


    y_true = []
    y_pred = []
    for i in range(max_n_leaks):

        y_true.append(df[f'CdAl{i}-act'])
        y_pred.append(df[f'CdAl{i}-pred'])

    y_true = list(itertools.chain(*y_true))
    y_pred = list(itertools.chain(*y_pred))
    
    corcoef_ = CorCoef(y_true, y_pred)
    r2_ = R2(y_true, y_pred)
    mse_ = MSE(y_true, y_pred)
    mae_ = MAE(y_true, y_pred)

    report_str = f"CorCoef= {corcoef_:.4f}, "\
                    f"R2= {r2_:.4f}, RMSE={mse_**0.5:.4f}, "\
                        f"MSE={mse_:.4f}, MAE={mae_:.4f}, "
    
    logger.info(report_str)
    print(report_str)






'''
TODO: The outcome of the optimization is a leak info
consisting of e.g., {'xL': [100, 200, 300],
					'CdAl': [0.5, 0.4, 0.1] }
The ultimate purpose of this function is to compare the predictions
of ML models and optimization models
So, first, the optimization results need to be converted to sections
Then, the results of each section need to be treated as a classification.
'''