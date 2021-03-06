#Loading dependencies
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def evaluate_classification(*args, **params):
    '''Report classification results
    
    *args should be lists of [label,x , y, inds]
    '''

    report_directory = params.get('report_directory')
    model_name = params.get('model_name')
    logger = params.get('logger')
    noises = params.get('noises')

    counter = 0
    for ls in args:

        if len(ls) == 5:
            label, x, y_true, inds, y_pred = ls
        elif len(ls) == 6:
            label, x, y_true, inds, y_pred, info = ls

        logger.info(f"----------Classification Report for {model_name}-{label}------------\n" + \
                        str(classification_report(y_true, y_pred))+"\n")
        logger.info(f"----------Confusion Matrix for {model_name}-{label}------------\n" + \
                        str(confusion_matrix(y_true, y_pred))+"\n")
        logger.info(f'----------Accurcay for {label}------------\n' + \
                        str(round(accuracy_score(y_true, y_pred),4)))
        
        print (classification_report(y_true, y_pred))
        print (f'Accuracy score for {model_name}-{label}', round(accuracy_score(y_true, y_pred),4))
        print ("------------------------------------------------")
        
        report = pd.DataFrame()
        report['Actual'] = y_true
        report['Predicted'] = y_pred
        report['Ind'] = inds
        
        for col in info.columns:
            report[col] = info[col].values

        report.set_index('Ind', inplace=True)
        report.to_csv(report_directory + "/" + f'{model_name}-{label}.csv') 