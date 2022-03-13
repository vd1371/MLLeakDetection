import itertools
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def report_accuracy_of_solutions(df, **params):
    logger = params.get('logger')
    report_directory = params.get('report_directory')
    model_name = params.get('model_name')
    n_sections = params.get("n_sections")

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








'''
TODO: The outcome of the optimization is a leak info
consisting of e.g., {'xL': [100, 200, 300],
					'CdAl': [0.5, 0.4, 0.1] }
The ultimate purpose of this function is to compare the predictions
of ML models and optimization models
So, first, the optimization results need to be converted to sections
Then, the results of each section need to be treated as a classification.
'''