import pandas as pd
import numpy as np

def save_solutions_to_csv(holder_of_solutions_and_info, **params):
    direc = params.get("direc")
    n_sections = params.get("n_sections")
    model = params.get("model")

    report = pd.DataFrame()

    report['Actual'] = holder_of_solutions_and_info[0][1]
    report['Predicted'] = np.zeros(n_sections)

    for leak_loc in holder_of_solutions_and_info[0][2]['xL']:
        idx = int(np.floor(leak_loc/80))

        if idx <= n_sections:
            report['Predicted'][idx] = 1
        else:
            report['Predicted'][n_sections-1] = 1
            
    # report['Ind'] = inds
    # for col in info.columns:
    #     report[col] = info[col].values

    # report.set_index('Ind', inplace=True)

    # report.to_csv(direc + "/" + f'{model}-{label}.csv')
    report.to_csv(direc + "/" + f'{model}.csv')