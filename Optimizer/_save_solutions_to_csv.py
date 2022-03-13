import os
import pandas as pd
import numpy as np
from itertools import chain

def save_solutions_to_csv(holder_of_solutions_and_info, **params):
    report_directory = params.get("report_directory")
    n_sections = params.get("n_sections")
    L = params.get("L")
    model_name = params.get("model_name")
    leak_pred = params.get("leak_pred")
    max_n_leaks = params.get("max_n_leaks")

    l_sections = L/n_sections

    report_act = pd.DataFrame()
    report_pred = pd.DataFrame()
    report_xl = pd.DataFrame()
    report_cdal = pd.DataFrame()


    cols_act = [f'Actual-{idx}' for idx in range(n_sections)]
    cols_pred = [f'Predicted-{idx}' for idx in range(n_sections)]
    cols_xl = [f'xL{idx}' for idx in range(max_n_leaks)]
    cols_cdal = [f'CdAl{idx}' for idx in range(max_n_leaks)]

    for i in range(len(holder_of_solutions_and_info)):

        holder_act = holder_of_solutions_and_info[i][1].tolist()
        holder_pred = np.zeros(n_sections)
        holder_xl = holder_of_solutions_and_info[i][0][:max_n_leaks]
        holder_cdal = holder_of_solutions_and_info[i][0][max_n_leaks:]

        counter = 0

        for leak_loc in holder_of_solutions_and_info[i][2]['xL']:
            idx = int(np.floor(leak_loc/l_sections))

            if idx < n_sections:
                holder_pred[idx] = 1
            else:
                counter += 1
                if counter == 1:
                    holder_pred[n_sections-1] = 1
                elif counter == 2:
                    holder_pred[n_sections-2] = 1
                elif counter == 3:
                    holder_pred[n_sections-3] = 1

        tmp_lst = []
        tmp_lst.append(holder_act)        
        tmp_df = pd.DataFrame(tmp_lst, columns=cols_act)    
        report_act = pd.concat([report_act, tmp_df], axis=0, ignore_index=True)

        tmp_lst = []
        tmp_lst.append(holder_pred)
        tmp_df = pd.DataFrame(tmp_lst, columns=cols_pred)    
        report_pred = pd.concat([report_pred, tmp_df], axis=0, ignore_index=True)

        tmp_lst = []
        tmp_lst.append(holder_xl)
        tmp_df = pd.DataFrame(tmp_lst, columns=cols_xl)    
        report_xl = pd.concat([report_xl, tmp_df], axis=0, ignore_index=True)

        tmp_lst = []
        tmp_lst.append(holder_cdal)
        tmp_df = pd.DataFrame(tmp_lst, columns=cols_cdal)    
        report_cdal = pd.concat([report_cdal, tmp_df], axis=0, ignore_index=True)       


    report = pd.concat([report_act, report_pred, report_xl, report_cdal], axis=1)

    report.to_csv(report_directory + "/" + f'{model_name}-{leak_pred}.csv', index=False)

    return report