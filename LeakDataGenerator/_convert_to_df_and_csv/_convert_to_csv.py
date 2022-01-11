import pandas as pd
import numpy as np

def convert_to_csv(df_leak_locs, df_leak_size, batch_number):

	df_leak_locs = df_leak_locs.applymap(np.float32)
	df_leak_locs.to_csv(f"./Data/LeakLocs-{batch_number}.csv")

	df_leak_size = df_leak_size.applymap(np.float32)
	df_leak_size.to_csv(f"./Data/LeakSize-{batch_number}.csv")