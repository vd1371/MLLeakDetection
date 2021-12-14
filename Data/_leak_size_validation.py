import pandas as pd
from statistics import mean

df = pd.read_csv("./LeakSize-1000.csv")
columns = [f"LS{i}" for i in range(40)]

df_sizes = df[columns]

def average_non_zeros(df):
	average = []
	for column in df.columns:
		holder = []
		
		for i in df_sizes[column]:
			if i != 0 and type(i) != type("w"):
				holder.append(i)

		average.append(mean(holder))

	return mean(average)

if __name__ == "__main__":
	print(average_non_zeros(df_sizes))