# import numpy as np
# import pandas as pd

# def make_2(x):
# 	x = 2
# 	return x

# df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
# print(df)
# df.applymap(lambda x: len(str(x)))
# print(df)
# # a = np.array([1,2,3])
# # print(a)


# Implementation of matplotlib function
# import matplotlib.pyplot as plt
     
# plt.plot([1, 2, 3, 4], [16, 4, 1, 8]) 
# # plt.clf()
# plt.ion()

# plt.title('matplotlib.pyplot.clf Example')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
   
# #the function to turn on interactive mode
# # plt.ion()
  
# #creating randomly generate collections/data
# random_array = np.arange(-4, 5)
# collection_1 = random_array ** 2
# collection_2 = 10 / (random_array ** 2 + 1)
# figure, axes = plt.subplots()
  
# axes.plot(random_array, collection_1,
#           'rx', random_array,
#           collection_2, 'b+', 
#           linestyle='solid')
  
# axes.fill_between(random_array, 
#                   collection_1, 
#                   collection_2,
#                   where=collection_2>collection_1, 
#                   interpolate=True,
#                   color='green', alpha=0.3)
  
# lgnd = axes.legend(['collection-1',
#                     'collection-2'], 
#                    loc='upper center', 
#                    shadow=True)
  
# lgnd.get_frame().set_facecolor('#ffb19a')


# import matplotlib.pyplot as plt
  
# plt.ion()
# plt.plot([1.4, 2.5])
# plt.title(" Sampple interactive plot")
  
# axes = plt.gca()
# axes.plot([3.1, 2.2])

# print("{:.2f}".format(25.226))

# import matplotlib.pyplot as plt
# import numpy as np

# xticks = [i for i in range(25)]
# yticks = [i for i in range(25)]

# plt.plot(xticks, np.imag(yticks), label = 'imag')
# plt.plot(xticks, np.real(yticks), label = 'real')

# plt.legend()
# plt.draw()

# import numpy as np

# a = np.array([1,2,3])
# print(type(a))

# from RegressionReport import evaluate_regression


# import exponential
# import numpy as np
# import matplotlib.pyplot as plt
  
# # Using exponential() method
# gfg = np.random.exponential(0.5,10000)
  
# count, bins, ignored = plt.hist(gfg, 14, density = True)
# plt.show()
# # print(len(gfg))

# import pandas as pd
# from statistics import mean

# df = pd.read_csv("C:/Users/Pishtaz/Desktop/gittraining/MLLeakDetection-main/Data/LeakSize-1000.csv")

# def average_of_non_zeros(df):

# 	averages_list = []
# 	columns = [f"LS{i}" for i in range(39)]
# 	holder = []

# 	for column in df[columns]:

# 		for j in column:
# 			print(type(column))
# 			if j != 0 and type(j) != type("word"):
# 				holder.append(j)

# 		# averages_list.append(mean(holder))
# 	return averages_list


# print(average_of_non_zeros(df))


# import pandas as pd
# from statistics import mean

# df = pd.read_csv("C:/Users/Pishtaz/Desktop/gittraining/MLLeakDetection-main/Data/LeakSize-1000.csv")
# columns = [f"LS{i}" for i in range(39)]

# df_sizes = df[columns]

# holder = []

# for i in df_sizes.LS33:
# 	if i != 0 and type(i) != type("word"):
# 		holder.append(i) 

# # print(holder)
# print(mean(holder))

# import numpy as np
# random_location = True
# random_size = True

# # pipe length. range 100:10'000
# L = 2000
# # sound speed. 900:1200 important
# a = 1200
# # pipe diameter  not important        
# D = 0.5
# # area of pipe
# A = np.pi*(D/2)**2
# # friction parameter
# f = 0.00
# n_leaks = 3
# max_n_leaks = 3

# # location of leaks: two leaks far away from each other
# if random_location:
# 	# n_leaks = np.random.choice([1, 2, 3])
# 	xL = list(np.array(sorted(np.random.random(max_n_leaks)))*L)
# else:
# 	xL = list(np.array([0.8, 0.6, 0.25])*L)

# if random_size:
# 	CdAl = np.random.exponential(0.2,max_n_leaks)
# else:
# 	CdAl = np.array([0.01, 0.01, 0.01])
# params = {"L": L,
# 		'n_leaks': n_leaks,
# 		'max_n_leaks': max_n_leaks,
# 		'xL': xL,
# 		'sound_speed': a,
# 		"diameter": D,
# 		'Area_of_pipe': A,
# 		'friction_parameter': f,
# 		'CdAl': CdAl
# 	}

# print(params.pop("n_leakss",))
# print(params)


# import pandas as pd

# data = pd.read_csv("C:/Users/Pishtaz/Desktop/gittraining/MLLeakDetection-main/Data/LeakLocs-1000.csv", header = 0, index_col = 0)
# print(data.head())
# import pandas as pd

# data = pd.read_csv("../Data/LeakLocs-1000.csv", header = 0, index_col = 0)
# print(data.head())


# import os
# import pandas as pd
# data_directory = "../Data/"
# batch_number = 1000
# data = pd.read_csv(os.path.join(data_directory, f"LeakLocs-{batch_number}.csv") , header = 0, index_col = 0)

# print(data.head())


# import pandas as pd
# from keras.layers import Dense
# import tensorflow
# from tensorflow.keras import Input
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from keras.layers import Dropout
# from tensorflow.keras.optimizers import Adam

# df = pd.read_csv('../Data/LeakLocs-1000.csv')

# X, Y = df.iloc[:,1:-40], df.iloc[:,-40:]
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# model = Sequential()

# model.add(Dense(100, input_shape = (50,), activation = 'relu'))
# model.add(Dropout(0.2))

# model.add(Dense(200, activation = 'relu'))
# model.add(Dropout(0.2))

# model.add(Dense(400, activation = 'relu'))
# model.add(Dropout(0.2))

# model.add(Dense(800, activation = 'relu'))
# model.add(Dropout(0.2))

# model.add(Dense(400, activation = 'relu'))
# model.add(Dropout(0.2))

# model.add(Dense(200, activation = 'relu'))
# model.add(Dropout(0.2))

# model.add(Dense(100, activation = 'relu'))
# model.add(Dropout(0.2))

# model.add(Dense(40, activation = 'relu'))
# model.add(Dropout(0.2))

# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# model.summary()

# history = model.fit(X_train, Y_train,
# 		  batch_size = 32,
# 		  epochs = 50, 
# 		  verbose = 2, 
# 		  validation_split = 0.2)


# import pprint

# # stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
# # _ = pprint.pformat(stuff)

# # print(type(_))
# from logging import Logger
# # import Logger
# import logging

# Logger.debug(msg='ssss')
# Logger.info(msg='ssss')
# Logger.warning(msg='ssss') #log output: WARNING:root:ssss
# Logger.error(msg='ssss') #log output: ERROR:root:ssss
# Logger.critical(msg='ssss') #log output: CRITICAL:root:ssss



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../Data/LeakLocs-1000.csv')
X, Y = df.iloc[:,1:-40], df.iloc[:,-40:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
print(type(X_train))