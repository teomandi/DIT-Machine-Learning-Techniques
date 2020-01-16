import pandas as pd
import numpy as np
import math


def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    dinom = math.sqrt((2 * 3.14)**d * np.linalg.det(covariance))
    expNom = np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2)
    return (1/dinom)*expNom


#load the dataset
file_path = "../UCIdata-exercise1/pima-indians-diabetes.data"
df = pd.read_csv(file_path, header=None).sample(frac=1)

sum = 0
for index in range(10):


	# # Split to trainning testing set
	# train = df[:int(df.shape[0]*0.9)]
	# test = df[int(df.shape[0]*0.9):]

	# Make the train/test set
	test = df[int(index*len(df)*0.1):int(index*len(df)*0.1)+int(len(df)*0.1)]
	train = df[0:int(index*len(df)*0.1)]
	train = train.append(df[int(index*len(df)*0.1)+int(len(df)*0.1):])

	train0 = train.loc[train[8] == 0]
	train0_x = train0.iloc[:,:-1].values
	train0_y = train0.iloc[:,-1].values
	P0 = train0.values.shape[0]/train.values.shape[0]
	m0 = np.mean(train0_x, axis=0)
	s0 = np.cov(train0_x.T)


	train1 = train.loc[train[8] == 1]
	train1_x = train1.iloc[:,:-1].values
	train1_y = train1.iloc[:,-1].values
	P1 = train1.values.shape[0]/train.values.shape[0]
	m1 = np.mean(train1_x, axis=0)
	s1 = np.cov(train1_x.T)

	test_x = test.iloc[:,:-1].values
	test_y = test.iloc[:,-1].values

	test_pred =[]

	for row in test_x:
		p0 = multivariate_normal(row, train0_x.shape[1], m0, s0)
		p1 = multivariate_normal(row, train1_x.shape[1], m1, s1)

		if (P0*p0 > P1*p1): 
			test_pred.append(0)
		else:
			test_pred.append(1)

	total_correct = 0
	for i in range(len(test_pred)): 
		if test_pred[i] == test_y[i]: 
			total_correct += 1

	accuracy = total_correct/len(test_pred)
	sum+=accuracy
	print(index, ") Accuracy", accuracy)
print("Mean : ", sum/10)