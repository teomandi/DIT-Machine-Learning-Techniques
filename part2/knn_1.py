import pandas as pd
import numpy as np
import math
# %matplotlib inline
import matplotlib.pyplot as plt



def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)

def knn(data, query, k): 
    neighbor_distances = []

    for index, row in enumerate(data):
        distance = euclidean_distance(row, query)
        neighbor_distances.append((distance, index))
    
    sorted_neighbor_distances = sorted(neighbor_distances)
    k_nearest_distances_and_indices = sorted_neighbor_distances[:k]

    #return only the indeces
    return [i[1] for i in k_nearest_distances_and_indices] 

def most_frequent(label_list): 
    counter = 0
    num = label_list[0] 
    for i in label_list: 
        curr_frequency = label_list.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num 

#load the dataset
file_path = "../UCIdata-exercise1/pima-indians-diabetes.data"
# file_path = "../UCIdata-exercise1/iris.data"
df = pd.read_csv(file_path, header=None).sample(frac=1)


ks = []
means = []



for k in range(1,20):

	sum = 0

	print("Rows: ", len(df), " K: ", k)
	for index in range(10):
		# Make the train/test set
		test = df[int(index*len(df)*0.1):int(index*len(df)*0.1)+int(len(df)*0.1)]
		train = df[0:int(index*len(df)*0.1)]
		train = train.append(df[int(index*len(df)*0.1)+int(len(df)*0.1):])


		train_x = train.iloc[:,:-1].values
		train_y = train.iloc[:,-1].values

		test_x = test.iloc[:,:-1].values
		test_y = test.iloc[:,-1].values


		## ------ KNN ------ ## 
		total_correct = 0
		t_y = list(train_y)
		for i in range(len(test_x)):
			results=knn(train_x, test_x[i], k)
			label_results = []
			for j in sorted(results):
				label_results.append(t_y[j])
			# print(list(label_results))
			final_label = most_frequent(list(label_results))

			if(final_label == test_y[i]):
				total_correct +=1

		accuracy = total_correct/len(test_x)
		sum += accuracy
		# print(index, ") Accuracy: ", round(accuracy, 4))


	means.append(sum/10)
	ks.append(k)
	print("Mean : ", round(sum/10, 4))


plt.plot(means, ks)
plt.xlabel('Mean')
plt.ylabel('K')