import pandas as pd
import numpy as np
import math

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for data in dataset:
		class_value = data[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(data[:-1]) # removeve the last column
	return separated

# Calculate the Gaussian probability distribution function for x
def calculate_probability(row_x, mean, std, class_prop):
	p = class_prop #not equal posibility
	for i, x in enumerate(row_x):
		exponent = math.exp(-((x-mean[i])**2 / (2 * std[i]**2 )))
		p *= (1 / (math.sqrt(2 * math.pi) * std[i])) * exponent
	return p


file_path = "../UCIdata-exercise1/pima-indians-diabetes.data"
df = pd.read_csv(file_path, header=None).sample(frac=1)

sum = 0

for index in range(10):
	# Make the train/test set
	test = df[int(index*len(df)*0.1):int(index*len(df)*0.1)+int(len(df)*0.1)]
	train = df[0:int(index*len(df)*0.1)]
	train = train.append(df[int(index*len(df)*0.1)+int(len(df)*0.1):])
	train = train.values
	test = test.values

	train_seperated = separate_by_class(train)

	std_list = []
	mean_list = []
	class_propability = []
	labels = []
	for class_label in train_seperated:
		mean_list.append(np.mean(train_seperated[class_label], axis=0))
		std_list.append(np.std(train_seperated[class_label], axis=0))

		class_propability.append(len(train_seperated[class_label])/(float(len(df))*0.9))
		labels.append(class_label)

	print(class_propability)

	total_correct = 0
	for i, row in enumerate(test):
		p0 = calculate_probability(row[:-1], mean_list[0], std_list[0], class_propability[0])
		p1 = calculate_probability(row[:-1], mean_list[1], std_list[1], class_propability[1])

		final_label = labels[0] if p0 > p1  else labels[1]
		if(final_label == row[-1]):
			total_correct +=1	
	accuracy = total_correct/len(test)
	sum += accuracy
	print(index, ") Accuracy: ", round(accuracy, 4))


print("Mean : ", round(sum/10, 4))