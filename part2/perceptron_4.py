import pandas as pd
import numpy as np
import math




file_path = "../UCIdata-exercise1/iris.data"
df = pd.read_csv(file_path, header=None).sample(frac=1)

acc=[]
sum = 0
for index in range(10):
	# Make the train/test set
	test = df[int(index*len(df)*0.1):int(index*len(df)*0.1)+int(len(df)*0.1)]
	train = df[0:int(index*len(df)*0.1)]
	train = train.append(df[int(index*len(df)*0.1)+int(len(df)*0.1):])

	train = train.values
	test = test.values

	w = [0, 0, 0, 0, 0] # as many as the columns+1

	iteration = 0
	while(1):
		stop_flag = True
		iteration+=1
		for row in train:
			t=1
			if(row[-1] == "Iris-setosa"):
				t=-1
			wtx = 0
			for i, x in enumerate(row[:-1]):
				wtx += w[i]*x*t
			wtx += w[4]*(-1)*t 
			if wtx <= 0:
				#update
				for i in range(len(w)):
					if i<4:
						w[i] = w[i]+row[i]*t
					else:
						w[i] = w[i]+t*(-1)
				# print(t, "*", row ,"===>", w)
				stop_flag = False
		if stop_flag:
			break;
		else:
			continue;


	total_correct = 0
	for row in test:
		pos = 0
		for i, x in enumerate(row[:-1]):
			pos += x*w[i]
		pos += w[4]
		if pos < 0:
			label = "Iris-setosa"
		else:
			label = "Iris-versicolor"
		if label == row[-1]:
			total_correct += 1

	accuracy = total_correct/len(test)
	sum += accuracy
	acc.append(round(accuracy, 4))
	print(index, ") Accuracy: ", accuracy, " iterations: ", iteration)
	# exit()

print("Mean : ", round(sum/10, 4))
print(acc)


