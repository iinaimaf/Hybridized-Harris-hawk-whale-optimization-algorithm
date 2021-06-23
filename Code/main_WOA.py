import numpy
import ANN
import time
import csv
import random
import WOA
from sklearn.model_selection import train_test_split
from csv import reader
import NeuralNetwork

def best_index(fitness):
	max_fitness = 0
	pos = -1
	for i in range(len(fitness)):
		if fitness[i] > max_fitness:
			max_fitness = fitness[i]
			pos = i
	return pos

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			m = []
			for r in row:
				m.append(float(r))
			dataset.append(m)
	return dataset

def load_data():
    data_inputs = []
    with open('datasets/banknote_authentication.csv','r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            row = numpy.array(row,dtype = float)
            data_inputs.append(row)
    return data_inputs

filename = 'datasets/banknote_authentication.csv'
dataset = load_csv(filename)        #checked
random.shuffle(dataset)				#checked
random.shuffle(dataset)
# dataset = load_data()
minmax = NeuralNetwork.dataset_minmax(dataset)				#checked

NeuralNetwork.normalize_dataset(dataset,minmax)				#checked

data_input = [row[:-1] for row in dataset]
data_output = [row[-1] for row in dataset]

data_inputs, X, data_outputs, y = train_test_split(data_input, data_output,test_size = 0.15, random_state = 0)
train_dataset = []
print(len(data_inputs),len(data_outputs))
# print(data_inputs.shape, data_outputs.shape)
for i in range(len(data_inputs)):
	data_inputs[i].append(data_outputs[i])
	train_dataset.append(data_inputs[i]) 					#checked
# for i in train_dataset:
# 	print(i)
test_dataset = []
for i in range(len(X)):
	X[i].append(y[i])
	test_dataset.append(X[i])

sol_per_pop = 7
num_generations = 100
n_inputs = len(dataset[0]) - 1
HL1_neurons = int(n_inputs*2)
HL2_neurons = int(n_inputs/2)
# h_neuron = [HL1_neurons, HL2_neurons]
h_neuron = [HL1_neurons]
n_outputs = len(set([row[-1] for row in dataset]))
# n_outputs = 1
train_dataset = train_dataset[:5]
weight_range_1 = -1
weight_range_2 = 1

initial_pop_weights = []

for i in range(sol_per_pop):
	initial_pop_weights.append(NeuralNetwork.initialize_network(n_inputs, h_neuron, n_outputs, weight_range_1, weight_range_2))
for i in initial_pop_weights:
	print()
	print(i)
fitness = []
accuracies = numpy.empty(shape=(num_generations))
for i in range(sol_per_pop):
	fitness.append(NeuralNetwork.cal_fitness(initial_pop_weights[i],train_dataset))
print(fitness)
best = best_index(fitness)
best_agent = initial_pop_weights[best]
a = 2
rand_pop = [i for i in range(0,sol_per_pop)]
print(rand_pop, best)
for generation in range(num_generations):
    fitness = []
    print("Generation : ", generation)
    a = a - 2/num_generations
    for i in range(0,sol_per_pop):
        if i != best:
            r = random.random()
            A = 2*a*r - a
            C = 2*r
            prob = [0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,1,1,0]
            l = random.random()*-1 if random.choice(prob) else random.random()  #for value between -1, 1
            p = random.random()
            if(p < 0.5):
                if(abs(A) < 1):
                    parent  = list(initial_pop_weights[i])
                    new_agent = list(best_agent)
                    initial_pop_weights[i] = WOA.update_WOA1(parent, new_agent, C, A)
                else:
                    rand_pop.remove(i)
                    parent  = list(initial_pop_weights[i])
                    rand_agent = list(initial_pop_weights[random.choice(rand_pop)])
                    initial_pop_weights[i] = WOA.update_WOA1(parent, rand_agent, C, A)
                    rand_pop.append(i)
            else:
                parent  = list(initial_pop_weights[i])
                agent = list(best_agent)
                initial_pop_weights[i] = WOA.update_WOA3(agent, parent, C, l)

            print(initial_pop_weights[i])
            time.sleep(3)
    for i in range(sol_per_pop):
        fitness.append(NeuralNetwork.cal_fitness(initial_pop_weights[i],train_dataset))
    for i in initial_pop_weights:
        print()
        print(i)
    for i in train_dataset:
        print()
        print(i)
    time.sleep(15)
    print(fitness)
    best = best_index(fitness)
    best_agent = initial_pop_weights[best]

best_weights = best_agent
print(best_agent)
acc, predictions = ANN.predict_outputs(best_weights, X, y, activation="sigmoid")
print(acc)
