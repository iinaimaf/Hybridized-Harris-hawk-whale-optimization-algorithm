import numpy
import ANN
import csv
import random
import Hawk
from sklearn.model_selection import train_test_split

def best(fitness):
	max_fitness = 0
	pos = -1
	for i in range(len(fitness)):
		if fitness[i] > max_fitness:
			max_fitness = fitness[i]
			pos = i
	return pos

def Calculate_mean(pop_weights_mat):
	mean = pop_weights_mat[0]
	for i in range(1,len(pop_weights_mat)):
		mean += pop_weights_mat[i]
	return mean/len(pop_weights_mat)

def load_data():
    data_inputs = []
    with open('datasets/breast_wisconsin.csv','r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            row = numpy.array(row,dtype = float)
            data_inputs.append(row)
    # data_output = [row[-1] for row in data_inputs]
    # data_input = [row[:-1] for row in data_inputs]
    data_outputs = []
    with open('datasets/label_wisconsin.csv','r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            row = numpy.array(row,dtype = float)
            data_outputs.append(row)
    return data_inputs,data_outputs

data_inputs, data_outputs = load_data()

minmax = ANN.dataset_minmax(data_inputs)
ANN.normalize_dataset(data_inputs, minmax)
data_inputs = numpy.array(data_inputs)
data_outputs = numpy.array(data_outputs)
data_inputs, X, data_outputs, y = train_test_split(data_inputs, data_outputs,test_size = 0.15, random_state = 1)

print(data_inputs.shape)

sol_per_pop = 12
num_generations = 100

# HL1_neurons = data_inputs.shape[1] * 2                             #for cancer_patient
# HL2_neurons = int(data_inputs.shape[1]/2)
# output_neurons = 3

HL1_neurons = data_inputs.shape[1] * 2                             #for cancer_patient
HL2_neurons = int(data_inputs.shape[1]/2)
output_neurons = 2

weight_range_1 = -1
weight_range_2 = 1

initial_pop_weights = []
for curr_sol in numpy.arange(0, sol_per_pop):
	input_HL1_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(data_inputs.shape[1], HL1_neurons))
	HL1_HL2_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(HL1_neurons, HL2_neurons))
	HL2_output_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(HL2_neurons, output_neurons))
	initial_pop_weights.append(numpy.array([input_HL1_weights, HL1_HL2_weights, HL2_output_weights]))

pop_weights_mat = numpy.array(initial_pop_weights)

fitness = ANN.fitness(pop_weights_mat, data_inputs,data_outputs, activation="tanh")
best_index = best(fitness)
rabbit = pop_weights_mat[best_index]
E_value = []
best_result = []
best_pop = []
mean_fitness = []
max_f = []
accuracies = numpy.empty(shape=(num_generations))
sigma = pow(((1.33*.707)/(.9064*1.5*pow(2,.25))),2/3)
for generation in range(num_generations):
	print("Generation : ", generation)
	fitness = ANN.fitness(pop_weights_mat, data_inputs,data_outputs, activation="tanh")
	print(fitness)
	best_index = best(fitness)
	max_f.append(fitness[best_index])
	rabbit = pop_weights_mat[best_index]

	accuracies[generation] = fitness[best_index]
	Mean_pop_weight = Calculate_mean(pop_weights_mat)
	mean_fitness.append(Calculate_mean(fitness))
	for i in range(len(pop_weights_mat)):
		if i == best_index:
			continue
		input_HL1_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(data_inputs.shape[1], HL1_neurons))
		HL1_HL2_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(HL1_neurons, HL2_neurons))
		HL2_output_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(HL2_neurons, output_neurons))
		S = []
		S.append(numpy.array([input_HL1_weights, HL1_HL2_weights, HL2_output_weights]))
		S = numpy.array(S)
		base_E = 2*random.random() - 1
		J = 2*(1 - random.random())
		r = random.random()
		E = 2*base_E*(1 - generation/num_generations)
		E_value.append(E)
		if abs(E) >= 1:
			rand_hawk_index = random.randrange(0,sol_per_pop)
			while(rand_hawk_index != i):
				rand_hawk_index = random.randrange(0,sol_per_pop)
			rand_hawk = pop_weights_mat[rand_hawk_index]
			pop_weights_mat[i] = Hawk.update_hawk1(pop_weights_mat[i], rabbit, Mean_pop_weight, rand_hawk, weight_range_1, weight_range_2) #update using eq 1
		else:
			if r >= .5 and abs(E) >= .5:
				pop_weights_mat[i] = Hawk.update_hawk2(pop_weights_mat[i], rabbit, J, E) #update using eq 4
			elif r >= .5 and abs(E) < .5:
				pop_weights_mat[i] = Hawk.update_hawk3(pop_weights_mat[i], rabbit, E) #update using eq6
			elif r < .5 and abs(E) >= .5:
				pop_weights_mat[i] = Hawk.update_hawk4(pop_weights_mat[i], rabbit, J, E, sigma, S, data_inputs, data_outputs) #update using eq10
			else:
				pop_weights_mat[i] = Hawk.update_hawk5(pop_weights_mat[i], rabbit, Mean_pop_weight, J, E, sigma, S, data_inputs, data_outputs) #update using eq11



import matplotlib.pyplot as plt

plt.plot(E_value)
plt.show()
x = [i for i in range(0,100)]
fig, ax = plt.subplots()
ax.plot(x,mean_fitness)
ax.plot(x, max_f)
plt.show()
best_weights = rabbit
acc, predictions = ANN.predict_outputs(best_weights, X, y, activation="tanh")
# print(fitness[best_index])
print(acc)
