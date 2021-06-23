import numpy
import genetic as ga
import ANN
import csv
import math
import time
import SSA
import Neural_network as neural
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



def load_data():
    data_inputs = []
    with open('datasets/dataR2.csv','r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            row = numpy.array(row,dtype = float)
            data_inputs.append(row)
    data_outputs = []
    with open('datasets/label_dataR2.csv','r') as csvfile:
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
inp = data_inputs
out = data_outputs
data_inputs, X, data_outputs, y = train_test_split(data_inputs, data_outputs,test_size = 0.15, random_state = 0)

sol_per_pop = 12
num_generations = 200

HL1_neurons = data_inputs.shape[1] * 2
HL2_neurons = int(data_inputs.shape[1] / 2)
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

best_outputs = []
accuracies = {}
F = pop_weights_mat[0]
F_acc = 0
for generation in range(num_generations):
    print("Generation : ", generation)
    fitness = ANN.fitness(pop_weights_mat, data_inputs,data_outputs, activation="tanh")
    fitness = numpy.array(fitness)
    pop_weights_mat = numpy.array(pop_weights_mat)
    indices = fitness.argsort()
    pop_weights_mat = pop_weights_mat[indices]
    pop_weights_mat = pop_weights_mat[::-1]
    fitness = numpy.sort(fitness)
    fitness = fitness[::-1]
    # if F_acc < fitness[0]:
    #     F = pop_weights_mat[0]
    #     F_acc = fitness[0]
    F = pop_weights_mat[0]
    c1 = 2*math.exp(-(pow((4*(generation+1)/num_generations),2)))
    pop_weights_mat[0] = SSA.update_leader(F,weight_range_1,weight_range_2,c1)
    # leader_fitness = ANN.fitness(numpy.array([new_leader]), data_inputs,data_outputs, activation="sigmoid")
    # if fitness[0] < leader_fitness[0]:
    #     pop_weights_mat[0] = new_leader
    # print(pop_weights_mat[0])
    # time.sleep(15)
    pop_weights_mat = SSA.update_follower(pop_weights_mat, fitness,data_inputs,data_outputs)

    accuracies[generation] = fitness
    print("Fitness")
    print(fitness)


best_weights = F
acc, predictions = ANN.predict_outputs(best_weights, inp, out, activation="tanh")
print(predictions)
print(out)
f = f1_score(out,predictions)
print(f)
