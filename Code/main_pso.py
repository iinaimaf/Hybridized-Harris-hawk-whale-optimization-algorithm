import numpy
import ANN
import csv
import genetic as ga
import Neural_network as neural
from sklearn.model_selection import train_test_split
import pso


def load_data():
    data_inputs = []
    with open('datasets/No_weighting.csv','r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            row = numpy.array(row,dtype = float)
            data_inputs.append(row)
    data_outputs = []
    with open('datasets/label_weight.csv','r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            row = numpy.array(row,dtype = float)
            data_outputs.append(row)
    return data_inputs,data_outputs

data_inputs, data_outputs = load_data()
# if(len(data_inputs)>1000):
#     data_inputs = numpy.array(data_inputs[:,0:1000])
#     data_outputs = numpy.array(data_outputs[:,0:1000])
data_inputs = numpy.array(data_inputs)
data_outputs = numpy.array(data_outputs)
data_inputs, X, data_outputs, y = train_test_split(data_inputs, data_outputs,test_size = 0.15, random_state = 1)

print(data_inputs.shape)

sol_per_pop = 12
num_generations = 200

HL1_neurons = 150
HL2_neurons = 60
output_neurons = 2

weight_range_1 = -5
weight_range_2 = 5

initial_pop_weights = []
for curr_sol in numpy.arange(0, sol_per_pop):

    input_HL1_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(data_inputs.shape[1], HL1_neurons))
    HL1_HL2_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(HL1_neurons, HL2_neurons))
    HL2_output_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(HL2_neurons, output_neurons))
    initial_pop_weights.append(numpy.array([input_HL1_weights, HL1_HL2_weights, HL2_output_weights]))

pop_weights_mat = numpy.array(initial_pop_weights)
pop_weights_vector = ga.mat_to_vector(pop_weights_mat)

weight_range_1 = 0
weight_range_2 = 0
velocity = []

for curr_sol in numpy.arange(0, sol_per_pop):
    input_HL1_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(data_inputs.shape[1], HL1_neurons))
    HL1_HL2_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(HL1_neurons, HL2_neurons))
    HL2_output_weights = numpy.random.uniform(low=weight_range_1, high=weight_range_2, size=(HL2_neurons, output_neurons))
    velocity.append(numpy.array([input_HL1_weights, HL1_HL2_weights, HL2_output_weights]))
velocity = numpy.array(velocity)

best_outputs = []
accuracies = numpy.empty(shape=(num_generations))
personal_best_wt = pop_weights_mat
personal_best_acc = ANN.fitness(pop_weights_mat, data_inputs,data_outputs, activation="sigmoid")
global_best_wt = []
for generation in range(num_generations):
    print("Generation : ", generation)

    fitness = ANN.fitness(pop_weights_mat, data_inputs,data_outputs, activation="sigmoid")
    accuracies[generation] = fitness
    if(accuracies[generation] == accuracies[generation-10]):
        break
    print("personal_best")
    print(personal_best_acc)

    global_best_index = pso.global_best(fitness)
    global_best_wt = pop_weights_mat[global_best_index]

    personal_best_acc, personal_best_wt = pso.personal_best(fitness,personal_best_acc,personal_best_wt,pop_weights_mat)
    pop_weights_mat, velocity = pso.update_position(global_best_wt,personal_best_wt,pop_weights_mat,velocity)

best_weights = global_best_wt
acc, predictions = ANN.predict_outputs(best_weights, X, y, activation="sigmoid")
print(acc)
