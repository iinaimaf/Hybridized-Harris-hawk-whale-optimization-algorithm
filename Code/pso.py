import numpy
import random
def global_best(fitness):
	max_fitness = 0
	pos = -1
	for i in range(len(fitness)):
		if fitness[i]>max_fitness:
			max_fitness = fitness[i]
			pos = i
	return pos

def personal_best(fitness,personal_best_acc,personal_best_wt,current_pop_weights):
	for i in range(len(personal_best_acc)):
		if personal_best_acc[i] < fitness[i]:
			personal_best_acc[i] = fitness[i]
			personal_best_wt[i] = current_pop_weights[i]
	return personal_best_acc, personal_best_wt

def update_position(global_best_wt,personal_best_wt,pop_weights_mat,velocity):
	for i in range(len(pop_weights_mat)):
		velocity[i] = velocity[i] + 2*random.random()*(personal_best_wt[i] - pop_weights_mat[i]) + 2*random.random()*(global_best_wt - pop_weights_mat[i])
		pop_weights_mat[i] = pop_weights_mat[i] + velocity[i]
	return pop_weights_mat,velocity

def update_position_val(global_best_wt,personal_best_wt,pop_weights_mat,velocity,weight_range_1,weight_range_2):
	for i in range(len(pop_weights_mat)):
		velocity[i] = velocity[i] + 2*random.random()*(personal_best_wt[i] - pop_weights_mat[i]) + 2*random.random()*(global_best_wt - pop_weights_mat[i])
		pop_weights_mat[i] = pop_weights_mat[i] + velocity[i]
		if (pop_weights_mat[i]<weight_range_1) or (pop_weights_mat[i]>weight_range_2):
			pop_weights_mat[i] = random.uniform(weight_range_1,weight_range_2)
	return pop_weights_mat,velocity

# function testing
# datasets boosting   german, australian, banknote, credit card
# pso optimising
#
