import random
import ANN
import numpy
def update_leader(F,lb,ub,c1):
    for i in range(len(F)):
        for j in range(len(F[i])):
            c3 = random.uniform(-1,1)
            c2 = random.uniform(0,1)
            if c3 >= 0:
                new_f = F[i][j] + c1*((ub-lb)*c2 + lb)
            else:
                new_f = F[i][j] - c1*((ub-lb)*c2 + lb)
            F[i][j] = new_f
    return F

def update_leader_val(F,lb,ub,c1):
    c3 = random.uniform(-1,1)
    c2 = random.uniform(0,1)
    if c3 >= 0:
        F = F + c1*((ub-lb)*c2 + lb)
    else:
        F = F - c1*((ub-lb)*c2 + lb)
    return F


def update_follower(pop_weights_mat,fitness,data_inputs,data_outputs):
    for i in range(len(pop_weights_mat)):
        if i != 0:
            pop_weights_mat[i] = 0.5*(pop_weights_mat[i] + pop_weights_mat[i-1])
            # new_fitness = ANN.fitness(numpy.array([new_wts]),data_inputs,data_outputs,activation = 'sigmoid')
            # if fitness[i] < new_fitness[0]:
            #     pop_weights_mat[i] = new_wts
    return pop_weights_mat

def update_follower_val(pop_weights_mat,weight_range_1,weight_range_2):
    for i in range(len(pop_weights_mat)):
        if i != 0:
            pop_weights_mat[i] = 0.5*(pop_weights_mat[i] + pop_weights_mat[i-1])
            if (pop_weights_mat[i] < weight_range_1) or (pop_weights_mat[i]>weight_range_2):
                pop_weights_mat[i] = random.uniform(weight_range_1,weight_range_2)
    return pop_weights_mat
