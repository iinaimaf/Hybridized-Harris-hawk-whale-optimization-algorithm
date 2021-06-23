import ANN
import random
import numpy


def update_WOA(E, J, rand_hawk, hawk):
	D = abs(J*rand_hawk - hawk)
	return (rand_hawk - E*D)

def update_hawk1(hawk, rabbit, mean_hawk, rand_hawk, LB, UB):
    q = random.random()
    r1 = random.random()
    r2 = random.random()
    if(q >= 0.5):
        return (rand_hawk - r1*(rand_hawk - 2 * r2 * hawk))
    else:
        return ((rabbit - mean_hawk) - r1*(LB + r2*(UB - LB)))

def update_hawk2(hawk, rabbit, J, E):
    return ((rabbit - hawk) - E*(J*rabbit - hawk))

def update_hawk3(hawk, rabbit, E):
    return (rabbit - E*abs(rabbit - hawk))   #is absolute needed?

def update_hawk4(hawk, rabbit, J, E, sigma, S, data_inputs, data_outputs):
    Y = numpy.array(rabbit - E*(abs(J*rabbit - hawk)))
    LF = 0.01*random.random()*sigma/(pow(random.random(),2/3))
    Z = numpy.array(Y + S[0]*LF)
    fitness_Y = ANN.fitness(numpy.array([Y]), data_inputs,data_outputs, activation="sigmoid")
    fitness_Z = ANN.fitness(numpy.array([Z]), data_inputs,data_outputs, activation="sigmoid")
    fitness_hawk = ANN.fitness(numpy.array([hawk]), data_inputs,data_outputs, activation="sigmoid")
    if(fitness_Y > fitness_hawk):
        return Y
    elif(fitness_Z > fitness_hawk):
        return Z
    elif(fitness_Y > fitness_Z):
        return Y
    return Z

def update_hawk5(hawk, rabbit, mean_hawk, J, E, sigma, S, data_inputs, data_outputs):
    Y = numpy.array(rabbit - E*(abs(J*rabbit - mean_hawk)))
    LF = 0.01*random.random()*sigma/(pow(random.random(),2/3))
    Z = numpy.array(Y + S[0]*LF)
    fitness_Y = ANN.fitness(numpy.array([Y]), data_inputs,data_outputs, activation="sigmoid")
    fitness_Z = ANN.fitness(numpy.array([Z]), data_inputs,data_outputs, activation="sigmoid")
    fitness_hawk = ANN.fitness(numpy.array([hawk]), data_inputs,data_outputs, activation="sigmoid")
    if(fitness_Y > fitness_hawk):
        return Y
    elif(fitness_Z > fitness_hawk):
        return Z
    elif(fitness_Y > fitness_Z):
        return Y
    return Z
