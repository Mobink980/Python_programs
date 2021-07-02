import numpy as np
from scipy.optimize import minimize
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy


#========================================================================================
#                        OPTIMIZATION BY SOLVING THE EQUATION
#========================================================================================
def objective(x):
    return 8*(x[0])**2 + 9*(x[1])**2 + 6*(x[2])**2 + 10*(x[3])**2 + 10*(x[4])**2 \
    + 14*(x[5])**2 + 6*(x[6])**2 + 8*(x[7])**2 + 10*(x[8])**2 + 5*(x[9])**2

#For inequality constraints f(x) >= 0
def constraint1(x):
    #x[0] >= (10/66)
    return x[0]-(10/66) 

def constraint2(x):
    #(10/x[0]) + (12/x[1]) <= 88
    return ((x[0]*x[1])/(12*x[0] + 10*x[1])) - (1/88) 

def constraint3(x):
    #(10/x[0]) + (12/x[1]) + (7/x[2]) <= 107
    return ((x[0]*x[1]*x[2])/(10*x[1]*x[2] + 12*x[0]*x[2] + 7*x[0]*x[1])) - (1/107) 

def constraint4(x):
    #(10/x[0]) + (12/x[1]) + (7/x[2]) + (14/x[3]) <= 128
    return ((x[0]*x[1]*x[2]*x[3])/(10*x[1]*x[2]*x[3] + 12*x[0]*x[2]*x[3] + 7*x[0]*x[1]*x[3] + 14*x[0]*x[1]*x[2])) - (1/128)

def constraint5(x):
    #(10/x[0]) + (12/x[1]) + (7/x[2]) + (14/x[3]) + (15/x[4]) <= 157
    return ((x[0]*x[1]*x[2]*x[3]*x[4])/(10*x[1]*x[2]*x[3]*x[4] + 12*x[0]*x[2]*x[3]*x[4] + 7*x[0]*x[1]*x[3]*x[4] + 14*x[0]*x[1]*x[2]*x[4] + 15*x[0]*x[1]*x[2]*x[3])) - (1/157)

def constraint6(x):
    #(10/x[0]) + (12/x[1]) + (7/x[2]) + (14/x[3]) + (15/x[4]) + (20/x[5])<= 192
    return ((x[0]*x[1]*x[2]*x[3]*x[4]*x[5])/(10*x[1]*x[2]*x[3]*x[4]*x[5] + 12*x[0]*x[2]*x[3]*x[4]*x[5] + 7*x[0]*x[1]*x[3]*x[4]*x[5] + 14*x[0]*x[1]*x[2]*x[4]*x[5] + 15*x[0]*x[1]*x[2]*x[3]*x[5] + 20*x[0]*x[1]*x[2]*x[3]*x[4])) - (1/192) 

def constraint7(x):
    #(10/x[0]) + (12/x[1]) + (7/x[2]) + (14/x[3]) + (15/x[4]) + (20/x[5]) +  (10/x[6]) <= 222
    return ((x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6])/(10*x[1]*x[2]*x[3]*x[4]*x[5]*x[6] + 12*x[0]*x[2]*x[3]*x[4]*x[5]*x[6] + 7*x[0]*x[1]*x[3]*x[4]*x[5]*x[6] + 14*x[0]*x[1]*x[2]*x[4]*x[5]*x[6] + 15*x[0]*x[1]*x[2]*x[3]*x[5]*x[6] + 20*x[0]*x[1]*x[2]*x[3]*x[4]*x[6] + 10*x[0]*x[1]*x[2]*x[3]*x[4]*x[5])) - (1/222) 

def constraint8(x):
    #(10/x[0]) + (12/x[1]) + (7/x[2]) + (14/x[3]) + (15/x[4]) + (20/x[5]) +  (10/x[6]) + (10/x[7]) <= 242
    return ((x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7])/(10*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7] + 12*x[0]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7] + 7*x[0]*x[1]*x[3]*x[4]*x[5]*x[6]*x[7] + 14*x[0]*x[1]*x[2]*x[4]*x[5]*x[6]*x[7] + 15*x[0]*x[1]*x[2]*x[3]*x[5]*x[6]*x[7] + 20*x[0]*x[1]*x[2]*x[3]*x[4]*x[6]*x[7] + 10*x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[7] + 10*x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6])) - (1/242) 

def constraint9(x):
    #(10/x[0]) + (12/x[1]) + (7/x[2]) + (14/x[3]) + (15/x[4]) + (20/x[5]) +  (10/x[6]) + (10/x[7]) + (16/x[8]) <= 268
    return ((x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8])/(10*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8] + 12*x[0]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8] + 7*x[0]*x[1]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8] + 14*x[0]*x[1]*x[2]*x[4]*x[5]*x[6]*x[7]*x[8] + 15*x[0]*x[1]*x[2]*x[3]*x[5]*x[6]*x[7]*x[8] + 20*x[0]*x[1]*x[2]*x[3]*x[4]*x[6]*x[7])*x[8] + 10*x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[7]*x[8] + 10*x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[8] + 16*x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]) - (1/268) 


def constraint10(x):
    #(10/x[0]) + (12/x[1]) + (7/x[2]) + (14/x[3]) + (15/x[4]) + (20/x[5]) +  (10/x[6]) + (10/x[7]) + (16/x[8]) + (8/x[9]) <= 292
    return ((x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8]*x[9])/(10*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8]*x[9] + 12*x[0]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8]*x[9] + 7*x[0]*x[1]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8]*x[9] + 14*x[0]*x[1]*x[2]*x[4]*x[5]*x[6]*x[7]*x[8]*x[9] + 15*x[0]*x[1]*x[2]*x[3]*x[5]*x[6]*x[7]*x[8]*x[9] + 20*x[0]*x[1]*x[2]*x[3]*x[4]*x[6]*x[7]*x[8]*x[9] + 10*x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[7]*x[8]*x[9] + 10*x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[8]*x[9] + 16*x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[9] + 8*x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8])) - (1/292) 


def solve_equation():
    # initial guesses
    n = 10
    x0 = np.zeros(n)
    x0[0] = 0.01
    x0[1] = 0.02
    x0[2] = 0.03
    x0[3] = 0.08
    x0[4] = 0.09
    x0[5] = 0.08
    x0[6] = 0.07
    x0[7] = 0.04
    x0[8] = 0.09
    x0[9] = 0.05

    # show initial objective
    # print('Initial Objective: ' + str(objective(x0)))

    # optimize
    b = (0.01, 1.0)
    bnds = (b, b, b, b, b, b, b, b, b, b)
    con1 = {'type': 'ineq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}
    con3 = {'type': 'ineq', 'fun': constraint3}
    con4 = {'type': 'ineq', 'fun': constraint4}
    con5 = {'type': 'ineq', 'fun': constraint5}
    con6 = {'type': 'ineq', 'fun': constraint6}
    con7 = {'type': 'ineq', 'fun': constraint7}
    con8 = {'type': 'ineq', 'fun': constraint8}
    con9 = {'type': 'ineq', 'fun': constraint9}
    con10 = {'type': 'ineq', 'fun': constraint10}


    cons = ([con1, con2, con3, con4, con5, con6, con7, con8, con9, con10])
    solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
                    
    results = solution.x
    # print(solution)
    energy  = solution.fun

    # show final objective
    # print('Final Objective: ' + str(objective(x)))

    return results, energy

    


#========================================================================================
#                        OPTIMIZATION USING BRUTE FORCE
#========================================================================================
def brute_force(step):  
    """
    A function that searches all possible combinations (Brute Force) and finds
    a collection of values for DVFS coefficients that minimizes the energy function
    due to the constraints (optimization depends on the step).
    Returns optimum coefficients and corresponding energy
    """
    #for the length of for loop
    length = (int)(1/step)
    task_numbers = 10 
    x = [step]*task_numbers #list contining different set of coefficients
    optimum = [step]*task_numbers # A list that saves the best results for minimizing the energy function
    energy = calculate_energy(x) # energy consumption with coefficients x
    value = 0.0 #for storing energy values for different coefficients 

    for a in range(length-1):
        for b in range(length-1):
            for c in range(length-1):
                for d in range(length-1):
                    for e in range(length-1):
                        for f in range(length-1):
                            for g in range(length-1):
                                for h in range(length-1):
                                    for i in range(length-1):
                                        for j in range(length-1):
                                            value = calculate_energy(x)
                                            if (value < energy):
                                                energy = value #update value of energy
                                                optimum = x    #update optimum coefficients
                                            x[9] += step
                                        #End of loop1
                                        x[8] += step
                                        x[9] = step
                                    #End of loop2
                                    x[7] += step
                                    x[8] = step
                                    x[9] = step
                                #End of loop3
                                x[6] += step
                                x[7] = step
                                x[8] = step
                                x[9] = step
                            #End of loop4
                            x[5] += step
                            x[6] = step
                            x[7] = step
                            x[8] = step
                            x[9] = step
                        #End of loop5
                        x[4] += step
                        x[5] = step
                        x[6] = step
                        x[7] = step
                        x[8] = step
                        x[9] = step
                    #End of loop6
                    x[3] += step
                    x[4] = step
                    x[5] = step
                    x[6] = step
                    x[7] = step
                    x[8] = step
                    x[9] = step
                #End of loop7
                x[2] += step
                x[3] = step
                x[4] = step
                x[5] = step
                x[6] = step
                x[7] = step
                x[8] = step
                x[9] = step
            #End of loop8
            x[1] += step
            x[2] = step
            x[3] = step
            x[4] = step
            x[5] = step
            x[6] = step
            x[7] = step
            x[8] = step
            x[9] = step
        #End of loop9
        x[0] += step
        x[1] = step
        x[2] = step
        x[3] = step
        x[4] = step
        x[5] = step
        x[6] = step
        x[7] = step
        x[8] = step
        x[9] = step
    #End of loop10 

    return optimum, energy 

def conditions(x):  
    """
    This function will check whether the constraints that apply to 
    our optimization are met or not.
    """
    if ( (10/x[0]) > 66.0 ):
        return False
    elif ( (10/x[0] + 12/x[1]) > 88.0 ):
        return False
    elif ( (10/x[0] + 12/x[1] + 7/x[2]) > 107.0 ):
        return False
    elif ( (10/x[0] + 12/x[1] + 7/x[2] + 14/x[3]) > 128.0 ):
        return False
    elif ( (10/x[0] + 12/x[1] + 7/x[2] + 14/x[3] + 15/x[4]) > 157.0 ):
        return False
    elif ( (10/x[0] + 12/x[1] + 7/x[2] + 14/x[3] + 15/x[4] + 20/x[5]) > 192.0 ):
        return False
    elif ( (10/x[0] + 12/x[1] + 7/x[2] + 14/x[3] + 15/x[4] + 20/x[5] + 10/x[6]) > 222.0 ):
        return False
    elif ( (10/x[0] + 12/x[1] + 7/x[2] + 14/x[3] + 15/x[4] + 20/x[5] + 10/x[6] + 10/x[7]) > 242.0 ):
        return False
    elif ( (10/x[0] + 12/x[1] + 7/x[2] + 14/x[3] + 15/x[4] + 20/x[5] + 10/x[6] + 10/x[7] + 16/x[8]) > 268.0 ): 
        return False
    elif ( (10/x[0] + 12/x[1] + 7/x[2] + 14/x[3] + 15/x[4] + 20/x[5] + 10/x[6] + 10/x[7] + 16/x[8] + 8/x[9]) > 292.0 ): 
        return False

    return True 



def calculate_energy(x):  
    """
    This function takes the DVFS coefficients (as a list named x),
    and will calculate the energy consumption for this sequence of
    DVFS coefficients that are applied to the task sequence. 
    """  
    # Defining a positive infinite integer
    positive_infinity = float('inf')
    energy = positive_infinity
    value = 0.0
    #backslash at the end of the line means the instruction continues to the next line (for easier readability)
    value = 8*(x[0])**2 + 9*(x[1])**2 + 6*(x[2])**2 + 10*(x[3])**2 + 10*(x[4])**2  \
    + 14*(x[5])**2 + 6*(x[6])**2 + 8*(x[7])**2 + 10*(x[8])**2 + 5*(x[9])**2
    if (conditions(x)):
        #if True we conclude that the result is valid (otherwise invalid)
        energy = value

    return energy


#========================================================================================
#                        OPTIMIZATION BY GENETIC ALGORITHM
#========================================================================================  
class Chromosome():
    """
    Description of class `Chromosome`:
    This class represents a simple chromosome. In the method describe, a simple description
    of the chromosome is provided, when it is called. 
    """
    def __init__(self, genes, id_=None, fitness=-1):
        self.id_ = id_
        self.genes = genes
        self.fitness = fitness       
       
    def describe(self): 
        """
        Prints the ID, fitness, and genes
        """
        print(f"ID=#{self.id_}, Fitness={self.fitness}, \nGenes=\n{self.genes}")
 
    def get_chrom_length(self): 
        """
        Returns the length of `self.genes`
        """
        return len(self.genes)
# a floating-point chromosome with genes between 0 and 1
def floating_point_chrome_generator(n):
    """
    Produces chromosomes with floating-point values for each gene. (floating-point alleles)
    n: The length of the chromosome to be produced 
    """  
    return  np.random.uniform(low=0, high=1, size=n).tolist()
     
def fitness_function(chrom): #TODO needs to be more generalize and useful
    """
    This function, takes a chromosome and returns a value as its fitness.
    chrom: The chromosome which its fitness is calculated and returned.
    """  
    return (1/calculate_energy(chrom))
 
def pop_sort(pop):
    """
    This function sorts the population based on their fitnesses, using selection sort.
    pop: The population that are sorted based on their fitnesses.
    """  
    for i in range(len(pop)): 
        min_index = i 
        for j in range(i+1, len(pop)): 
            if pop[min_index].fitness > pop[j].fitness: 
                min_index = j        
        pop[i], pop[min_index] = pop[min_index], pop[i]    
    return pop


def find_best_chromosome(pop):
    """
    This function searches a list of chromosomes and returns the chromosome with the best fitness.
    pop: The given list of chromosomes 
    """  
    best_chrom = Chromosome(genes= np.array([0.0]*10), id_=200, fitness = 0.0)
    for i in range(len(pop)): 
        if(pop[i].fitness > best_chrom.fitness):
            best_chrom = pop[i]
      
    return best_chrom


def pop_initialization(pop_size, chrom_size): 
    """
    This function creates a population with the size pop_size, with chrom_size for the size of the chromosomes.
    pop_size: Size of the population
    chrom_size: Size of each chromosome in the population
    """  
    pop = []
    for i in range(pop_size):
        chrom = floating_point_chrome_generator(chrom_size)
        chrom_fitness = fitness_function(chrom)
        pop.append(Chromosome(genes= np.array(chrom),id_=i,fitness = chrom_fitness))#fitness fixed 

    return pop

def truncation_selection(selection_size, pop):
    """
    In Truncation Selection, Only the fittest members of the population will be selected
    and these individuals will be duplicated to maintain the population.
    This function takes a population, and returns the fittest ones inside.
    selection_size: Number of individuals to be selected within the population.
    pop_size: Size of the population
    pop: The population which we want to select a number of individuals among them. 
    """  
    pop_size = len(pop) #Size of the population
    sorted_pop = pop_sort(pop)
    select = []
    if selection_size > pop_size: #assuring that the selection size is less than the size of the population
        selection_size = pop_size
    d = 0
    for i in range(pop_size-1,-1,-1): #backward loop
        if (d < selection_size):
            select.append(sorted_pop[i])
            d += 1
        else:
            break
    return select


"Heuristic crossover(HC) or Intermidiate crossover(IC)"
def heuristic_crossover(parent_one, parent_two, pc):
    """
    This function takes two parents, and performs Heuristic crossover on them. 
    parent_one: First parent
    parent_two: Second parent
    pc: The probability of crossover
    """  
    
    # print("\nParents")
    # print("=================================================")
    # Chromosome.describe(parent_one)
    # Chromosome.describe(parent_two)
    
    chrom_length = Chromosome.get_chrom_length(parent_one)
    random_integer = np.random.randint(5, 98)
    child_one = Chromosome(genes= np.array([0.0]*chrom_length),id_=random_integer,fitness = 0.0)  # child
    a = np.random.uniform(low=0, high=0.4)
    if np.random.rand() < pc:  # if pc is greater than random number
        for i in range(chrom_length):
            child_one.genes[i] = round(parent_one.genes[i] + a*np.abs(parent_two.genes[i] - parent_one.genes[i]), 5)
            if (child_one.genes[i] > 0.8): child_one.genes[i] = np.random.uniform(low=0.2, high=0.75)

    else:  # if pc is less than random number then don't make any change
        child_one = deepcopy(parent_one)

    #calculating the fitness
    child_one.fitness = fitness_function(child_one.genes)
      
    return child_one

"Creep mutation"
def creep_mutation(parent_one, pm):
    """
    This function takes one chromosome, and performs Creep mutation on it. 
    parent_one: The parent
    pm: The probability of mutation
    """ 
  
    # print("\nParent")
    # print("=================================================")
    # Chromosome.describe(parent_one)
    chrom_length = Chromosome.get_chrom_length(parent_one)
    random_integer = np.random.randint(5, 98)
    child_one = Chromosome(genes= np.array([0.0]*chrom_length),id_=random_integer,fitness = 0.0)  # child

    if np.random.rand() < pm:  # if pm is greater than random number
        point = np.random.randint(0, chrom_length)
        child_one.genes = parent_one.genes 
        child_one.genes[point] = round(np.random.uniform(low=0.2, high=0.85), 5)

    else:  # if pm is less than random number then don't make any change
        child_one = deepcopy(parent_one)
    
    #calculating the fitness
    child_one.fitness = fitness_function(child_one.genes)

    return child_one

def generation_production(pop_size, iterations):
    """
    This function produces generation after generation, 
    to get the DVFS coefficients that sufficiently minimize the energy. 
    pop_size: The size of initial population given for further optimization.
    iterations: The number of iterations that a generation is updated with new offsprings.
    """ 
    pop = pop_initialization(pop_size, 10) #initializing the population
    offsprings = [] # a list containing offsprings produced in crossover and mutation
    best_chroms = [] # a list for saving the best chromosome in each generation
    best_chrom = Chromosome(genes= np.array([0.0]*10), id_=200, fitness = 0.0)
    selection_size = 15
    for i in range(iterations):
        selection = truncation_selection(selection_size, pop) #choose the best 15 from the population
        for m in range(selection_size):
            offsprings.append(selection[m]) #Add selected chromosomes to the list offsprings
        for j in range(10):
            a =  np.random.randint(0, selection_size-1) #choose two random integers in selection_size
            b =  np.random.randint(0, selection_size-1)
            cross = heuristic_crossover(selection[a], selection[b], 0.8) #perform crossover on the chosen chromosomes
            offsprings.append(cross)

        for k in range(5):
            c = np.random.randint(0, selection_size-1) #choose a random integer in selection_size
            mutation = creep_mutation(selection[c], 0.7) #perform mutation on the chosen chromosome
            offsprings.append(mutation)

        #selecting the best of offsprings
        best_offsprings = truncation_selection(20, offsprings) #choose the best 20 from the offsprings

        #replacing the worst members of the population with the best offsprings
        sorted_pop = pop_sort(pop) #sort the population based on their fitnesses (in ascending order)
        for t in range(len(best_offsprings)):
            sorted_pop[t] = best_offsprings[t] #replace the bad chromosomes with good ones

        pop = sorted_pop #update the population for the next iteration

        best_chroms.append(find_best_chromosome(pop)) #saving the best answer in each generation

        # print("\n================ population chroms in generation",i,"=====================\n")
        # for i in range(len(pop)): #printing the selection
        #     Chromosome.describe(pop[i])

    best_chrom = find_best_chromosome(best_chroms) #find the best chromosome from the list of best chromosomes

    return best_chrom
#========================================================================================
#                                    SHOWING THE RESULTS
#========================================================================================
#Showing the results that are achieved by solving the equation
optimum_vector, optimum_energy = solve_equation()
print("==================================================")
print("OPTIMUM SOLUTION")
print("==================================================")
print("Values for DVFS coefficients by solving the equation are:\n", optimum_vector)
print("The optimum energy that can be achieved is:", optimum_energy)



# Showing the results that are achieved by genetic algorithm approach
best_chrom = generation_production(50, 10) #10 iterations
while (calculate_energy(best_chrom.genes) > 16.4):
    best_chrom = generation_production(50, 15) #15 iterations
print("==================================================")
print("GENETIC ALGORITHM SOLUTION")
print("==================================================")
print("\n================   BEST CHROMOSOME ====================\n")
Chromosome.describe(best_chrom)
genetic_energy = calculate_energy(best_chrom.genes)
print("Values for DVFS coefficients by using Genetic Algorithm are:\n", best_chrom.genes)
print("The best energy achieved by genetic algorithm approach is:", genetic_energy)



#Showing the results that are achieved by brute force approach
brute_force_vector, brute_force_energy = brute_force(0.2)
print("==================================================")
print("BRUTE FORCE SOLUTION")
print("==================================================")
print("Values for DVFS coefficients by using Brute Force are:\n", brute_force_vector)
print("The best energy achieved by brute_force approach is:", brute_force_energy)


#Showing the results that are achieved by maximum voltage & frequency
maximum_frequency_vector = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
maximum_frequency_energy = calculate_energy(maximum_frequency_vector)
print("==================================================")
print("MAXIMUM FREQUENCY SOLUTION")
print("==================================================")
print("Values for DVFS coefficients for using Maximum Voltage & Frequency are:\n", maximum_frequency_vector)
print("The best energy achieved by Maximum Frequency approach is:", maximum_frequency_energy)


#Showing the results that are achieved by minimum voltage & frequency possible (THANKS TO MATLAB)
minimum_frequency_vector = [(5/33), (6/11), (7/19), (2/3), (15/29), (4/7), (1/3), (1/2), (8/13), (1/3)]
minimum_frequency_energy = calculate_energy(minimum_frequency_vector)
print("==================================================")
print("MINIMUM FREQUENCY SOLUTION")
print("==================================================")
print("Values for DVFS coefficients for using Minimum Voltage & Frequency are:\n", minimum_frequency_vector)
print("The best energy achieved by Minimum Frequency approach is:", minimum_frequency_energy)



#========================================================================================
#                                    PLOTTING THE ENERGY RESULTS
#========================================================================================
# creating the dataset
data = {'OPTIMUM':optimum_energy, 'GENETIC ALGORITHM':genetic_energy, 'BRUTE FORCE':brute_force_energy,
        'MAXIMUM FREQUENCY':maximum_frequency_energy, 'MINIMUM FREQUENCY':minimum_frequency_energy}
solutions = list(data.keys())
energies = list(data.values())

# creating the bar plot
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
# plt.bar(solutions, energies, color ='maroon', width = 0.4)        
plt.bar(solutions[0], energies[0], color = 'tab:blue', width = 0.4)
plt.bar(solutions[1], energies[1], color = 'tab:green', width = 0.4)
plt.bar(solutions[2], energies[2], color = 'tab:orange', width = 0.4)
plt.bar(solutions[3], energies[3], color = 'tab:red', width = 0.4)
plt.bar(solutions[4], energies[4], color = 'tab:purple', width = 0.4)

plt.xlabel("DIFFERENT APPROACHES")
plt.ylabel("ENERGY")
plt.title("Amount of energy consumed by solving DVFS problem with different approaches")
plt.show()



































































# x = [0.2, 0.4, 0.3, 0.6, 0.6, 0.5, 0.4, 0.4, 0.7, 0.3]
# y = [0.9, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.9, 1.0, 0.9]
# w = [0.500070, 0.508782, 0.490985, 0.525148, 0.518293, 0.617947, 0.555630, 0.487515, 0.534278, 0.534306]
# result1 = conditions(x)
# result2 = conditions(y)
# result3 = conditions(w)
# energy1 = calculate_energy(x)
# energy2 = calculate_energy(y)
# energy3 = calculate_energy(w)
# print(result1)
# print(result2)
# print(result3)
# print(energy1)
# print(energy2)
# print(energy3)

