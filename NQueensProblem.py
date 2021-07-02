import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from copy import deepcopy



global N
N = 8

"Defining the class chromosome with the property of length"
"In constructor function we create a list with random true and false values"
"and we set the fitness of that chromosome to -inf"


class Chromosome:
    def __init__(self, length):
        self.genes = chrom_generator(N)  # with length N from [1,N+1)
        self.cost = float('+inf')

    def __len__(self):
        return len(self.genes)

    def reset(self):
        self.cost = float('+inf')


"We initialize the first generation (make a size number of lists with the length chrome_size)"
"chrome_size is the number of queens"


def population_init(size, chrom_size): return np.array(
    [Chromosome(chrom_size) for _ in range(size)])

"This function generates a chromosome with deferent genes"


def chrom_generator(n):
    a = []
    rand = np.random.randint(0, n)
    for i in range(n):
        rand = np.random.randint(0, n)
        while rand in a:
            rand = np.random.randint(0, n)
        else:
            a.append(rand)
    return a

# Evaluation of the cost of the chromosome


def cost_eval(chrom, epsilon=-1):
    cost = 0
    chrom_length = len(chrom)
    for i in range(chrom_length):
        for j in range(i+1, chrom_length):  # check for cross threats
            if(abs(chrom.genes[i] - chrom.genes[j]) == abs(i - j)):
                cost += 1
    return cost



"To implement Tournament selection, we choose sel_pressure number of chromosomes"
"from pop array with the help of choice function"
"then we choose the chromosome that has maximum fitness with the help of max function"


def tournament_selection(pop, sel_pressure): return min(
    np.random.choice(pop, sel_pressure), key=lambda c: c.cost)


pop_size = 10
# You can change this to 'tournament' for changing the selection function
selection_method = 'tournament'
# Initializing First Generation
pop = population_init(pop_size, N)
# Calculating Fitness of Each Individual
for chrom in pop:
    chrom.cost = cost_eval(chrom)

# Selecting just an individual for ensuring that implemented functions work properly
if selection_method == 'tournament':
    sel_ind = tournament_selection(pop, 3)
else:
    sel_ind = tournament_selection(pop,3)
print('Selected Individual cost: {}\nGenes: {}'.format(
    sel_ind.cost, sel_ind.genes))


"partially-mapped-crossover"

def PMX(pop, selection_method, pc):

    p1 = tournament_selection(pop,3)  # choose 2 individuals from the pop
    p2 = tournament_selection(pop,3)

    chrom_length = len(p1)
    # select two random points
    point1 = np.random.randint(1, abs(chrom_length/2) - 1)
    point2 = np.random.randint(abs(chrom_length/2), chrom_length - 1)

    NotSeenList1 = []
    NotSeenList2 = []

    if np.random.random() < pc:  # if pc is greater than random number

        c1 = Chromosome(chrom_length)  # childs
        c2 = Chromosome(chrom_length)

        for i in range(chrom_length):  # transfer genes
            if i > point1 and i <= point2:
                c1.genes[i] = p1.genes[i]
                c2.genes[i] = p2.genes[i]
        for i in range(chrom_length):
            if i <= point1 or i > point2:
                # elements that did not already transfered to c1
                NotSeenList1.append(p1.genes[i])
                # elements that did not already transfered to c2
                NotSeenList2.append(p2.genes[i])

        for i in range(chrom_length):  # First child
            if(c1.genes[i] != 0 and p2.genes[i] in NotSeenList1):
                ans = p2.genes[i]
                while c1.genes[i] != 0:
                    i = p2.genes.index(c1.genes[i])
                c1.genes[i] = ans

        for i in range(chrom_length):
            if p2.genes[i] in NotSeenList1:
                if(c1.genes[i] == 0):
                    c1.genes[i] = p2.genes[i]

        for i in range(chrom_length):  # Second child
            if(c2.genes[i] != 0 and p1.genes[i] in NotSeenList2):
                tmp = p1.genes[i]
                while c2.genes[i] != 0:
                    i = p1.index(c2.genes[i])
                c2.genes[i] = tmp

        for i in range(chrom_length):
            if p1.genes[i] in NotSeenList2:
                if(c2.genes[i] == 0):
                    c2.genes[i] = p1.genes[i]

    else:  # if pc is less than random number then don't make any change
        c1 = deepcopy(p1)
        c2 = deepcopy(p2)

    # Reset fitness of each parent
    c1.reset()
    c2.reset()

    return c1, c2

"Swap mutation"
def swap_mutation(chrom, pm):
    chrom_length = len(chrom)
    if np.random.random() < pm:# select two random points       
        point1 = np.random.randint(1, abs(chrom_length/2) - 1)
        point2 = np.random.randint(abs(chrom_length/2), chrom_length - 1)
    chrom[point1], chrom[point2] = chrom[point2], chrom[point1]
    return chrom




def Nqueens_genetic_algorithm(pop_size=10, iter_num=5, pm=.1,
                               pc=.9, seed=0):
   
    # Creating Arrays for saving all the individuals in all populations
    pop_config = np.ndarray((iter_num, pop_size), dtype=object)
    least_cost = float('+inf')

    # Initializing the population
    pop = population_init(pop_size, N)

    # Calculating Fitness of each individual
    for i in range(pop_size):
        pop[i].cost = cost_eval(pop[i])
        if least_cost > pop[i].cost:
            least_cost = pop[i].cost

    pop_config[0] = pop
    # Loop : Selection, Crossover, Mutation
    for generation in range(1, iter_num):
        print('Least Cost in generation {} : {}'.format(
            generation-1, least_cost))
        new_pop = np.array([])

        # Crossover
        for i in range(int(pop_size/2)):
            parent1, parent2 = PMX(pop, tournament_selection(pop,3), pc)
            new_pop = np.append(new_pop, [parent1, parent2])

        # Mutation
        for i in range(pop_size):  # for all individuals in new-pop
            new_pop[i] = swap_mutation(new_pop[i], pm/pop_size)

        # Fitness Calculation
        for i in range(pop_size):
            new_pop[i].cost = cost_eval(pop[i])
            if least_cost > pop[i].cost:
                least_cost = pop[i].cost

        pop = new_pop
        pop_config[generation] = pop

    print('least cost in generation {} : {}'.format(iter_num, least_cost))
    return pop_config


t0 = time.time()  # measure wall time
pop_size = 5
iter_num = 5
pm = 1
pc = 1
items_number = 20
generations = Nqueens_genetic_algorithm(
    pop_size, iter_num, pm, pc)
t1 = time.time() - t0, "seconds"
print("Wall time:" + str(t1))  # seconds

#backtracking

def printSolution(board): 
    for i in range(N): 
        for j in range(N): print (board[i][j], end=" ") 
        print("")
  
  
# A utility function to check if a queen can 
# be placed on board[row][col]. Note that this 
# function is called when "col" queens are 
# already placed in columns from 0 to col -1. 
# So we need to check only left side for 
# attacking queens 
def isSafe(board, row, col): 
  
    # Check this row on left side 
    for i in range(col): 
        if board[row][i] == 1: 
            return False
  
    # Check upper diagonal on left side 
    for i,j in zip(range(row,-1,-1), range(col,-1,-1)): 
        if board[i][j] == 1: 
            return False
  
    # Check lower diagonal on left side 
    for i,j in zip(range(row,N,1), range(col,-1,-1)): 
        if board[i][j] == 1: 
            return False
  
    return True
  
def solveNQUtil(board, col): 
    # base case: If all queens are placed 
    # then return true 
    if col >= N: 
        return True
  
    # Consider this column and try placing 
    # this queen in all rows one by one 
    for i in range(N): 
  
        if isSafe(board, i, col): 
            # Place this queen in board[i][col] 
            board[i][col] = 1
  
            # recur to place rest of the queens 
            if solveNQUtil(board, col+1) == True: 
                return True
  
            # If placing queen in board[i][col 
            # doesn't lead to a solution, then 
            # queen from board[i][col] 
            board[i][col] = 0
  
    # if the queen can not be placed in any row in 
    # this colum col  then return false 
    return False
  
# This function solves the N Queen problem using 
# Backtracking. It mainly uses solveNQUtil() to 
# solve the problem. It returns false if queens 
# cannot be placed, otherwise return true and 
# placement of queens in the form of 1s. 
# note that there may be more than one 
# solutions, this function prints one  of the 
# feasible solutions. 
def solveNQ(): 
    board = [ [0, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0, 0, 0, 0, 0, 0, 0], 
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0] 
             ] 
  
    if solveNQUtil(board, 0) == False: 
        print ("Solution does not exist")
        return False
  
    printSolution(board) 
    return True
  
# driver program to test above function 
t2 = time.time()  # measure wall time
solveNQ() 
t3 = time.time() - t2, "seconds"
print("Wall time:" + str(t3))  # seconds