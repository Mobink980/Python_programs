import numpy as np
from scipy.optimize import minimize

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
print('Initial Objective: ' + str(objective(x0)))

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
                    
x = solution.x

# show final objective
print('Final Objective: ' + str(objective(x)))

# print solution
print('Solution')
print('x1 = ' + str(x[0]))
print('x2 = ' + str(x[1]))
print('x3 = ' + str(x[2]))
print('x4 = ' + str(x[3]))
print('x5 = ' + str(x[4]))
print('x6 = ' + str(x[5]))
print('x7 = ' + str(x[6]))
print('x8 = ' + str(x[7]))
print('x9 = ' + str(x[8]))
print('x10 = ' + str(x[9]))