import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets
import os

# pd.set_option('display.height', 6)
pd.set_option('display.max_rows', 480)
pd.set_option('display.max_columns', 18)
# pd.set_option('display.width', 150)

data_root = 'data'

categorical_attr = ['gender', 'NationalITy', 'PlaceofBirth', 'GradeID', 'StageID',
                    'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',
                   'ParentschoolSatisfaction', 'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class']
# loading dataset
df = pd.read_excel(
    'C:\\Users\\taban\\Desktop\\First Exercise\\CI\\Iris\\students.xlsx')

# Converting Categorical values to scaler values
le = LabelEncoder()
df[categorical_attr] = df[categorical_attr].apply(le.fit_transform, axis=0)


# X: Features 
# y: Classes   (.iloc[:, :-1] selects all columns but not the last one)
X = np.array(df.iloc[:, 1:15])
Y = np.array(df['Class']) # Only the last column which is the Class column

# Dividing Dataset to training set and test set(validation):
# train_test_split, Splits arrays or matrices into random train and test subsets
X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

# Implementing Logistic Regression Model
y1 = (df['Class'] == 1) * 1 # low students and others(output 1 if L, otherwise 0)
y2 = (df['Class'] == 2) * 1 # medium students and others(output 1 if M, otherwise 0)
y3 = (df['Class'] == 0) * 1 # high stuents and others(output 1 if H, otherwise 0)

# desired outputs for the test part in three class of students 
o1 = (Y_test == 1) * 1 
o2 = (Y_test == 2) * 1 
o3 = (Y_test == 0) * 1 


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
"Logistic Regression is a Machine Learning classification algorithm that is used"
"to predict the probability of a categorical dependent variable."
# Forward propagation: we will implement linear multiplication

#this function will calculate Z which determines the output of a TLU
def linear_mult(X, w, b): #formula number (1)
    return np.dot(w.T, X) + b  # Z = wTx +b


# we implement a function to generate W and b

def initialize_with_zeros(dim):
    
    w = np.zeros((dim, 1))
    b = 0
# assert statement has a condition or expression which is supposed to be always true. 
# If the condition is false assert halts the program and gives an AssertionError
    assert(w.shape == (dim, 1)) #make sure w is in shape (dim,1)
    assert(isinstance(b, float) or isinstance(b, int)) #b is either an integer or float

    return w, b

# Next we implement sigmoid function to map calculated value to a probablity:
def sigmoid(z): #formula number (2)
    return 1 / (1 + np.exp(-z))

# Now we implement the cost function, which represents the difference between our
# predictions and actual labels(y is the actual label and a is our predicted label):
def cost_function(y, a): #cost is the difference between predicted values and real values
    return -np.mean(y*np.log(a) + (1-y)*np.log(1-a))


"Now we implement the whole forward propagation"
# which will calculate cost and the predicted value for the each data point:
def forward_propagate(w, b, X, Y):

    Z = linear_mult(X, w, b) #calculate Z
    
    A = sigmoid(Z) #calculate A
    
    cost = cost_function(Y, A)
    cost = np.squeeze(cost)

    assert(cost.shape == ())
# we create this dictionary to use A in backward propagation 
    back_require = {
        'A': A
    }

    return back_require, cost


# Backward propagation: Now we calculate W and b derivative as follow
def backward_propagate(w, b, X, Y, back_require):

    m = X.shape[1]

    A = back_require['A']
# calculate the derivations
    dw = (1/m) * np.dot(X, (A-Y).T) # formula number (4)
    db = (1/m) * np.sum(A - Y) # formulanumber (5)

    assert(dw.shape == w.shape) #make sure of the types of derivations
    assert(db.dtype == float) 
# we make this dictionary to use the derivations in optimize function 
    grads = {"dw": dw,
             "db": db}

    return grads

# Complete propagation function to use forward & backward propagation respectively
def propagate(w, b, X, Y):

    # FORWARD PROPAGATION
    back_require, cost = forward_propagate(w, b, X, Y)

    # BACKWARD PROPAGATION
    grads = backward_propagate(w, b, X, Y, back_require)

    return grads, cost



# Combining All together:
# Now we combine all our implemented functions together to create an
# optimizer which can find a linear function to devide the zero labeled data from one
# labeled data points by optimizng W and b
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w -= learning_rate*dw # updating w & b with gradient decsent
        b -= learning_rate*db

        # Record the costs
        if i % 100 == 0: #append the cost to costs each 100 times
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Training the model on the whole dataset
def train_Model1():
    X_t, Y1_t = np.array(X.T), np.array(y1.T)
    w, b = initialize_with_zeros(14)
    params, grads, costs = optimize(
        w, b, X_t, Y1_t, num_iterations=1000, learning_rate=0.1, print_cost=False)
    return params['w'],params['b']

def train_Model2():
    X_t, Y2_t = np.array(X.T), np.array(y2.T)
    w, b = initialize_with_zeros(14)
    params, grads, costs = optimize(
        w, b, X_t, Y2_t, num_iterations=1000, learning_rate=0.4, print_cost=False)
    return params['w'],params['b']

def train_Model3():
    X_t, Y3_t = np.array(X.T), np.array(y3.T)
    w, b = initialize_with_zeros(14)
    params, grads, costs = optimize(
        w, b, X_t, Y3_t, num_iterations=1000, learning_rate=0.1, print_cost=False)
    return params['w'],params['b']


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Evaluating the model
# Prediction
def predict(w, b, X):
    
    m = X.shape[1] # m is the total number of data which is 480 in this case
    Y_prediction = np.zeros((1,m)) #initialize with an array of zeros
    Z = linear_mult(X, w, b)  
    A = sigmoid(Z)
    for i in range(m):
        Y_prediction[0][i] = 1 if A[0][i] > .5 else 0 #based on the sigmoid function
   
    assert(Y_prediction.shape == (1, m))




    return Y_prediction
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Evaluation of the data by the test data
weight,threshold = train_Model1()
X_transpose = X_test.T
preds = predict(weight, threshold, X_transpose) #check the predicted Y  with desired Y
print("   predicted Y1", "   real Y1")
for i in range(len(o1)):print(i,"  ",int(preds[0][i]),"         ",int(o1[i]))
print('Accuracy on training set: %{}'.format((preds[0] == o1).mean()*100)) 

weight,threshold = train_Model2()
X_transpose = X_test.T
preds = predict(weight, threshold, X_transpose) #check the predicted Y  with desired Y
print("   predicted Y2", "   real Y2")
for i in range(len(o2)):print(i,"  ",int(preds[0][i]),"         ",int(o2[i]))
print('Accuracy on training set: %{}'.format((preds[0] == o2).mean()*100)) 

weight,threshold = train_Model3()
X_transpose = X_test.T
preds = predict(weight, threshold, X_transpose) #check the predicted Y  with desired Y
print("   predicted Y3", "   real Y3")
for i in range(len(o3)):print(i,"  ",int(preds[0][i]),"         ",int(o3[i]))
print('Accuracy on training set: %{}'.format((preds[0] == o3).mean()*100)) 