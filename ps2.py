import numpy as np
import torch
from torch import nn
import torch.optim as optim


################################ BEGIN NUMPY STARTER CODE #################################################
def sigmoid(x):
    #Numerically stable sigmoid function.
    #Taken from: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)


def sample_logistic_distribution(x,a):
    #np.random.seed(1)
    num_samples = len(x)
    y = np.empty(num_samples)
    for i in range(num_samples):
        y[i] = np.random.binomial(1,logistic_positive_prob(x[i],a))
    return y

def create_input_values(dim,num_samples):
    #np.random.seed(100)
    x_inputs = []
    for i in range(num_samples):
        x = 10*np.random.rand(dim)-5
        x_inputs.append(x)
    return x_inputs


def create_dataset():
    x= create_input_values(2,100)
    a=np.array([12,12])
    y=sample_logistic_distribution(x,a)

    return x,y

################################ END NUMPY STARTER CODE ####################################################



################################ BEGIN PYTORCH STARTER CODE ################################################

class TorchLogisticClassifier(nn.Module):

  def __init__(self, num_features):
    super().__init__()
    self.weights = nn.Parameter(torch.zeros(num_features))

  def forward(self, x_vector):
    logit = torch.dot(self.weights, x_vector)
    prob = torch.sigmoid(logit)
    return prob


def loss_fn(y_predicted, y_observed):
    return -1 * (y_observed * torch.log(y_predicted)
                 + (1 - y_observed) * torch.log(1 - y_predicted))

def extract_num_features(dataset):
    first_example = dataset[0]
    # first_example is a pair (x,y), where x is a vector of features and y is 0 or 1
    # note that both x and y are torch tensors
    first_example_x = first_example[0]
    first_example_y = first_example[1]
    num_features = first_example_x.size(0)
    return num_features

def nonbatched_gradient_descent(dataset, num_epochs=10, learning_rate=0.01):
    num_features = extract_num_features(dataset)
    model = TorchLogisticClassifier(num_features)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(num_epochs):
        for d_x, d_y in dataset:
            optimizer.zero_grad()
            prediction = model(d_x)
            loss = loss_fn(prediction, d_y)
            loss.backward()
            optimizer.step()
    return model

def generate_nonbatched_data(num_features=3, num_examples=100):
    x_vectors = [torch.randn(num_features) for _ in range(num_examples)]
    prob_val = 0.5 * torch.ones(1)
    y_vectors = [torch.bernoulli(prob_val) for _ in range(num_examples)]

    dataset = list(zip(x_vectors, y_vectors))

    return dataset

def main():
    nonbatched_dataset = generate_nonbatched_data()
    nonbatched_gradient_descent(nonbatched_dataset)

################################ END PYTORCH STARTER CODE ###################################################


# NOTICE: DO NOT EDIT FUNCTION SIGNATURES 
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES


# PROBLEM 1
def logistic_positive_prob(x,a):
    return sigmoid(np.dot(a,x))

# PROBLEM 2
def logistic_derivative_per_datapoint(y_i,x_i,a,j):
    return -1*(y_i - logistic_positive_prob(x_i, a))*x_i[j]

# PROBLEM 3
def logistic_partial_derivative(y,x,a,j):
    return (sum(logistic_derivative_per_datapoint(y[i],x[i],a,j) for i in range(len(y))))/len(y)

# PROBLEM 4
def compute_logistic_gradient(a,y,x):
    gradient = np.zeros(len(a))
    
    for j in range(len(a)):
        gradient[j] = logistic_partial_derivative(y, x, a, j)
    
    return gradient

# PROBLEM 5
def gradient_update(a,lr,gradient):
    return a - lr * gradient

# PROBLEM 6
def gradient_descent_logistic(initial_a,lr,num_iterations,y,x):
    for _ in range(num_iterations):
        initial_a = gradient_update(initial_a,lr,compute_logistic_gradient(initial_a,y,x))
    return initial_a

# PROBLEM 7
# Free Response Answer Here: 
# Line 78 is an example of when __init__ is called. __init__ is a function that initializes a 
# class with the neccessary parameters. In this instance the init function takes in the number
# of features and initializes the weights for each feature that the model will optimize.  

# PROBLEM 8
# Free Response Answer Here: 
# Line 83 is where forward is called. Forward is a function that computes the logistic function, 
# which is the sigmoid of the logit, which is the linear combination in the form of a dot product
# of the the weights and the input vector.

# PROBLEM 9
def batched_gradient_descent(dataset, num_epochs=10, learning_rate=0.01, batch_size=2):
    num_features = extract_num_features(dataset)
    model = TorchLogisticClassifier(num_features)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    num_batches = len(dataset) // batch_size

    for _ in range(num_epochs):
        for i in range(num_batches):

            batch = dataset[i * batch_size:(i + 1) * batch_size]
            # wait am I supposed to zero the gradient before each batch
            optimizer.zero_grad()
            tot_loss = 0

            for d_x, d_y in batch:

                prediction = model(d_x)
                loss = loss_fn(prediction, d_y)
                tot_loss += loss
            
            tot_loss /= batch_size
            tot_loss.backward()
            
            optimizer.step()

    return model


# PROBLEMS 10-12
def split_into_batches(dataset, batch_size):
    pass

def alt_gradient_descent(dataset, num_epochs=10, learning_rate=0.01, batch_size=2):
    num_features = extract_num_features(dataset)
    model = TorchLogisticClassifier(num_features)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    batches = split_into_batches(dataset, batch_size)
    for i in range(num_epochs):
        # optimizer.zero_grad() # 1
        for batch in batches:
            # optimizer.zero_grad() # 2
            for d_x, d_y in batch:
                # optimizer.zero_grad() # 3
                prediction = model(d_x)
                loss = loss_fn(prediction, d_y)
                loss.backward()
                # optimizer.step() # C
            # optimizer.step() # B
        # optimizer.step() # A
    return model

# PROBLEM 10
# Free Response Answer Here: 
# In this iternation, the line 1 (optimizer.zero_grad) is called in the beginning of each epoch
# before the code loops through each of the batches. For each batch the model calculates
# how confidient it is in its prediction and how much loss there is in those predictions.
# Then  loss.backwards() backproprogates adds up the graident for that batch. Once
# all the graidents for all the batches have been calculated. Line A (optimizer.step())
# is called where the gradients that were previously calculated are then used to update the 
# weights. In other words our weights are updated once per epoch after the graidents for 
# each batch has been computed. 
#  $\nabla_{\vec{w}}L(\vec{w}|B) := \sum_{d \in B} \nabla_{\vec{w}}L(\vec{w}|d)$

# PROBLEM 11
# In this iternation, there optimizer.zero_grad() is never called meaning that our 
# gradient values are never set back to zero. Since the graidents are never set back to zero,
# our gradients continue to accumlate giving us an innacurate gradient decsent steps since 
# later updates in the weights are created from the sum of gradients from ealier states
# of the model's graidents. 
#  $\nabla_{\vec{w}} L_{\text{total}} = \sum_{e=1}^{t} \sum_{d \in D} \nabla_{\vec{w}^{(e)}} L\left( \vec{w}^{(e)} \mid d \right)$

# PROBLEM 12
# 2B is the correct answer. In order to make both gradient descents equivalent, 
# the main changes are with regards to where the gradient is set to 0 and where
# the updates using the step take place. The gradients must be set to 0 for 
# each individual batch, so that the accumulated gradients from previous 
# batches do not carry over. However, the weights are actively being updated 
# once for each batch, so the step must take place within each individual 
# batch. Therefore, 2B is the optimal answer, with the arguments for 
# alt_gradient_descent being set to (dataset, learning_rate=lr, batch_size=k),
# so that the arguments are the exact same.
