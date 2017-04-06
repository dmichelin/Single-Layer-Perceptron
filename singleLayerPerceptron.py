from random import choice
from numpy import array,dot,random

# Step function
unit_step = lambda x: 0 if x < 0 else 1
def train(eta, n, training_data):
    errors = []


    w = random.rand(3)
    for i in xrange(n):
        #Choose a random input from the training data
        x, expected = choice(training_data)
        #Come up with the dot product of the weights and the data
        result = dot(w,x)

        # Calculate the error
        error = expected - unit_step(result)
        # Add the errors to the list
        errors.append(error)
        #Add the error to the rate times the learning rate
        w += eta * error * x
    return w,errors

def predict(testing_data, w):
    for x, _ in testing_data:
        result = dot(x, w)
        print("{}: {} -> {}".format(x[:2], result, unit_step(result)))


training_data = [
    (array([0,0,0]), 0),
    (array([0,1,0]), 0),
    (array([1,0,0]), 0),
    (array([1,1,1]), 1),
]




#List of errors to plot later on
errors = []

#initialize the learning rate
eta = 0.2

#Learning iterations
n = 100

weights,errors = train(eta,n,training_data)
print weights
predict(training_data,weights)

