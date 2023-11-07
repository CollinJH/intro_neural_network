# we will build a neuron using NumPy

import matplotlib.pyplot as plt
import numpy as np

def mse_loss(y_true, y_pred):
    # y_true and y_pred and numpy arrays of same length
    return ((y_true - y_pred) ** 2).mean()

def sigmoid(x):
    # this is our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # derivative of the sigmoid f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def feedforward(self, inputs):
        # weight inputs, add bias, and use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 0 

n = Neuron(weights, bias)

x = np.array([2, 3]) # x1 = 2, x2 = 3

# NEW CODE STARTS HERE

class OurNeuralNetwork:
    '''
    neural network with
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
    each neuron has the same wights and bias:
    - w = [0, 1]
    - b = 0

    not optimal code what so ever, only to be considered for education purposes :D
    '''
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
        
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()    
    
    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

        return o1 

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n is # of samples in dataset
        - all_y_trues is a numpy array with n elements
        Elements in all_y_trues correspond to those in data
        '''
        learn_rate = 0.1
        epochs = 1000 # number of times to loop through dataset
        losses = []

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # feed forward
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * x[1] + self.b3
                o1 = sigmoid(sum_o1)

                y_pred = o1

                # calcluating partials
                # d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- update weights and biases
                # Neruon h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                # calculate total loss at the end of each epoch

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
                losses.append((epoch, loss))

        return losses

        
    def plot_loss(self, losses):
        # unpack list of losses
        epochs, loss_values = zip(*losses)
        plt.plot(epochs, loss_values, marker='o', linestyle='-', color='blue')
        plt.title('Loss over epochs')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()

# dataset

data = np.array([
    [-2, 1],
    [25, 6],
    [17, 4],
    [-15, 6],
    
])

all_y_trues = np.array([
    1,
    0,
    0,
    1,
])


# train neural network
network = OurNeuralNetwork()
loss_data = network.train(data, all_y_trues)



network.plot_loss(loss_data)






    


