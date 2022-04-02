# Slide 34
import sys, os
from mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
##
np.set_printoptions(linewidth=115)

class Network:
    def __init__(self):
        with open("./sample_weight.pkl", 'rb') as f:
            network = pickle.load(f)

        self.W1, self.W2, self.W3 = network['W1'], network['W2'], network['W3']
        self.b1, self.b2, self.b3 = network['b1'], network['b2'], network['b3']

    def forward(self, x):
        h = np.dot(x, self.W1) + self.b1
        h = self.sigmoid(h)

        k = np.dot(h, self.W2) + self.b2
        k = self.sigmoid(k)

        y = np.dot(k, self.W3) + self.b3
        print(y)
        output = self.softmax(y)

        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, y):
        exp_y = np.exp(y)
        sum_exp_y = np.sum(exp_y, axis=1)
        print("sum: ", sum_exp_y)

        output = np.transpose(exp_y) / sum_exp_y
        return np.transpose(output)

###
def criterion(y, t):
    return 0.5 * np.sum(np.power((y-t), 2), axis=1)

def cross_entropy(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) / batch_size

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    # input = np.array([[3, 9]])
    ##
    input = x_test[0:1]
    label = t_test[0:1]
    ##
    print(input)
    print("label: ", label)
    NN = Network()

    output = NN.forward(input)
    ##
    print("probability: ", np.round(output, 3))

    predicted_index = np.argmax(output, axis=1)
    ##
    one_hot_predicted = np.eye(10)[predicted_index]
    ##
    print("prediction of model: ", predicted_index)
    print("one_hot_predicted: ", one_hot_predicted)

    ##
    #loss = criterion(output, label)
    #print("loss: ", loss)
    cross_entropy = cross_entropy(output, label)
    print("cross_entropy: ", cross_entropy)

    comparison_gt = (label == predicted_index)
    correct_cnt = np.sum(comparison_gt)
    accuracy = correct_cnt/len(label)

    print("output: ", output)
    print("output: ", predicted_index)
    print("Accuracy: ", round(accuracy, 3))