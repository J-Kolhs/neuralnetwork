import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

# Use PCA
def getTransformedData(fname = 'train'):
    # Pre-processing of data
    with open(str(fname) + '.csv') as file:
        c = file.read()
    data = c.split('\n')
    
    headers = data[0].split(',')
    
    y_train = []
    x_train = []
    
    data = data[1:-1]
    
    # Shuffle data
    data = shuffle(data)
    
    N = len(data)
    
    
    extract_y = data[0].split(',')
    
    for i in range(N):
        extract = data[i].split(',')
        y_train.append(extract[0])
        x_train.append(extract[1:])
    
    x_train, y_train = np.array(x_train, dtype = 'int32'), np.array(y_train, dtype = 'int32')
    
    # Transformation of inputs
    mu = x_train.mean(axis = 0)
    x_train = x_train - mu
    pca = PCA()
    Z = pca.fit_transform(x_train)
    
    # Take the first 300 columns of Z
    x_train = Z[:, :300]
    
    # Normalize
    mu = x_train.mean(axis = 0)
    std = x_train.std(axis = 0)
    
    x_train = (x_train - mu) / std
    
    # Separate test set from training set
    x_test = x_train[-1000:]
    y_test = y_train[-1000:]
    x_train = x_train[:-1000]
    y_train = y_train[:-1000]
    
    
    return x_train, y_train, x_test, y_test

# Standard normalization
def getData(fname = 'train'):
    # Pre-processing of data
    with open(str(fname) + '.csv') as file:
        c = file.read()
    data = c.split('\n')
    
    headers = data[0].split(',')
    
    y_train = []
    x_train = []
    
    data = data[1:-1]
    
    # Shuffle data
    data = shuffle(data)
    
    N = len(data)
    
    
    extract_y = data[0].split(',')
    
    for i in range(N):
        extract = data[i].split(',')
        y_train.append(extract[0])
        x_train.append(extract[1:])
    
    x_train, y_train = np.array(x_train, dtype = 'int32'), np.array(y_train, dtype = 'int32')
    
    # Normalization of inputs
    x_train = x_train / 255
    
    # Separate test set from training set
    x_test = x_train[-1000:]
    y_test = y_train[-1000:]
    x_train = x_train[:-1000]
    y_train = y_train[:-1000]
    
    
    return x_train, y_train, x_test, y_test


class SoftmaxLogistic():
    
    def __init__(self, x_train, y_train):

        # Weights initialization
        K = len(set(y_train))
        N, D = x_train.shape
        W = np.random.rand(D, K)
        b = np.zeros(K)
        
        # Training indicator
        target = np.zeros((N, K))
        for j in range(N):
            target[j, y_train[j]] = 1
            
        # Define class variables
        self.N = N
        self.K = K
        self.D = D
        self.W = W
        self.b = b
        self.target = target
        

    
    def forward(self, x):
        
        a = x.dot(self.W) + self.b
        y = softmax(a)
        return y
        
    def train(self, x, y, lr):
        
        train_cost = []
        
        # Training loop
        for i in range(5000):
            
            # Get predicted output
            p_y = self.forward(x)
            
            # Get cost
            J = -(self.target * np.log(p_y)).sum()
        
            # Gradient descent   
            difference = self.target - p_y
            dJdW = x.T.dot(difference)
            self.W += lr * dJdW
            
            self.b += lr * difference.sum(axis=0)
            
            if i % 20 == 0:
                train_cost.append(J)
                print("i: ", i, "Cost: ", J)
        
        
        predictions = np.argmax(p_y, axis = 
                                1)
        score = np.mean(y == predictions)
        plt.plot(train_cost)
        plt.show()
        return J, score, predictions
    
    def test(self, x_test, y_test):
        
        p_y_test = self.forward(x_test)
        predictions = np.argmax(p_y_test, axis = 1)
        score = np.mean(y_test == predictions)
        return score, predictions

def main():
    # x, y, x_test, y_test = getData() 
    # Use the above if you don't want to have PCA
    x, y, x_test, y_test = getTransformedData()
    
    
    model = SoftmaxLogistic(x,y)
    p_y = model.forward(x)
    J, score, predictions = model.train(x, y, 0.000001)
    
    test_score, test_prediictions = model.test(x_test, y_test)
    print("\n Score on training set: ", score, "\n Score on test set: ", test_score)

if __name__ == '__main__':
    main()
