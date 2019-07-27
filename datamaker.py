import csv
import numpy as np

def load_planar_dataset():
    '''
    CREDIT: Andrew Ng of deeplearning.ai. Taken from course "Neural Networks and
    Deep Learning, Week 3: Planar data classification with a hidden layer."

    DESCRIPTION:
    Creates 2D data of 400 samples in the shape of an eight-pedal flower.
    Binary classification.
    '''
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

def makeFakeData_LinearBoundary(f, a, b):
    '''
    DESCRIPTION:
    Creates a dataset with a linear decision boundary along.

    PARAMETERS:
    f -- (str) Filename.
    a -- (float) Coefficient of X.
    b -- (float) Y-intercept.

    RETURNS:
    None
    '''
    with open(f, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(50):
            for j in range(20):
                x1 = round(float(np.random.randn(1)), 2)
                x2 = round(float(np.random.randn(1)), 2)
                y = 1*((a2*x2) > (a1*x1 + b))
                inp = [x1, x2, y]
#                 print(inp)
                writer.writerow(inp)

    csvFile.close()

def makeFakeData_EllipticalBoundary(f, a, b):
    '''
    DESCRIPTION:
    Creates a dataset with an elliptical decision boundary along.

    PARAMETERS:
    f -- (str) Filename.
    a -- (float) Vertical axis.
    b -- (float) Horizontal axis.

    RETURNS:
    None
    '''
    with open(f, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(50):
            for j in range(20):
                x1 = round(float(np.random.randn(1)), 2)
                x2 = round(float(np.random.randn(1)), 2)
                y = 1*np.sqrt((x1 / a)**2 + (x2 / b)**2 >= 1)
                inp = [x1, x2, y]
#                 print(inp)
                writer.writerow(inp)

    csvFile.close()

def makeFakeData_AnonymousBoundary(f, a, b, r, func=lambda x1, x2: a*x1 + b*x2):
    '''
    DESCRIPTION:
    Creates a dataset with an elliptical decision boundary along.

    PARAMETERS:
    f -- (str) Filename.
    a -- (float) Vertical axis.
    b -- (float) Horizontal axis.

    RETURNS:
    None
    '''
    with open(f, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for i in range(50):
            for j in range(20):
                x1 = round(float(np.random.randn(1)), 2)
                x2 = round(float(np.random.randn(1)), 2)
                y = 1*(func(x1, x2) == r)
                inp = [x1, x2, y]
#                 print(inp)
                writer.writerow(inp)

    csvFile.close()
