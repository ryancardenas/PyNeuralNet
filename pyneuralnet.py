import numpy as np
from IPython.display import clear_output

def loadCSVData(filename, labelcol=-1, usecols=None):
    '''
    DESCRIPTION:
    Loads data from CSV file into two arrays: one containing features and the other
    containing labels.

    PARAMETERS:
    filename -- (str) Location of CSV file to be processed.
    labelcol -- (int) Index of column containing labels.
    usecols -- (int, tuple) Indices of columns to be extracted from CSV.

    RETURNS:
    xdat -- [m,n](numpy array) Feature data.
    ydat -- [n,p](numpy array) Label data.
    '''
    data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=usecols)
    np.random.shuffle(data)

    xdat = np.copy(data[:])
    xdat = np.delete(xdat, labelcol, axis=1)
    ydat = np.copy(data[:,labelcol])
    m = np.shape(ydat)[0]
    ydat = ydat.reshape(m,1)

    return xdat, ydat


def update_progress(progress, msg=''):
    '''
    DESCRIPTION:
    Displays progress bar and percent completion.

    PARAMETERS:
    progress -- (float) Decimal in range [0,1] representing percent completion.
    msg -- (str) Optional message to display with progress bar.
    '''
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    text += "\n" + msg
    print(text)


def arrayMemorySize(x, base=10):
    '''
    DESCRIPTION:
    Prints memory size of a numpy array.

    PARAMETERS:
    x -- (numpy array)
    base -- (int) Specifies base10 (for SI) or base2 (for IEC) computation.
    '''
    if base == 2:
        units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
        for i in range(5):
            if round(x.nbytes/2**(i * 10), 2) > 1:
                unit = units[i]
                power = i * 10

    else:
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        for i in range(5):
            if round(x.nbytes/10**(i * 3), 2) > 1:
                unit = units[i]
                power = i * 3

    print(np.shape(x), 'array uses', round(x.nbytes/base**power, 2), unit)


class Layer(object):
    def __init__(self, num_prev, num_neurons, activation_function='sigmoid'):
        '''
        Initialize weights randomly with He scaling.

        self.A -- [num_neurons, 1] Array of activations
        self.Z -- [num_neurons, 1] Array of linearly aggregated inputs
        self.W -- [num_neurons, num_prev] Array of weights applied to previous layer's activations A.
        self.b -- [num_neurons, 1] Array of biases
        '''
        self.n = num_neurons
        self.act = activation_function

        self.A = np.zeros((num_neurons,1))
        self.Z = np.zeros((num_neurons,1))
        self.b = np.zeros((self.n,1))

        # He initialization.
        if self.act == 'sigmoid':
            self.W = np.random.randn(num_neurons, num_prev) * np.sqrt(2.0/num_prev)

    def forward(self, A_prev):
        '''
        Description:
        Perform forward pass by computing Z, then A.

        Parameters:
        A_prev -- [num_prev, m_samples] Array of activations from previous layer.

        Computes:
        self.A -- [self.n, m_samples] Array of activations for this layer.
        '''
        self.Z = self.W @ A_prev + self.b

        if self.act == 'linear':
            self.A = self.Z

        elif self.act == 'sigmoid':
            self.A = 1 / (1 + np.exp(-self.Z))

        return self.A

    def backward(self, W_next, dZ_next, A_prev):
        '''
        Description:
        Compute dA first, then dZ, then dW and db.

        Parameters:
        W_next -- [num_next, self.n] Array of weights between this layer and next layer.
        dZ_next -- [num_next, m_samples] Array of linear inputs to next layer.
        A_prev -- [num_prev,m_samples] Array of activations from previous layer.

        Computes:
        self.dA -- [self.n, m_samples] Gradient of cost w.r.t. activations.
        self.dZ -- [self.n, m_samples] Gradient of cost w.r.t. linear inputs.
        self.dW -- [self.n, num_prev] Gradient of cost w.r.t. weights.
        self.db -- [self.n, 1] Gradient of cost w.r.t. biases.
        '''
        m = self.A.shape[1]
        self.dA = W_next.T @ dZ_next

        if self.act == 'linear':
            self.dZ = self.dA

        elif self.act == 'sigmoid':
            self.dZ = self.A * (1 - self.A) * self.dA

        self.dW = 1 / m * self.dZ @ A_prev.T
        self.db = 1 / m * np.sum(self.dZ, axis=1, keepdims=True)

    def update(self, learning_rate):
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db


def buildNetwork(layout, num_features):
    '''
    Arguments:
    layout -- (num_layers, 2) Tuple where first column contains number of neurons in each layer
                and second column contains activation functions for each layer.

    Returns:
    network -- (num_layers,:) List of layers, each with number of neurons specified in 'network'.
    '''
    network = []
    network.append(Layer(num_features, layout[0][0], layout[0][1]))
    print('Layer 1:', layout[0])

    for i in range(1, len(layout)):
        num_prev = layout[i-1][0]
        num_neurons = layout[i][0]
        activation_function = layout[i][1]
        network.append(Layer(num_prev, num_neurons, activation_function))
        print('Layer', i+1, '\b:', layout[i])

    return network
