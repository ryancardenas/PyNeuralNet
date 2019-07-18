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
