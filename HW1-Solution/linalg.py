import numpy as np
from numpy import linalg as LA
   # Data to be used in the program
    '''
    Commented so they doesnt mix with other parameters
    M  = np.array([[1,2,3],[4,5,6],[7,8,9], [10,11,12]])
    
    row_vec = np.array([1,1,10])
    a = row_vec[np.newaxis, :]

    col_vec = np.array([-1,2,5])
    b = col_vec[: , np.newaxis]
    '''

def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    Args:
        a: numpy array of shape (x, n)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1) """

    out = None
    
    def dot_product(x, y):
        dp = 0
        for i in range(len(x)):
            dp += (x[i]*y[i])
        return dp
 

    x = dot_product(a.T,b)

    out = x
    
    return out


def complicated_matrix_function(M, a, b):
    """Implement (a * b) * (M * a.T).

    (optional): Use the `dot_product(a, b)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (1, n).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """

    d1 = np.matmul(a,b)
    d2 = np.matmul(M,a.T)
    d3 = np.matmul(d2,d1)
    print(d3.shape)
    
    out = d3.shape
   
    return out


def svd(M):
    """Implement Singular Value Decomposition.

    (optional): Look up `np.linalg` library online for a list of
    helper functions that you might find useful.

    Args:
        M: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """
    
    u = None
    s = None
    # new matrix as in 1.4
    #M  = np.array([[1,2,3],[4,5,6],[7,8,9], [10,11,12]])

    u,s,v = np.linalg.svd(M)

    return u, s, v


def get_singular_values(M, k):
    """Return top n singular values of matrix.

    (optional): Use the `svd(M)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (m, n).
        k: number of singular values to output.

    Returns:
        singular_values: array of shape (k)
    """
    singular_values = None
     
        # new matrix M
    #M = np.array([[1,2,3],[4,5,6],[7,8,9], [10,11,12]])

    u,s,v= np.linalg.svd(M,1)

    for i  in range(k):
        print(s[i])
        singular_values = s[i]    
    
    return singular_values


def eigen_decomp(M):
    """Implement eigenvalue decomposition.
    
    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    w = None
    v = None

    w,v = LA.eig(np.array([[1,2,3],[4,5,6],[7,8,9]]))

    return w, v


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """
    eigenvalues = []
    eigenvectors = []
      
    w,v = LA.eig(M)
    for i in range(k):
        print("EigenVal = ",w[i])
        print("EIgenVec = ",v[i])
        eigenvalues.append(w[i])
        eigenvectors.append(v[i])
        
    return eigenvalues, eigenvectors
