# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<Mingyan Zhao>
<Math 321>
<10/23/2018>
"""

import numpy as np
from scipy import linalg as la
from scipy import stats
from scipy.spatial import KDTree
from matplotlib import pyplot as plt



# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    #calculating the difference between each row of X and z
    A = X- np.vstack(([z]*X.shape[0]))
    B = la.norm(A, axis=1)
    return X[np.argmin(B)], min(B)

# Problem 2: Write a KDTNode class.

class KDTNode:
    def __init__(self,data):
        if type(data)  is not np.ndarray:
            raise TypeError("Input needs to be a nparray.")
        self.value = data
        self.right = None
        self.left = None
        self.pivot = None
        
        

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
        """
        x = KDTNode(data)
        if self.root is None:
            
            self.root = x
            self.root.pivot = 0
            self.k = len(data)
        else:
            if len(data) != self.k:
                raise ValueError(str(data) + "is not in R^" + str(self.k) )
            def _step(current):
                """Recursively step through the tree until finding the node
                containing the data. If there is no such node, raise a ValueError.
                """ 
                if np.allclose(data, current.value): # Base case 1: same data.
                    raise ValueError(str(data) + " is already in the tree")
                
                # Base case 2: dead end!                    
                elif data[current.pivot] < current.value[current.pivot]:
                    if current.left is None:
                        current.left = x
                        x.pivot = current.pivot + 1
                        if x.pivot == self.k:
                            x.pivot = 0
                    else:
                        return _step(current.left)          # Recursively search left.
                else:
                    if current.right is None:
                        current.right = x
                        x.pivot = current.pivot + 1
                        if x.pivot == self.k:
                            x.pivot = 0
                    else:
                        return _step(current.right)         # Recursively search right.

                # Start the recursive search at the root of the tree.
            return _step(self.root)
            
            
    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        if len(z) != self.k:
                raise ValueError(str(z) + "is not in R^" + str(self.k))
                
        def KDSearch(current, nearest, d_):
            if current is None:
                return nearest, d_
            x = current.value
            i = current.pivot
            if la.norm(x-z) < d_:
                nearest = current
                d_ = la.norm(x-z)
            if z[i] < x[i]:
                nearest, d_ = KDSearch(current.left, nearest, d_)
                if z[i] + d_ >= x[i]:
                    nearest, d_ = KDSearch(current.right, nearest, d_)
            else:
                nearest, d_ = KDSearch(current.right, nearest, d_)
                if z[i] - d_ <= x[i]:
                    nearest, d_ = KDSearch(current.left, nearest,d_)                    
            return nearest, d_
        
        node, d_ = KDSearch(self.root, self.root, la.norm(self.root.value - z))
        
        return node.value, d_
        
        
    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    def __init__(self, n_neighbors):
        """Initialize the root and k attributes."""
        self.k = n_neighbors
        self.labels = None
        self.tree = None
        
    def fit(self, X, y):
        
        self.tree = KDTree(X)
        self.labels =y
    def predict(self, z):
        distance, indices = self.tree.query(z, k=self.k)
        
        return stats.mode(self.labels[indices])[0][0]
        
        

# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(np.float)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float)
    y_test = data["y_test"]
    
    a = KNeighborsClassifier(n_neighbors)
    a.fit(X_train, y_train)
    
    matches = 0
    for i in range(len(y_test)):
        if y_test[i] == a.predict(X_test[i]):
            matches += 1
    return matches/len(y_test)

    """
    plt.imshow(X_test[0].reshape((28,28)), cmap="gray")
    plt.show()  
    """
    
    
    