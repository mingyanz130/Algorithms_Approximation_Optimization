# binary_trees.py
"""Volume 2: Binary Trees.
<Mingyan Zhao>
<Math 321>
<10/04/2018>
"""

# These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import numpy as np
import time
import random


class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        
        
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the list.")
            if data == current.value:               # Base case 2: data found!
                return current
            else:                                   # Recursively search next.
                return _step(current.next)

        # Start the recursion on the head of the list.
        return _step(self.head)
        
        
class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        n = BSTNode(data)
        #base case: the tree is empty      
        if self.root == None:
            self.root = n
            return
        
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            #add the node at the end of the tree
            if current is None: 
                p = current.prev
                if p.left == current:
                    p.left = n
                    n.prev = p
                else:
                    p.right = n
                    n.prev = p  
            elif data == current.value:                     
                raise ValueError("The element is already in the tree.")
            
            else:
                #go to left if it is smaller than the parent
                if data < current.value:
                    if current.left is None:
                        current.left = n
                        n.prev = current
                    else:
                        return _step(current.left)
                #go to right if it is greater than the parent
                elif data > current.value:               
                    if current.right is None:
                        current.right = n
                        n.prev = current
                    else:
                        return _step(current.right)
                    
        # Start the recursion on the root of the tree.
        return _step(self.root)
            

    # Problem 3    
    def remove(self, data):
        """
        Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
            """
    
        #find the Node
        n = self.find(data)
                
        #removing the root node
        if n is self.root:
            #remove a leaf node
            if n.right is None and n.left is None:
                self.root = None
            #Removing a Node with one child
            elif n.right is None and n.left is not None:
                q = n.left
                n = None
                self.root = q
            elif n.left is None and n.right is not None:
                q = n.right
                n = None
                self.root = q
            #Removing a Node with two children   
            elif n.left is not None and n.right is not None:
                p = n.left
                x=0
                while p.right is not None:
                    p = p.right
                    x= 1
                if p.left is None:
                    if x != 1:
                        n.value = p.value
                        p.prev.left = None
                    else:
                        n.value = p.value
                        p.prev.right = None
                        
                else:
                    n.value = p.value
                    q = p.left
                    m = p.prev
                    p = None
                    m.right = q
                    q.prev = m    
                    
        #remove a leaf node
        else:
            if n.right is None and n.left is None:
                if n.prev.right == n:
                    n.prev.right = None
                elif n.prev.left == n:
                    n.prev.left = None
            #Removing a Node with one child
            elif n.right is None and n.left is not None:
                if n.prev.right == n:
                    p = n.prev
                    q = n.left
                    n = None
                    p.right = q
                    q.prev = p
                elif n.prev.left == n:
                    p = n.prev
                    q = n.left
                    n = None
                    p.left = q
                    q.prev = p
                
            elif n.left is None and n.right is not None:
                if n.prev.right == n:
                    p = n.prev
                    q = n.right
                    n = None
                    p.right = q
                    q.prev = p
                elif n.prev.left == n:
                    p = n.prev
                    q = n.right
                    n = None
                    p.left = q
                    q.prev = p
            #Removing a Node with two children
            elif n.left is not None and n.right is not None:
                p = n.left
                x = 0
                while p.right is not None:
                    p = p.right
                    x= 1
                if p.left is None:
                    if x != 1:
                        n.value = p.value
                        p.prev.left = None
                    else:
                        n.value = p.value
                        p.prev.right = None
                else:
                    n.value = p.value
                    q = p.left
                    m = p.prev
                    p = None
                    m.right = q
                    q.prev = m                
                    
        
        
            
            
        

    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    #initialize th etime
    times1 = []
    times2 = []
    times3 = []
    times4 = []
    times5 = []
    times6 = []
    #set the domain
    domain = 2**np.arange(3,11)
    #read the file
    with open("english.txt", 'r') as file:
        contents= file.readlines()
    
    #save the time with each function    
    for n in domain:            
        subset = random.sample(contents, n)
        subset_find = random.sample(subset, 5)
        
        #inserting for LinkedList
        start1 = time.clock()
        A = SinglyLinkedList()
        for i in subset:
            A.append(i)    
        times1.append(time.clock() -start1)
        #inserting for BST
        start2 = time.clock()
        B = BST()
    
        for i in subset:
            B.insert(i)
        times2.append(time.clock() -start2)
        #inserting for AVL
        start3 = time.clock()
        C = AVL()
        for i in subset:
            C.insert(i)
        times3.append(time.clock() -start3)
        #finding for LinkedList
        start4 = time.clock()
        
        for i in subset_find:
            A.iterative_find(i)    
        times4.append(time.clock() -start4)
        
        #finding for BTS
        start5 = time.clock()
        for i in subset_find:
            B.find(i)
        times5.append(time.clock() -start5)
        #finding for AVL
        start6 = time.clock()
        for i in subset_find:
            C.find(i)
        times6.append(time.clock() -start6)
        
    
    #graph each function and label them
    a = plt.subplot(121)
    a.loglog(domain, times1, 'g.-', basex = 2, basey = 10, linewidth = 2, markersize = 15, label = "LinkedList")
    a.loglog(domain, times2, 'b.-', basex = 2, basey = 10, linewidth = 2, markersize = 15, label = "BST")
    a.loglog(domain, times3, 'c.-', basex = 2, basey = 10, linewidth = 2, markersize = 15, label = "AVL")
    a.legend(loc = "upper left")
    plt.title('Insert')
    plt.xlabel('n')
    plt.ylabel('time')
    b = plt.subplot(122)
    b.loglog(domain, times4, 'g.-', basex = 2, basey = 10, linewidth = 2, markersize = 15, label = "linkedList")
    b.loglog(domain, times5, 'b.-', basex = 2, basey = 10, linewidth = 2, markersize = 15, label = "BST")
    b.loglog(domain, times6, 'c.-', basex = 2, basey = 10, linewidth = 2, markersize = 15, label = "AVL")
    b.legend(loc = "upper left")
    plt.title('find')
    plt.xlabel('n')
    plt.ylabel('time')
    plt.show()
    

