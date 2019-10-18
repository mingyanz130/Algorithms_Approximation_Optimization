# linked_lists.py
"""Volume 2: Linked Lists.
<Mingyan Zhao>
<Math 321>
<09/18/2018>
"""


# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute."""
        if type(data) != int and type(data) != float and type(data) != str:
            raise TypeError("The data type is wrong.")
        self.value = data
        


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None
        self.tail = None

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        if self.head is None or self.tail is None:
            raise ValueError("The node is empty")
        
        #always look for the value n, if the data is n, return n
        #if it is not the head, change the head to next value, untill find the value. if 
        #the head is the tal, then the data is not in the node
        
        n = self.head
        while True:    
            if data is n.value:
                return n
            else:
                n = n.next
            
            if n is self.tail:
                if data is n.value:
                    return n
                else:
                    raise ValueError("The data is not in the node")
            
           

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        if i < 0:
            raise IndexError("The index is out of range")
        
        n = self.head
        for j in range(i):
           
            if n is self.tail and j <= i-1:
                raise IndexError("The index is out of range")
            else:
             n = n.next
        return n    
        
            
            
    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        if self.head == None and self.tail == None:
            return 0
        if self.head == self.tail:
            return 1
        
        n = self.head
        i = 0
        while True:                
            n = n.next
            i += 1            
            if n is self.tail:
                return i + 1
               
        
    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        k = len(self)
        
        list1 = []
        if self.head is None:
            return repr(list1)
        else:
            n = self.head
            for i in range(k):             
                list1.append(n.value)
                if n is self.tail:
                    return repr(list1)
                else:
                    n = n.next
            
        return list1

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        d= self.find(data)
        
        if d is self.head and self.head is self.tail:
            self.head = None
            self.tail = None
        
        elif d is self.head:
            self.head = d.next
            d = None
        elif d is self.tail:
            self.tail = d.prev
            d = None
        else:
            d.prev.next = d.next
            d.next = d.prev.next
            d = None
           
        
        
        """
        self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
        """
        

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        new_node = LinkedListNode(data)
        n = self.head        
        if index == len(self):
            self.append(data)
        elif index < 0 or index > len(self):
            raise IndexError("The index is out of range")
        elif index == 0:
            # If the list is not empty, place new_node after the tail.
            self.head.prev = new_node               # tail --> new_node
            new_node.next = self.head               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.head = new_node
        else:
            n = self.get(index)
            A = n.prev
            A.next = new_node
            new_node.prev = A
            n.prev = new_node
            new_node.next = n
            


# Problem 6: Deque class.
class Deque(LinkedList):
    
    def __init__(self):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        LinkedList.__init__(self)       # Use inheritance to set self.value.
        
    def pop(self):
        if len(self) == 0 :
            raise ValueError("The list is empty.")
        elif len(self) == 1 :
            n = self.tail.value
            self.head = None
            self.tail = None
            return n
        else:
            n = self.tail
            m = n.value
            self.tail = n.prev
            n = None
            return m
            
    def popleft(self):
        if len(self) == 0 :
            raise ValueError("The list is empty.")
        elif len(self) == 1 :
            n = self.head.value
            self.head = None
            self.tail = None
            return n            
        else:
            n = self.head
            m = n.value
            self.head = n.next
            n = None
            return m
    def appendleft(self,data):
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node before the head.
            self.head.prev = new_node               # new_code --> head
            new_node.next = self.head               # new_code <-- head
            # Now the first node in the list is new_node, so reassign the head.
            self.head = new_node

    def remove(*args, **kwargs):
        raise NotImplementedError("Use pop() or popleft() for removal")
    
    def insert(*args, **kwargs):
        raise NotImplementedError("Use append() or appendleft() for inserting")
    

# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    #open the file and read it by lines
    a= Deque()
    with open(infile, 'r') as myfile:
        
        for line in myfile:
            a.append(line.strip())
        
    #create a new file and write down the elements pop out from the deque       
    with open(outfile, 'w') as file:
        
        for i in range(len(a)):
            file.write(a.pop() + "\n")