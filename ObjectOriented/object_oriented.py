# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<Mingyan Zhao>
<Math 321>
<09/18/2018>
"""
import math

class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        color (str): the color of the backpack
        contents (list): the contents of the backpack.
        max_size (integer): the maximum for number of the content
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        """Set the name and initialize an empty list of contents.

        name (str): the name of the backpack's owner.
        color (str): the color of the backpack
        contents (list): the contents of the backpack.
        max_size (integer): the maximum for number of the content
        """
        self.name = name
        self.color = color
        self.max_size = max_size
        self.contents = []
        
        

    def put(self, item):
        """Add an item to the backpack's list of contents if the size is 
        not over the compacity. If it dose gose ove, prints "no room!"."""
        if len(self.contents) < self.max_size:
            self.contents.append(item)
        else:
            print("No Room!")

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)
        
    def dump(self):
        """Resets the contents of the backpack to an empty list"""
        self.contents = []
	 	  
    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)
    
    def __eq__(self, other):
        """
        determine if two objects are equal(same name, color, and number of contents)
        """
        if self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents):
            return True
        else:
            return False
        
    def __str__(self):
        Owner = str("Owner:\t   ")+self.name+str("\n")
        Color = str("Color:\t   ")+self.color+str("\n")
        Size = str("size:\t   ")+str(len(self.contents))+str("\n")
        Maxsize = str("Max size:  ") + str(self.max_size)+str("\n")
        Contents = str("contents:  ") + str(self.contents)+str("\n")
        return Owner + Color + Size + Maxsize + Contents
        


# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True
        self.max_size=max_size

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)

#Problem 1
        
def test_backpack():
    
	 testpack = Backpack("Barry", "black")
	 if testpack.name != "Barry":
	 	 print("Backpack.name assigned incorrectly")
	 for items in ["pencil","pen", "paper", "computer"]:
	    testpack.put(items)
	 print("Contents: ", testpack.contents)
     
     
		 
# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.

class Jetpack(Backpack):
    """A Jetpack object class. Inherits from the Backpack class.
    A Jetpack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the Jetpack's owner.
        color (str): the color of the Jetpack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the Jetpack is tied shut.
    """
    def __init__(self, name, color, max_size = 2, amount_fuel = 10):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A Jetpack only holds 2 item by default.

        Parameters:
            name (str): the name of the Jetpack's owner.
            color (str): the color of the Jetpack.
            max_size (int): the maximum number of items that can fit inside.
            amount_fuel(float): the amount of fuel
        """
        Backpack.__init__(self, name, color, max_size)
        self.amount_fuel = amount_fuel

    def fly(self, fuel):
        """accepts an amount of fuel to be burned and decrements the fuel 
        attribute by that amount
        """
        
        if fuel <= self.amount_fuel:
            self.amount_fuel = self.amount_fuel - fuel
        else:
            
            print("Not enough fuel!")
            
        
    
    def dump(self):
        """Resets the contents of the backpack to an empty list and amount of 
        fuel to zero"""
        self.contents = []
        self.amount_fuel = 0



# Problem 4: Write a 'ComplexNumber' class.


class ComplexNumber:
    
    def __init__(self, real, imag):
        """
        Set the real and imaginary part of the complex number.
        """
        self.real = float(real)
        self.imag = float(imag)
        
    def conjugate(self):        
        return ComplexNumber(self.real, -self.imag)

    def __str__(self):
        if self.imag % 1 == 0:
            self.imag = int(self.imag)
        if self.real % 1 == 0:
            self.real = int(self.real)
            
        if self.imag >= 0:
            return "("+str(self.real)+"+"+str(self.imag)+"j)"
        else:
            return "("+str(self.real)+str(self.imag)+"j)"
        
    def __abs__(self):        
        return math.sqrt(self.real**2 + self.imag**2)
    
    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag
    
    def __add__(self, other):
        return ComplexNumber(self.real + other.real,self.imag + other.imag)
     
    def __sub__(self, other):
        return ComplexNumber(self.real - other.real,
                       self.imag - other.imag)

    def __mul__(self, other):
        return ComplexNumber(self.real*other.real - self.imag*other.imag,
                       self.imag*other.real + self.real*other.imag)

    def __truediv__(self, other):
        
        r = float((other.real**2 + other.imag**2))       
        return ComplexNumber((self.real*other.real + self.imag*other.imag)/r, (self.imag*other.real - self.real*other.imag)/r)

def test_ComplexNumber(a,b):
    py_cnum, my_cnum = complex(a, b), ComplexNumber(a, b)
   
    if my_cnum.real !=a or my_cnum.imag !=b:
        print("__int__() set self.real and self.imag incorrectly")
        
    if py_cnum.conjugate().imag != my_cnum.conjugate().imag:
        print("conjugate() failed for", py_cnum)
           
    if str(py_cnum) != str(my_cnum):
        print("__str__() failed")
    """     
    if complex(a,b)/complex(b,a) != my_cnum.__truediv__(ComplexNumber(b,a)):
        print(complex(a,b)/complex(b,a))
        print(my_cnum.__truediv__(ComplexNumber(b,a)))
        print("fail")
    """   
        
