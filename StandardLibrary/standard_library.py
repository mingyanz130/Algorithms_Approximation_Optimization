# standard_library.py
"""Python Essentials: The Standard Library.
<Mingyan Zhao>
<Math 321>
<09/06/2018>
"""
from itertools import chain, combinations
import calculator as ca
import box
import random
import time
import sys

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """

    return min(L),max(L),sum(L)/len(L)


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    int_1 = 3
    int_2 = int_1
    int_2 = 4

    if int_1 == int_2:
        print ("int is mutable")
    else:
        print ("int is immutable")

    str_1 = "a"
    str_2 = str_1
    str_2 = "b"
    if str_1 == str_2:
        print ("str is mutable")
    else:
        print ("str is immutable")

    list_1 = [4,5,6]
    list_2 = list_1
    list_2[0] = 1
    if list_1 == list_2:
        print ("list is mutable")
    else:
        print ("list is immutable")

    tuple_1 = ("a","b")
    tuple_2 = tuple_1
    tuple_2 += (1, )
    if tuple_1 == tuple_2:
        print ("tuple is mutable")
    else:
        print ("tuple is immutable")

    set_1 = {"a","b","c"}
    set_2 = set_1
    set_2.add('d')
    if set_1 == set_2:
        print ("set is mutable")
    else:
        print ("set is immutable")


# Problem 3

def hypot(a, b):

    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than those that are imported from your
    'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    # c = sqrt(a^2 + b^2) = sqrt((a+b)^2-2ab)
    return ca.sqrt(ca.Sum(a,b)**2 - 2* ca.Prod(a,b))

# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    tmp = list(chain.from_iterable(combinations(A,r) for r in range(len(A)+1)))
    return [set(i) for i in tmp]


# Problem 5: Implement shut the box.


def shut_the_box(name, total_time):

    """
    if len(sys.argv) != 3:
	     raise ValueError("You need at least three Arguements")

    name = sys.argv[1]
    total_time = float(sys.argv[2]
    """

    number_left = list(range(1,10))
    dice_1 = list(range(1,7))
    dice_2 = list(range(1,7))
    first_roll = int(random.choice(dice_1)) + int(random.choice(dice_2))
    start_time = time.time()

    time_left = round(total_time, 2)

    number_picked = ""

    success = True

    while success == True:
    #box.isvalid(random.choice(dice_1) + random.choice(dice_1), number_left):
        print("Number left: ", number_left)
        if max(number_left) > 6:
            roll = int(random.choice(dice_1)) + int(random.choice(dice_1))
        else:
            roll = int(random.choice(dice_1))
        print("Roll: ", roll)

        if success != box.isvalid(roll, number_left):
            print("Game over!")
            print("Score for player ", name, ": ", sum(number_left), "points")
            end_time = time.time()
            time_spent = round((end_time - start_time),2)
            print("Time played: ", time_spent, "seconds")
            print("Better luck next time >:)")
            success = False
        else:
            end_time = time.time()
            time_spent = (end_time - start_time)
            time_left = round(total_time - time_spent,2)
            if time_left <= 0:
                print("Game over!")
                print("Score for player ", name, ": ", sum(number_left), "points")
                end_time = time.time()
                time_spent = round((end_time - start_time),2)
                print("Time played: ", time_spent, "seconds")
                print("Better luck next time >:)")
                success = False
            else:
                print("Second left: ", time_left)
                number_picked = input("Number to eliminate: ")
                print("\n")


                if sum(box.parse_input(number_picked, number_left )) == roll:
                    for x in box.parse_input(number_picked, number_left ):
                        number_left.remove(x)

                else:
                    print("invalid input")
                    success = True
        if len(number_left) == 0:
            print("Score for player ", name,": ", sum(number_left), "points")
            end_time = time.time()
            time_spent = round((end_time - start_time),2)
            print("Time played: ", time_spent, "seconds")
            print("Congratulations! You shut the box!")
            success = False

if __name__ == "__main__":
    if len(sys.argv) == 3:
        name = str(sys.argv[1])
        total_time = (float(sys.argv[2]))
        shut_the_box(name,total_time)
