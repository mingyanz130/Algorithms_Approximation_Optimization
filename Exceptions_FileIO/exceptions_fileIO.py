# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Mingyan Zhao>
<Math 321>
<09/26/2018>
"""

from random import choice


# Problem 1
def arithmagic():
    """
    This function verifies the user's input at each step. Raise A ValueError \
    an informative error message.
    """
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    #Enter a 3-digit number where the first and last digits differ by 2 or more
    if len(step_1) != 3 or abs(int(step_1[0])-int(step_1[-1])) < 2:
        raise ValueError("you have to enter a 3-digit number where the first \
                         and last digits differ by 2 or more")
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    # Enter the reverse of the first number
    if step_2 != step_1[::-1]:
        raise ValueError("You need to enter the reverse of the first number, \
                         obtained by reading it backwards")
        
    step_3 = input("Enter the positive difference of these numbers: ")
    #Enter the positive difference of these numers.
    if int(step_3) != abs(int(step_1)-int(step_2)):
        raise ValueError("You need to enter the positive difference of \
                         these numbers")
    step_4 = input("Enter the reverse of the previous result: ")
    
    #Enter the reverse of the previous result.
    if step_3 != step_4[::-1]:
        raise ValueError("You need to enter the reverse of the previous result")
        
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
    """
    this algorithm will keep going until keyboard interrupt, we will record 
    the iteration when it happens
    """
    walk = 0
    directions = [1, -1]
    #function starts
    try:
        for i in range(int(max_iters)):
            walk += choice(directions)
            
    #raise a keyboardinterrupt error by pressing ctrl+c   
    except KeyboardInterrupt as e:
        print("Process interrupped at iteration ", i)
            
    else:
        print("Process completed")
    finally:    
        return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter:
    """
    define a class called ContentFilter. the parameters are filename and 
    contents, it will open a file and stroe the file as its contents.
    """
    
    def __init__(self, file_name):
        
        self.file_name = file_name
        success = True
        # if the filename is invalid in any way, prompt the user for another filename
        while success == True:
            
            try:
                file = open(file_name,"r")         
                success = False
                self.contents = "".join(file.readlines())
                
            #It will raise the errror if the input for filename does nort exist       
            except:
                file_name = input("Please enter a valid file name: ")
                success == True
    #write the data to the outfile with uniform case        
    def uniform(self, new_file ,mode = "w", case = "upper"):
        """ read the original file and write a new file and make it all upper case or lower case 
        depending on user's choice, it will also raise a error if the choice is wrong
        """
        #raise a ValueError whne the mode input is wrong
        if mode != 'w' and mode != 'x' and mode != 'a':
                raise ValueError("The mode is wrong")
        else:       
            with open(new_file, mode) as myfile:
                
                #when case is upper, write a newfile with all uppercase 
                if case == "upper":
                    reverse = self.contents.upper().split('\n')
                    result = ""
                    for i in range(len(reverse)-1):
                        result = result + ''.join(reverse[i])+"\n"
                        
                    result = result + ''.join(reverse[len(reverse)-1])
                    
                    myfile.write(result)
                #when case is lower, write a newfile with all lowercase 
                elif case == "lower":
                    reverse = self.contents.lower().split('\n')
                    result = ""
                    for i in range(len(reverse)-1):
                        result = result + ''.join(reverse[i])+"\n"
                        
                    result = result + ''.join(reverse[len(reverse)-1])
                    myfile.write(result)
                else:
                    #raise a ValueError
                    raise ValueError("Case needs to be upper or lower")
    #write the data to the outfile inreverse order    
    def reverse(self, new_file, mode = 'w', unit = "line"):
        """read the original file and write a new file and reverse it by line or words
        depending on user's choice, it will also raise a error if the choice is wrong
        """
        if mode != 'w' and mode != 'x' and mode != 'a':
                raise ValueError("The mode is wrong")
        else:       
            with open(new_file, mode) as myfile:
                
                #when the unit is word, reverse the word order on each line
                if unit == "word":
                    reverse = self.contents.split('\n')
                    result = ""
                    for i in range(len(reverse)-1):
                        reverse[i] = reverse[i].split(' ')[::-1]
                        result = result + ' '.join(reverse[i])+ "\n"
                    
                    reverse[len(reverse)-1] = reverse[len(reverse)-1].split(' ')[::-1]
                    result = result + ' '.join(reverse[len(reverse)-1])    
                    
                    myfile.write(result)
                #when the unit is line, reverse the line order
                elif unit == "line":
                    reverse = self.contents.split('\n')[:-1]          
                    result = ""
                    reverse1 = reverse[::-1]
                    
                    for i in range(len(reverse)-1):
                        result = result + ''.join(reverse1[i])+"\n"
                        
                    result = result + ''.join(reverse1[len(reverse)-1]) + "\n"  
                    myfile.write(result)
                else:
                    #raise a ValueError
                    raise ValueError("Unit needs to be word or line")
                    
    #write a transpose version of the data to the outfile             
    def transpose(self, new_file, mode = "w"):
        
        """read the original file and write a new file and transpose it
        depending on user's choice, it will also raise a error if the choice is wrong
        """
        if mode != 'w' and mode != 'x' and mode != 'a':
                raise ValueError("The mode is wrong")
        else:       
            with open(new_file, mode) as myfile:
                
                row = self.contents.split('\n')[:-1]
                
                column_num = len(row[0].split(' '))
                
        #write jth word of each row for jth row 
                for j in range(column_num):
                    result = ""
                    for i in range(len(row)):
                        column = row[i].split(' ')
                        
                        result = result + str(column[j]) +" "
                    result = result +"\n"
                    myfile.write(result)     
                    
                    
                   
                    
                    
    
    def __str__(self):
        lines=0
        words=0
        characters=0
        for line in str(self.contents):
            wordslist=line.split()
            lines=lines+1
            words=words+len(wordslist)
            characters += sum(len(word) for word in wordslist) 
            
        num_lines = len(self.contents.split('\n')) - 1
        letters = 0
        digit = 0
        space = 0
        other = 0
       
        for i in self.contents:
            if i.isalpha():
                letters+=1
            elif i.isdigit():
                digit+=1
            elif i.isspace():
                space+=1
            else:
                other+=1
                
        Sourcefile = str("Source file:\t\t")+self.file_name +str("\n")
        Total_characters = str("Total characters:\t") + str(characters)+str("\n")        
        Alphabetic_characters = str("Alphabetic characters:  ")+str(letters)+str("\n")
        Numerical_characters = str("Numerical characters:   ") + str(digit)+str("\n")
        Whitespace_characters = str("Whitespace characters:  ") + str(space)+str("\n")
        Number_of_lines = str("Number of lines:\t") + str(num_lines) + str("\n")
        
        return Sourcefile + Total_characters + Alphabetic_characters + Numerical_characters + Whitespace_characters + Number_of_lines 
        
                
    
