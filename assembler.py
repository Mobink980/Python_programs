import numpy as np
import math


 
    

# Python3 code to remove whitespace 
def remove_space(string): 
    return string.replace(" ", "")


def string_splitter(text):
    a = [] # for saving the 3 parts of the instruction
    str1 = text.split(',') #split at ','
    str2 = str1[0].split() #split at space

    a.append( remove_space(str2[0]) ) #first part of str2 is instruction
    a.append( remove_space(str2[1]) ) #second part of str2 is destination register
    a.append( remove_space(str1[1]) ) #second part of str1 which is after ',' is source register


    return a
    







def listToString(my_list):  
    """
    A function to convert a list to a string
    """
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for element in my_list:  
        str1 += element  
    
    # return string   
    return str1 

def assemble (instruction, addr):
    """
    Returns 6 bits string if the instruction is SUB or ADD
    Returns 12 bits string when the instruction is LOAD or JNZ
    When writing into the file, we split the 12 bits strings into half
    and we have a 6 bits string in each line of the text file
    ==> addr is the label address
    """

    assembled_string = list("000000")

    # We assume that if the string does not contain ',' then it is halt instruction
    if "," not in instruction:
        return "000000"


    # If the instruction contains ':' it means that we got a label
    if ":" in instruction:
        str1 = instruction.split(':') #split at ':'
        instruction = str1[1] # getting rid of the label
        
    

    parts = string_splitter(instruction)

    #Two first bits are opcode, thus we create the opcode based on the instruction
    if (parts[0].lower() == "load"):
        assembled_string[0] = '0'
        assembled_string[1] = '0'

    elif (parts[0].lower() == "add"):
        assembled_string[0] = '0'
        assembled_string[1] = '1'

    elif (parts[0].lower() == "sub"):
        assembled_string[0] = '1'
        assembled_string[1] = '0'

    else:
        assembled_string[0] = '1' #JNZ instruction
        assembled_string[1] = '1'


    #Two second bits represent the destination register
    if (parts[1].lower() == "r0"):
        assembled_string[2] = '0'
        assembled_string[3] = '0'

    elif (parts[1].lower() == "r1"):
        assembled_string[2] = '0'
        assembled_string[3] = '1'

    elif (parts[1].lower() == "r2"):
        assembled_string[2] = '1'
        assembled_string[3] = '0'

    else:
        assembled_string[2] = '1' #JNZ instruction
        assembled_string[3] = '1'


    #Two third bits represent the source register
    if (parts[2].lower() == "r0"):
        assembled_string[4] = '0'
        assembled_string[5] = '0'

    elif (parts[2].lower() == "r1"):
        assembled_string[4] = '0'
        assembled_string[5] = '1'

    elif (parts[2].lower() == "r2"):
        assembled_string[4] = '1'
        assembled_string[5] = '0'

    else:
        assembled_string[4] = '1' #JNZ instruction
        assembled_string[5] = '1'


    #If we have load or jnz instruction, source register is set to 11
    if (parts[0].lower() == "load" or parts[0].lower() == "jnz"):
        assembled_string[4] = '1'
        assembled_string[5] = '1'
        # the following cell of the load or JNZ instruction
        next_memory_cell = get_according_value(parts[2], addr)
        assembled_string.append(next_memory_cell)
    

    assembled_string = listToString(assembled_string)

    return assembled_string

#returns true if a string is integer 
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


# returns the binary number is string if the value is integer
# returns the address of the cell if the string is a label
def get_according_value(st, addr):

    if (RepresentsInt(st)):
        return ("{:06b}".format(int(st)))

    else:
        return ("{:06b}".format(addr))


    

# used to split a string in half
def split(s):
    half, rem = divmod(len(s), 2)
    return s[:half + rem], s[half + rem:]

frontA, backA = split('abcde')


# TODO
def write_into_file(machine_code):
    """
    A function to write the machine code into a text file.
    This function takes a list of machine code corresponding to 
    the assembly code and writes it line by line in a text file.
    """

    #This will return an error if the specified file exists
    f = open("memory.txt", "x") #create a text file by name assembly 

    for i in range (len(machine_code)):
        if (len(machine_code[i]) == 6):
            f.write(machine_code[i])
            f.write("\n")
        
        else:
            str_code = machine_code[i]
            first_part, second_part = split(str_code)
            f.write(first_part)
            f.write("\n")
            f.write(second_part)
            f.write("\n")
    
    f.close()



def read_from_file(filename):
    """
    A function to read the assembly code from a file.
    It take a file by its name, reads it line by line and 
    returns a list containing all the assembly instructions
    """
    # Using readline() 
    instructions = []
    file1 = open(filename, 'r') 
    count = 0
    
    while True: 
        count += 1
    
        # Get next line from file 
        line = file1.readline() 
    
        # if line is empty 
        # end of file is reached 
        if not line: 
            break
        instructions.append(line.strip())
    
    file1.close()

    return instructions


# gets the list of instructions and returns the address of the label in memory
def label_address_calculator(instructions):
    count = 0
    for i in range(len(instructions)):
        if (":" in instructions[i].lower()):
            break
            
        else:
            if ("load" in instructions[i].lower()):
                count += 2
            else:
                #the instruction is either Add or Sub
                count += 1

    return count



  

# list containing assembly instructions
instructions = read_from_file('myfile.txt')

# list containing the corresponding machine code of the assembly instructions
machine_code = []
label_address = label_address_calculator(instructions)
for i in range(len(instructions)):
    machine_code.append(assemble(instructions[i], label_address))


print (label_address)
print(instructions)
print(machine_code)

write_into_file(machine_code)

