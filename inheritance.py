import numpy as np

# In this file we will investigate inheritance in Python.

# ========================= CONSTRUCTOR ============================
# In Python the __init__() method is called the constructor and is 
# always called when an object is created.
# Syntax of constructor declaration : 

# def __init__(self):
#     # body of the constructor

# ==================== default constructor =========================
# The default constructor is a simple constructor which doesn’t accept 
# any arguments. Its definition has only one argument which is a 
# reference to the instance being constructed.
class GeekforGeeks:
    # default constructor
    def __init__(self):
        self.geek = "GeekforGeeks"
    
    # a method for printing data members
    def print_Geek(self):
        print(self.geek)
 
 
# creating object of the class
obj = GeekforGeeks()
# calling the instance method using the object obj
obj.print_Geek()

# ==================== parameterized constructor ===================
# The parameterized constructor takes its first argument as a reference 
# to the instance being constructed known as self and the rest of the 
# arguments are provided by the programmer.
class Addition:
    first = 0
    second = 0
    answer = 0    
    # parameterized constructor
    def __init__(self, f, s):
        self.first = f
        self.second = s
    
    def display(self):
        print("First number = " + str(self.first))
        print("Second number = " + str(self.second))
        print("Addition of two numbers = " + str(self.answer))
 
    def calculate(self):
        self.answer = self.first + self.second
 
# creating object of the class
# this will invoke parameterized constructor
obj = Addition(1000, 2000)
# perform Addition
obj.calculate()
# display result
obj.display()

# ==================================================================
# ======================  Inheritance  =============================
# ==================================================================
# Create a class named Person, with firstname and lastname properties, 
# and a printname method:
class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)

# Create a class named Student, which will inherit the properties 
# and methods from the Person class:
class Student(Person):
  # Use the pass keyword when you do not want to add 
  # any other properties or methods to the class.   
  pass 

# Now the Student class has the same properties and methods 
# as the Person class.


class Employee(Person):
  def __init__(self, fname, lname, salary):
    Person.__init__(self, fname, lname) 
    # calling Person's constructor allows us to make use of the 
    # firstname and lastname properties directly from the Employee.
    self.salary = salary

  def print_employee_info(self):
    print(self.firstname, self.lastname, "has " + str(self.salary) 
    + " bitcoin annual salary.")

  def give_raise(self, amount):
    self.salary += amount



#Use the Person class to create an object, and then execute 
#the printname method:
x = Person("John", "Doe")
x.printname()

# Use the Student class to create an object, and then 
# execute the printname method:
x = Student("Mike", "Olsen")
x.printname()

general = Employee("Sonya", "Blade", 20)
commander = Employee("Cassie", "Cage", 15)
soldier = Employee("James", "Darling", 8)

general.print_employee_info()
commander.print_employee_info()
soldier.print_employee_info()

commander.give_raise(3)
print("After raise:")
commander.print_employee_info()




