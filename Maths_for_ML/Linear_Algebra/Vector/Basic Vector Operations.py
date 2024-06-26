import numpy as np

#Addition 
num1 = np.array([3,4,5])
num2 = np.array([10,11,12])

def addition(a, b):
    total = a + b
    return total

print(addition(num1, num2))


#Subtraction 

def subtraction(a, b):
    total = a - b
    return total

print(subtraction(num1, num2))


#division

def division(a, b):
    total = a / b
    return total

print(division(num1, num2))


#Scalar Multiplication

num3 = 3
num4 = np.array([5,9,5,6])

def scalar_multiplication(a, b):
    total = a * b
    return total

print(scalar_multiplication(num3, num4))






#dot Product

def dot_product(a, b):
    total = np.dot(a,b)
    return total

print(dot_product(num1, num2))





#cross Product


def cross_product(a, b):
    total = np.cross(a, b)
    return total

print (cross_product(num1, num2))


#magnitude 


def magnitude(a):
    result = np.linalg.norm(a)
    return result

print(magnitude(num1))


#unit Vector

def unit_vector(a):
    magnitude = np.linalg.norm(a)
    unit_vec = a/magnitude
    return unit_vec
print(unit_vector(num1))
    

#angle between two vectors


def angle_between_two_vectors(a, b):
    magnitude1  = np.linalg.norm(a)
    magnitude2  = np.linalg.norm(b)
    product = dot_product(a, b)
    angle_between = np.arccos(product/(magnitude1* magnitude2))
    return angle_between
    
print(angle_between_two_vectors(num1, num2))

