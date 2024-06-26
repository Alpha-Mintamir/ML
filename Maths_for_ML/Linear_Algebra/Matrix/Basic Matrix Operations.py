
import numpy as np

num1 = np.array([[1,2], [3,5]])
num2 = np.array([[7,8], [9,10]])

#Addition 

def addition(a, b):
    result = a + b
    return result
print(addition(num1, num2))


#subtraction

def subtraction(a, b):
    result = a - b
    return result
print(subtraction(num1, num2))

#division

def division(a, b):
    result = a / b
    return result
print(division(num1, num2))

#scalar multiplication

def scalar_multiplication(a, b):
    result = a * b
    return result
print(scalar_multiplication(2, num2))


#matrix multiplication

def matrix_multiplication(a, b):
    result = np.dot(a,b)
    return result
print(matrix_multiplication(num1, num2))



#determinant


def determinant(a):
    result = np.linalg.det(a)
    return result
print(determinant(num1))


#Transpose

def transpose(a):
    result = a.T
    return result
print(transpose(num1))

#inverse

def inverse(a):
    result = np.linalg.inv(a)
    return result
print(inverse(num1))




