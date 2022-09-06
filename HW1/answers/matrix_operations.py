# -*- coding: utf-8 -*-
"""Matrix Operations.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mgr8SYC-6xIy4ikKV6IlkqC3tGtScX4f

# ADLA CSC522
#### HW1
#### Group: H21
#### Karan Gala 
#### Sahil Sawant
#### Akhil Namboodiri

# Generate a 5*5 identity matrix A.
"""

import numpy as np
A = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
print("A:",A)

"""# Change all elements in the 5th column of A to 5."""

for i in range(5):
    A[i][4] = 5
print("A:",A)

"""# Sum of all elements in the matrix (use ONE “for/while loop”)."""

sum = 0
for i in A:
    sum+=np.sum(i)
print("Sum of all elements in the matrix:",sum)

sum2 = np.sum(A)
print("Sum of all elements in the matrix:",sum2)

"""# Transpose the matrix A (A=AT )."""

AT = np.transpose(A)
print("Transpose of A:",AT)

AT = np.zeros((5,5))
for i in range(5):
  for j in range(5):
    AT[i][j] = A[j][i]
print("Transpose of A:",AT)

"""# Calculate the sum of the 5th row, the sum of the diagonal and the sum of the 1st column in matrix A, respectively (your answer should be three numbers). Use the transposed matrix from the previous part."""

sum5 = np.sum(AT[4])
print("Sum of 5th row:", sum5)
sum1 = 0
for i in range(5):
    sum1+=AT[i][0]
print("Sum of 1st column:", sum1)
sumD = 0
for i in range(5):
    sumD+=AT[i][i]
print("Sum of Diagonal elements:", sumD)

"""# Generate a 5*5 matrix B following standard normal distribution."""

B = np.random.normal(3,0.5,(5,5))
print("Matrix B:",B)

"""## From A and B, using matrix operations to get a new 2*5 matrix C such that, the first row of C is equal to the 1st row of B minus the 1st row of A, the second row of C is equal to the sum of the 5th row of A and the 5th row of B."""

C = np.empty(shape = (2,5))
C[0] = B[0] - A[0]
C[1] = B[4] + A[4]
print("Matrix C:",C)

"""# From C, using ONE matrix operation to get a new matrix D such that,the first column of D is equal to the first column of C, the second column of D is equal to the second column of C times 2, the third column of D is equal to the third column of C times 3, and so on."""

D = np.empty(shape = (5,2))
temp = np.transpose(C)
for i in range(5):
    D[i] = (i+1)*temp[i]
D = np.transpose(D)
print("D :",D)

"""# X = [1, 1, 1, 2]T , Y = [0, 3, 6, 9]T , Z = [4, 3, 2, 1]T . Compute the covariance matrix of X, Y, and Z. Then compute the Pearson correlation coefficients between X and Y."""

import numpy as np
X = np.array([[1],[1],[1],[2]])
Y = np.array([[0],[3],[6],[9]])
Z = np.array([[4],[3],[2],[1]])
covMatrix = np.cov(np.concatenate((X.T,Y.T,Z.T),axis = 0))
print("Covariance of matrix X,Y and Z:")
print(covMatrix)
print()
pc = np.corrcoef(X.T,Y.T)
print("Pearson Correlation Coefficients between X and Y:")
print(pc, pc[0][1])

"""# Verify the equation: ¯x 2 = (¯x2+σ2(x)) using x = [23, 19, 21, 22, 21, 23, 23, 20]T when (python library math is allowed): 
## i. σ(x) is the population standard  eviation. Show your work. 
## ii. σ(x) is the sample standard deviation. Show your work.
"""

x = (np.array([23,19,21,22,21,23,23,20]))
meanX = np.average(x)
import statistics
sdXSample = statistics.stdev(x)
sdXPopulation = statistics.pstdev(x)
print("Mean of X:", meanX)

popSD = 0
sampleSD = 0
n = len(x)
v = 0
v2 = 0
for i in x:
    v+=(meanX-i)*(meanX-i)/(n)
    v2+=(meanX-i)*(meanX-i)/(n-1)
popSD = v**0.5
sampleSD = v2**0.5
print("SD of population: \t",popSD)
print("SD of sample: \t\t",sampleSD)

meanXSq = meanX**2
print("Mean Squared with population standard deviation squared: ",meanXSq+popSD**2) #RHS
print("Mean Squared with population standard deviation squared: ",meanXSq+sampleSD**2)

print("Mean of sum of data points square: ",np.average(np.square(x))) #LHS

"""## The equation can be verified for the population standard deviation"""