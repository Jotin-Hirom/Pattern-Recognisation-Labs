import numpy as np

"""
Assignment 3: NumPy Operations

This script performs various matrix operations using NumPy, including:
1. Matrix operations (transpose, determinant, trace)
2. Matrix addition, subtraction, multiplication
3. Solving a system of linear equations
4. Calculating mean, standard deviation, and correlation matrix
5. Eigenvalues and eigenvectors
6. Covariance matrix calculation
7. Correlation matrix calculation
"""

# Question 1: Matrix operations
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Transpose:\n", A.T)  # Transpose of the matrix
print("Determinant:", np.linalg.det(A))  # Determinant of the matrix
print("Trace:", np.trace(A))  # Trace (sum of diagonal elements)

# Question 2: Matrix addition, subtraction, multiplication
A = np.array([[3, 1], [2, 4]])
B = np.array([[2, 5], [1, 3]])
print("A + B:\n", A + B)  # Element-wise addition
print("A - B:\n", A - B)  # Element-wise subtraction
print("A × B (matrix product):\n", A @ B)  # Matrix multiplication
print("A * B (element-wise):\n", A * B)  # Element-wise multiplication

# Question 3: Solve Linear Equations
C = np.array([[2, 1, 1], [3, 2, 3], [1, 4, 9]])  # Coefficients matrix
d = np.array([10, 18, 16])  # Constants
x = np.linalg.solve(C, d)  # Solving for x
print("Solution:", x)

# Question 4: Mean, Standard Deviation, Correlation
A = np.array([[5.1, 3.5, 1.4], [4.9, 3.0, 1.4], [4.7, 3.2, 1.3], [4.6, 3.1, 1.5], [5.0, 3.6, 1.4]])
print("Means:", np.mean(A, axis=0))  # Mean of each column
print("Std deviations:", np.std(A, axis=0))  # Standard deviation of each column
print("Correlation matrix:\n", np.corrcoef(A, rowvar=False))  # Correlation between columns

# Question 5: Eigenvalues & Eigenvectors
A = np.array([[4, 2, 2], [2, 5, 1], [2, 1, 6]])
values, vectors = np.linalg.eig(A)  # Eigen decomposition
print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)

# Question 6: Covariance Matrix
X = np.array([[1, 2], [2, 3], [3, 5], [4, 7], [5, 8]])
print("Covariance matrix:\n", np.cov(X, rowvar=False))  # Covariance between variables

# Question 7: Correlation Matrix
corr_matrix = np.corrcoef(X, rowvar=False)  # Correlation matrix
print("Correlation matrix:\n", corr_matrix)
 


'''
PS D:\Pattern Recognisation Lab> python -u "d:\Pattern Recognisation Lab\Mar_Lab\3\ASIGN3.py"
Transpose:
 [[1 4 7]
 [2 5 8]
 [3 6 9]]
Determinant: 6.66133814775094e-16
Trace: 15
A + B:
 [[5 6]
 [3 7]]
A - B:
 [[ 1 -4]
 [ 1  1]]
A × B (matrix product):
 [[ 7 18]
 [ 8 22]]
A * B (element-wise):
 [[ 6  5]
 [ 2 12]]
Solution: [ 7. -9.  5.]
Means: [4.86 3.28 1.4 ]
Std deviations: [0.18547237 0.23151674 0.06324555]
Correlation matrix:
 [[ 1.          0.68001929 -0.17049858]
 [ 0.68001929  1.         -0.13658959]
 [-0.17049858 -0.13658959  1.        ]]
Eigenvalues: [8.38761906 2.12592447 4.48645647]
Eigenvectors:
 [[ 0.53867823  0.82803335 -0.15552024]
 [ 0.51488378 -0.46965459 -0.71716055]
 [ 0.66687365 -0.30624393  0.67933364]]
Covariance matrix:
 [[2.5 4. ]
 [4.  6.5]]
Correlation matrix:
 [[1.         0.99227788]
 [0.99227788 1.        ]]
'''