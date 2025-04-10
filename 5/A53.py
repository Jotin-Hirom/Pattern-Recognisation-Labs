'''
minimum distance classifier for two class
ALGORITHM: Minimum Distance Classifier for Two Classes
INPUT:
- Training dataset {(X_i, y_i)}, where y_i ∈ {0, 1}
- Test samples X_test
OUTPUT:
- Predicted class labels y_pred ∈ {0, 1}
PROCEDURE:
1. Compute class centroids (prototypes)
a. For class 0:
μ₀ = (1/|X₀|) * ∑ x, for all x ∈ {X_i | y_i = 0}
b. For class 1:
μ₁ = (1/|X₁|) * ∑ x, for all x ∈ {X_i | y_i = 1}
2. For each test sample x in X_test:
a. Calculate Euclidean distance to each centroid
d₀ = ||x - μ₀|| = √[(x - μ₀)ᵀ(x - μ₀)]
d₁ = ||x - μ₁|| = √[(x - μ₁)ᵀ(x - μ₁)]
b. Apply minimum distance decision rule
If d₁ < d₀ then
Assign x to class 1 (y_pred = 1)
Else
Assign x to class 0 (y_pred = 0)
RETURN: y_pred
'''

import numpy as np
import matplotlib.pyplot as plt

def minimum_distance_classifier(X_train, y_train, X_test):
    # Minimum Distance Classifier for two classes.
    # Compute class prototypes (centroids)
    C0 = X_train[y_train==0]
    C1 = X_train[y_train==1]
    C0Mean = np.mean(C0,axis=0)
    C1Mean = np.mean(C1,axis=0)
    # print(C0Mean,C1Mean)
    
    y_pred = np.zeros(len(X_test))
    for i,x in enumerate(X_test):      
        # Classify test points based on minimum Euclidean distance
        C0ED = np.sqrt(np.sum((x - C0Mean)**2)) #||x^2 - mean|| = sqrt(x^T-mean . X-mean)
        # C1ED = np.sqrt(np.sum((x-C1Mean)**2)) 
        C1ED = np.sqrt((x - C1Mean).T @ (x - C1Mean))
        # print(C0ED)
        # print(C1ED)
        
        # Assign to class with minimum distance
        if C0ED>C1ED:
            y_pred[i] = 1
        else:
            y_pred[i] = 1
    return y_pred
    pass
    
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    # Class 0: 100 samples
    mean0 = [0, 1]
    cov0 = [[1, 0], [0, 1]]
    X0 = np.random.multivariate_normal(mean0, cov0, 100)
    # Class 1: 100 samples
    mean1 = [3, 3]
    cov1 = [[1, 0], [0, 1]]
    X1 = np.random.multivariate_normal(mean1, cov1, 100)
    # Combine datasets
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(100), np.ones(100)))
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    # Split into train/test (80/20)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    # Apply the classifier
    y_pred = minimum_distance_classifier(X_train, y_train, X_test)
    # Calculate accuracy
    accuracy = np.mean(y_pred==y_test)
    # print(f"Accuracy: {accuracy:.2f}")
    # print(X_test[y_pred==1,0])
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot test data with trains
    plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1],c="blue",alpha=0.5,s=40,label="Class 0 (train)" )
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
                color='red', alpha=0.5,s=40, label='Class 1 (train)')
    
    # Plot test data with predictions
    plt.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], 
                color='cyan', marker='x',s=60, label='Class 0 (pred)')
    plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], 
                color='magenta', marker='x',s=60, label='Class 1 (pred)')
        # Highlight misclassified points
    misclassified = X_test[y_pred != y_test]
    if len(misclassified) > 0:
        plt.scatter(misclassified[:, 0], misclassified[:, 1],
                   color='black', marker='o', s=120,
                   linewidths=2, facecolors='none',
                   label='Misclassified')
    plt.suptitle("Minimum Distance Classifier for Two Classes", fontsize=16)
    plt.title(f"Accuracy: {accuracy:.2f}",fontsize=13)
    plt.legend(loc="best",fontsize=10, framealpha=1)
    plt.xlabel("Feature 1",fontsize=10)
    plt.ylabel("Feature 2",fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()