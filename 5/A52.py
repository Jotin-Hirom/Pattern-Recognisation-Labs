'''
Bayes maximum likelihood rule classifier for two class
ALGORITHM: Bayes Maximum Likelihood Classifier
INPUT:
- Training dataset {(X_i, y_i)}, where y_i ∈ {0, 1}
- Test samples X_test
OUTPUT:
- Predicted class labels y_pred ∈ {0, 1}
PROCEDURE:
1. Separate training data by class
    X₀ = {X_i | y_i = 0} // All samples from class 0
    X₁ = {X_i | y_i = 1} // All samples from class 1
2. Compute maximum likelihood estimates of parameters
    a. For class 0:
    μ₀ = (1/|X₀|) * ∑ x, for all x ∈ X₀ // Sample mean
    Σ₀ = (1/|X₀|) * ∑ (x - μ₀)(x - μ₀)ᵀ, for all x ∈ X₀ // Sample covariance
    b. For class 1:
    μ₁ = (1/|X₁|) * ∑ x, for all x ∈ X₁ // Sample mean
    Σ₁ = (1/|X₁|) * ∑ (x - μ₁)(x - μ₁)ᵀ, for all x ∈ X₁ // Sample covariance

3. For each test sample x in X_test:
    a. Calculate likelihood for each class assuming Gaussian distribution
        P(x|C₀) = (1/√((2π)^d |Σ₀|)) * exp(-(1/2)(x - μ₀)ᵀΣ₀⁻¹(x - μ₀))
        P(x|C₁) = (1/√((2π)^d |Σ₁|)) * exp(-(1/2)(x - μ₁)ᵀΣ₁⁻¹(x - μ₁))
        Where:
        - d is the dimensionality of the feature space
        - |Σ| denotes the determinant of matrix Σ
        - Σ⁻¹ denotes the inverse of matrix Σ
    b. Apply maximum likelihood decision rule
        If P(x|C₁) > P(x|C₀) then
            Assign x to class 1 (y_pred = 1)
        Else
            Assign x to class 0 (y_pred = 0)
        RETURN: y_pred
'''

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def bayes_maximum_likelihood(X_train, y_train, X_test):
    # Separate data by class
    class0 = X_train[y_train==0]
    class1 = X_train[y_train==1]
    # print(class0,class1)
    
    # Compute maximum likelihood estimates
    # For Gaussian, ML estimates are sample mean and sample covariance
    class0Mean = np.mean(class0, axis=0)
    class1Mean = np.mean(class1, axis=0)
    # print(class0Mean,class1Mean)
    class0Cov = np.cov(class0,rowvar=False)
    class1Cov = np.cov(class1,rowvar=False)
    # print(class0Cov,class1Cov)
    
    # Create probability distributions
    class0pdf = multivariate_normal(class0Mean,class0Cov)
    class1pdf = multivariate_normal(class1Mean,class1Cov)
    # print(class0pdf,class1pdf)
    for i,x in enumerate(X_test):
        # Calculate likelihoods P(x|class) likelihood by using Gaussian distribution 
        class0Likelihood = class0pdf.pdf(x) 
        class1Likelihood = class1pdf.pdf(x)
        # print(class0Likelihood,class0Likelihood)
        
        y_pred = np.zeros(len(X_test))
        # Assign to class with maximum likelihood
        if class1Likelihood>class0Likelihood :
            y_pred[i] = 1
        else:
            y_pred[i] = 0
        
    return y_pred
    
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    # Class 0: 100 samples
    mean0 = [0, 1]
    cov0 = [[1, 0.2], [0.2, 1]]
    X0 = np.random.multivariate_normal(mean0, cov0, 100)
    # Class 1: 100 samples
    mean1 = [2, 3]
    cov1 = [[1, -0.2], [-0.2, 1]]
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
    y_pred = bayes_maximum_likelihood(X_train, y_train, X_test)
    # print(y_pred)
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
    plt.suptitle("Bayes Maximum Likelihood Classifier Rule for two class", fontsize=16)
    plt.title(f"Accuracy: {accuracy:.2f}",fontsize=13)
    plt.legend(loc="best",fontsize=10, framealpha=1)
    plt.xlabel("Feature 1",fontsize=10)
    plt.ylabel("Feature 2",fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()