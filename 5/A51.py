'''
 Apply Bayes Decision Rule for two-class problems 
 ALGORITHM: Bayes Decision Rule for Two-Class Problems
  P(ω|x) = [P(x|ω) * P(ω)] / P(x)
  Posterior = Likelihood * Prior/ Evidence
 INPUT:  
 - Training dataset {(X_i, y_i)}, where y_i ∈ {0, 1}  
 - Test sample X_test  - Prior probabilities P(C₀) and P(C₁)  
 
 OUTPUT:  
 - Predicted class label y_pred ∈ {0, 1} 
   
 PROCEDURE:  
 1. Separate training data by class  X₀ = {X_i | y_i = 0}  X₁ = {X_i | y_i = 1}  
 2. Estimate class parameters  
        μ₀ = mean(X₀)  
        μ₁ = mean(X₁)  
        Σ₀ = covariance(X₀)  
        Σ₁ = covariance(X₁)  
 3. For each test sample x in X_test:  
    a. Calculate likelihood for each class  
        P(x|C₀) = N(x; μ₀, Σ₀)  // Gaussian probability density  
        P(x|C₁) = N(x; μ₁, Σ₁)  
    b. Calculate posterior probabilities using Bayes' theorem  
        P(C₀|x) ∝ P(x|C₀) × P(C₀)  P(C₁|x) ∝ P(x|C₁) × P(C₁)  
    c. Apply decision rule  
        If P(C₁|x) > P(C₀|x) then  
            Assign x to class C₁ (y_pred = 1) 
        Else  
            Assign x to class C₀ (y_pred = 0)  
 
        RETURN: y_pred
'''
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import multivariate_normal 

def bayes_decision_rule(X_train, y_train, X_test, prior_0=0.5, prior_1=0.5): 
    # Separate the data by class  
    X0 = X_train[y_train == 0] #X₀ = {X_i | y_i = 0}
    X1 = X_train[y_train == 1] #X₁ = {X_i | y_i = 1} 
    # print(f"Class 0: \n {X0} \n")
    # print(f"Class 1: \n {X1} \n")
    
    # Calculate mean vectors  
    X0Mean = np.mean(X0, axis=0) #compute the mean vertically
    X1Mean = np.mean(X1, axis=0) #compute the mean vertically
    # print(X0Mean)
    # print(X1Mean)
     
    # Calculate covariance matrices
    X0COV = np.cov(X0, rowvar= False) 
    #If rowvar is True (default), then each row represents a variable, with observations in the columns. 
    # Otherwise, the relationship is transposed: each column represents a variable, 
    # while the rows contain observations.
    X1COV = np.cov(X1, rowvar= False)
    # print(X0COV)
    # print(X1COV) 
    
    # Create probability distribution for each class 
    X0PRO = multivariate_normal(X0Mean,X0COV)
    X1PRO = multivariate_normal(X1Mean,X1COV)
    # print(X0PRO)
    # print(X1PRO)
    
    y_pred = np.zeros(len(X_test))  # Initialize all the values as 0
    for i,x in enumerate(X_test): #enumerate: adds a counter to an iterable & returns it as an enumerate object
        # Calculate likelihoods for each test point  
        X0Likelihood = X0PRO.pdf(x) # pdf(Class, mean=None, cov=1, allow_singular=False)
        X1Likelihood = X1PRO.pdf(x) # pdf(Class, mean=None, cov=1, allow_singular=False)
        # print(X0Likelihood)
        # print(X1Likelihood)

        # Calculate posteriors using Bayes rule  
        X0POST = X0Likelihood * prior_0 #P(C₀|x) ∝ P(x|C₀) × P(C₀)
        X1POST = X1Likelihood * prior_1 #P(C₁|x) ∝ P(x|C₁) × P(C₁)
        # print(X0POST)
        # print(X1POST)
        
        # Classify based on highest posterior probability
        if X0POST > X1POST:
            y_pred[i]=0
        else:
            y_pred[i]=1
        # print(y_pred)
        # print(len(y_pred))
    return y_pred


if __name__ == "__main__": 
    # Generate some example data  
    np.random.seed(42) 
    
    # Class 0 data (100 points)  
    mean0 = [0, 0] #[x , y] 
    cov0 = [[1, 0], [0, 1]] 
    '''
    Cov(X,Y)=[ Var(X) Cov(Y,X) 
               Cov(X,Y) Var(Y)]
    '''
    class0_data = np.random.multivariate_normal(mean0, cov0, 100)  
    
    # Class 1 data (100 points)  
    mean1 = [3, 3]  #[x , y]
    cov1 = [[1, 0], [0, 1]]  
    class1_data = np.random.multivariate_normal(mean1, cov1, 100)
    
    # Combine data  
    y = np.hstack((np.zeros(100), np.ones(100)))  #hstack → horizontal (column-wise) stacking. np.concatenate(axis=1)
    X = np.vstack((class0_data, class1_data))  #vstack → vertical (row-wise) stacking. np.concatenate(axis=0)
    
    # Use first 80% for training, last 20% for testing  
    split = int(0.8 * len(X))  
    X_train, X_test = X[:split], X[split:]  
    y_train, y_test = y[:split], y[split:] 
    
    # Apply Bayes decision rule  
    y_pred = bayes_decision_rule(X_train, y_train, X_test)
    
    # Calculate accuracy
    # print(len(y_test))
    # print(len(y_pred))
    if len(y_test) == len(y_pred):
        accuracy = np.mean(y_pred == y_test)
        # print(f"Accuracy: {accuracy:.2f}")
    else:
        raise("Length of prediction is not equal to Length of Test. Please correct it.")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    '''
    First feature (x-axis values) 0
    Second feature (y-axis values) 1
    '''
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 
                color='blue', alpha=0.5,s=40, label='Class 0 (train)') 
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
    plt.suptitle("Bayes Decision Rule Classification", fontsize=14)
    plt.title(f"Accuracy: {accuracy:.2f}",fontsize=13)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.legend(fontsize=10, framealpha=1)
    plt.grid(True, alpha=0.3)
    plt.show()