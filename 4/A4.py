
# numpy: Fundamental package for numerical computations in Python
# Used for array operations, mathematical functions, and statistical calculations
import numpy as np 

# pandas: Data analysis library for creating and manipulating DataFrames
# Used for organizing and displaying transformed data
import pandas as pd

# sklearn.preprocessing: Provides data preprocessing utilities
# MinMaxScaler for easy min-max scaling implementation
from sklearn.preprocessing import MinMaxScaler

# sklearn.metrics: Provides various metrics for model evaluation
# Used for calculating classification metrics and error measures
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score,
                           mean_squared_error, mean_absolute_error)

# math.sqrt: For calculating square root (used in RMSE)
from math import sqrt

# scipy.spatial.distance.mahalanobis: For calculating Mahalanobis distance
# scipy.linalg.inv: For matrix inversion (needed for Mahalanobis)
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

def question1():
    """
    Calculate various distance metrics between points/vectors.
    
    Returns:
        dict: Dictionary containing all calculated distance metrics
    """
    
    def euclidean_distance(p1, p2):
        """
        Calculate Euclidean distance between two 2D points.
        
        Formula: √((x2 - x1)² + (y2 - y1)²)
        
        Args:
            p1, p2 (array-like): Two points in 2D space
            
        Returns:
            float: Euclidean distance between points
        """
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def manhattan_distance(p1, p2):
        """
        Calculate Manhattan (taxicab) distance between two 2D points.
        
        Formula: |x2 - x1| + |y2 - y1|
        
        Args:
            p1, p2 (array-like): Two points in 2D space
            
        Returns:
            float: Manhattan distance between points
        """
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    
    def chebyshev_distance(p1, p2):
        """
        Calculate Chebyshev (chessboard) distance between two 2D points.
        
        Formula: max(|x2 - x1|, |y2 - y1|)
        
        Args:
            p1, p2 (array-like): Two points in 2D space
            
        Returns:
            float: Chebyshev distance between points
        """
        return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))
    
    def minkowski_distance(p1, p2, p):
        """
        Calculate Minkowski distance between two 2D points with parameter p.
        
        Formula: (|x2 - x1|ᵖ + |y2 - y1|ᵖ)^(1/p)
        
        Special cases:
        - p=1: Manhattan distance
        - p=2: Euclidean distance
        - p→∞: Chebyshev distance
        
        Args:
            p1, p2 (array-like): Two points in 2D space
            p (float): Order parameter
            
        Returns:
            float: Minkowski distance between points
        """
        return (abs(p1[0]-p2[0])**p + abs(p1[1]-p2[1])**p)**(1/p)
    
    def hamming_distance(v1, v2):
        """
        Calculate Hamming distance between two binary vectors.
        
        Formula: (Number of unequal elements) / (Total elements)
        
        Args:
            v1, v2 (array-like): Two binary vectors of same length
            
        Returns:
            float: Normalized Hamming distance (between 0 and 1)
        """
        return sum(el1 != el2 for el1, el2 in zip(v1, v2)) / len(v1)
    
    def cosine_distance(v1, v2):
        """
        Calculate cosine distance between two vectors.
        
        Formula: 1 - (v1·v2) / (||v1|| * ||v2||)
        
        Args:
            v1, v2 (array-like): Two vectors of same length
            
        Returns:
            float: Cosine distance between vectors (0 to 2)
        """
        return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def mahalanobis_distance(v1, v2, data):
        """
        Calculate Mahalanobis distance between two vectors.
        
        Formula: √((v1-v2)ᵀ * S⁻¹ * (v1-v2))
        where S is the covariance matrix of the data
        
        Args:
            v1, v2 (array-like): Two vectors of same length
            data (array-like): Dataset for covariance calculation
            
        Returns:
            float: Mahalanobis distance between vectors
        """
        cov = np.cov(data.T)
        inv_cov = inv(cov)
        return mahalanobis(v1, v2, inv_cov)
    
    # Example usage with sample data
    point1 = np.array([1, 2])  # First 2D point
    point2 = np.array([4, 6])  # Second 2D point
    vec1 = np.array([1, 0, 1, 1])  # First binary vector
    vec2 = np.array([0, 0, 1, 0])  # Second binary vector
    data_matrix = np.array([vec1, vec2, [1,1,0,1], [0,1,0,1]])  # Data matrix for Mahalanobis
    
    return {
        "euclidean": euclidean_distance(point1, point2),
        "manhattan": manhattan_distance(point1, point2),
        "chebyshev": chebyshev_distance(point1, point2),
        "minkowski_p1": minkowski_distance(point1, point2, 1),
        "minkowski_p2": minkowski_distance(point1, point2, 2),
        "hamming": hamming_distance(vec1, vec2),
        "cosine": cosine_distance(vec1, vec2),
        # "mahalanobis": mahalanobis_distance(vec1, vec2, data_matrix)
    }

def question2():
    """
    Perform data transformation using min-max scaling techniques.
    
    Returns:
        dict: Contains transformed data samples and statistics
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data: 100 points from normal distribution (μ=50, σ=15)
    original_data = np.random.normal(loc=50, scale=15, size=100)
    
    def min_max_scale(data):
        """
        Perform manual min-max scaling to [0, 1] range.
        
        Formula: (X - X_min) / (X_max - X_min)
        
        Args:
            data (array-like): Input data to be scaled
            
        Returns:
            array: Scaled data between 0 and 1
        """
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def custom_scale(data, new_min, new_max):
        """
        Scale data to custom range [new_min, new_max].
        
        Formula: [(X - X_min)/(X_max - X_min)] * (new_max - new_min) + new_min
        
        Args:
            data (array-like): Input data to be scaled
            new_min (float): Minimum of new range
            new_max (float): Maximum of new range
            
        Returns:
            array: Scaled data in custom range
        """
        scaled = (data - np.min(data)) / (np.max(data) - np.min(data))
        return scaled * (new_max - new_min) + new_min
    
    # Using sklearn's MinMaxScaler (automatically scales to [0,1])
    scaler = MinMaxScaler()
    sklearn_scaled = scaler.fit_transform(original_data.reshape(-1, 1)).flatten()
    
    # Apply transformations
    manual_scaled = min_max_scale(original_data)  # [0,1] scaling
    custom_scaled = custom_scale(original_data, -1, 1)  # [-1,1] scaling
    
    # Create DataFrame for comparison
    df = pd.DataFrame({
        'Original': original_data,
        'Manual Scaled (0-1)': manual_scaled,
        'Custom Scaled (-1-1)': custom_scaled,
        'Sklearn Scaled (0-1)': sklearn_scaled
    })
    
    # Calculate statistics for each dataset
    stats = {
        'Original': {
            'Min': np.min(original_data),
            'Max': np.max(original_data),
            'Mean': np.mean(original_data),
            'Std': np.std(original_data)
        },
        'Manual Scaled': {
            'Min': np.min(manual_scaled),
            'Max': np.max(manual_scaled),
            'Mean': np.mean(manual_scaled),
            'Std': np.std(manual_scaled)
        },
        'Custom Scaled': {
            'Min': np.min(custom_scaled),
            'Max': np.max(custom_scaled),
            'Mean': np.mean(custom_scaled),
            'Std': np.std(custom_scaled)
        }
    }
    
    return {
        'dataframe': df.head(),  # First 5 rows for display
        'statistics': stats  # Statistical summaries
    }

def question3():
    """
    Calculate classification metrics and error measures.
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # True and predicted values
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 0, 1, 1]
    
    # Get confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    def binary_cross_entropy(y_true, y_pred, eps=1e-15):
        """
        Calculate binary cross-entropy loss.
        
        Formula: -[y_true*log(y_pred) + (1-y_true)*log(1-y_pred)]
        
        Args:
            y_true (array-like): Ground truth labels (0 or 1)
            y_pred (array-like): Predicted probabilities
            eps (float): Small value to avoid log(0)
            
        Returns:
            float: Binary cross-entropy loss
        """
        y_pred = np.clip(y_pred, eps, 1 - eps)  # Clip to avoid numerical issues
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return {
        'TP': tp,  # True Positives
        'FP': fp,  # False Positives
        'TN': tn,  # True Negatives
        'FN': fn,  # False Negatives
        'Accuracy': accuracy_score(y_true, y_pred),  # (TP+TN)/Total
        'Precision': precision_score(y_true, y_pred),  # TP/(TP+FP)
        'Recall': recall_score(y_true, y_pred),  # TP/(TP+FN)
        'F1 Score': f1_score(y_true, y_pred),  # 2*(Precision*Recall)/(Precision+Recall)
        'MSE': mean_squared_error(y_true, y_pred),  # (1/n)*Σ(y_true-y_pred)²
        'RMSE': sqrt(mean_squared_error(y_true, y_pred)),  # √MSE
        'MAE': mean_absolute_error(y_true, y_pred),  # (1/n)*Σ|y_true-y_pred|
        'Binary Cross-Entropy': binary_cross_entropy(np.array(y_true), np.array(y_pred))
    }
 
if __name__ == "__main__":
    # Execute all questions and store results
    q1_results = question1()  # Distance metrics
    q2_results = question2()  # Data transformations
    q3_results = question3()  # Classification metrics
    
    # Display results
    print("=== Question 1 Results ===")
    for key, value in q1_results.items():
        print(f"{key}: {value:.4f}")
    
    print("\n=== Question 2 Results ===")
    print("First 5 rows of data:")
    print(q2_results['dataframe'])
    print("\nStatistics:")
    for dataset, stats in q2_results['statistics'].items():
        print(f"\n{dataset}:")
        for stat, value in stats.items():
            print(f"{stat}: {value:.4f}")
    
    print("\n=== Question 3 Results ===")
    for key, value in q3_results.items():
        print(f"{key}: {value:.4f}")
        
        
        
        
        
'''
=== Question 1 Results ===
euclidean: 5.0000
manhattan: 7.0000
chebyshev: 4.0000
minkowski_p1: 7.0000
minkowski_p2: 5.0000
hamming: 0.5000
cosine: 0.4226

=== Question 2 Results ===
First 5 rows of data:
    Original  Manual Scaled (0-1)  Custom Scaled (-1-1)  Sklearn Scaled (0-1)
0  57.450712             0.696879              0.393758              0.696879
1  47.926035             0.554890              0.109780              0.554890
2  59.715328             0.730639              0.461278              0.730639
3  72.845448             0.926376              0.852752              0.926376
4  46.487699             0.533448              0.066896              0.533448

Statistics:

Original:
Min: 10.7038
Max: 77.7842
Mean: 48.4423
Std: 13.5542

Manual Scaled:
Min: 0.0000
Max: 1.0000
Mean: 0.5626
Std: 0.2021

Custom Scaled:
Min: -1.0000
Max: 1.0000
Mean: 0.1252
Std: 0.4041

=== Question 3 Results ===
TP: 3.0000
FP: 2.0000
TN: 2.0000
FN: 1.0000
Accuracy: 0.6250
Precision: 0.6000
Recall: 0.7500
F1 Score: 0.6667
MSE: 0.3750
RMSE: 0.6124
MAE: 0.3750
Binary Cross-Entropy: 12.9522
'''