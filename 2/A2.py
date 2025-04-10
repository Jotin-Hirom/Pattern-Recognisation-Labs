''' Write a function to calculate the probability of rolling a six on a die  based on experimental data.
The function should:  
- Accept a list of die roll results (1-6)  
- Return the probability of rolling a six  
Example Input: [1, 6, 3, 6, 2, 6, 4, 5, 6, 1] 
Expected Output: 0.4 (4 sixes out of 10 rolls) '''
from matplotlib import pyplot as plt
import numpy as np

 
def calculate_six_probability(ListData,findThis):
    NewDataList = {}
    if type(findThis) != int:
        exit
    else:
        for i in ListData:
            newData = ListData.count(i) 
            if newData >= 1 and i not in NewDataList:
                NewDataList.update({i: newData})
        prob = round((NewDataList[findThis] / ListData.__len__()),2)
        return (prob)


''' A weather station records daily data about:  
- Whether it rained (True/False)  
- Whether it was cloudy (True/False)  

Write a function to calculate the joint probability of it being both rainy AND cloudy.  
Example Input:  rain_data = [True, False, True, True, False, False, True, False]  
cloud_data = [True, False, True, False, False, True, True, False]
Expected Output: 0.375 (3 days out of 8 were both rainy and cloudy) '''
def calculate_weather_joint_probability(rain_data, cloud_data): 
    numOfDays = len(rain_data)
    countNumOfBothDays = 0
    for i in range(numOfDays):
        if (((rain_data[i]!=False)==(cloud_data[i])!=False)):
            countNumOfBothDays+=1
    probResult  = round((countNumOfBothDays / numOfDays),3)
    return (f"{probResult} ({countNumOfBothDays} days out of {numOfDays} were both rainy and cloudy)") 


'''
Using student data, calculate the probability of passing an exam given that the student studied.
The function should:  
- Accept two lists: study_data and exam_results (both boolean)  
- Return P(Pass|Studied) 
Example Input:  
studied = [True, True, True, False, False, True, False, True]  
passed = [True, True, False, False, False, True, False, True] 
Expected Output: 0.83 (5 passed out of 6 who studied)

Formula: P(A|B) = P(A ∩ B)/P(B)'''
def calculate_passing_probability(studied, passed): 
    numStudied = 0
    numPassed = 0 
    for i  in range(len(studied)):
        if studied[i]==True:
            numStudied+=1
        if passed[i]==True:
            numPassed+=1
    probResult = (round((numPassed+1)/(numStudied+1),2))
    return (f"{probResult} ({numPassed+1} passed out of {numStudied+1} who studied)")


''' A medical test for a rare disease has the following properties:  
- 1% of the population has the disease (prior probability)  
- The test is 95% accurate for people who have the disease (sensitivity)  
- The test is 90% accurate for people who don't have the disease (specificity)  
Write a function to calculate the probability that 
    a person has the disease given a positive test result using Bayes' Theorem. 
Expected Output: ~0.087 (8.7%)
Formula: P(A/B) = (P(B/A).P(A))/P(B) '''
def calculate_disease_probability(prior_prob, sensitivity, specificity): 
    false_positive_rate = 1 - specificity
    numerator = sensitivity * prior_prob
    denominator = (sensitivity * prior_prob) + (false_positive_rate * (1 - prior_prob))
    return (numerator / denominator) 

'''Write a function that:  
1. Generates 1000 random numbers following a normal distribution  
2. Calculates the probability that a number falls within one standard  deviation of the mean  
3. Plots the distribution with matplotlib 
Expected Output:  
- A probability value close to 0.68 (theoretical value)  
- A histogram showing the normal distribution
'''
def analyze_normal_distribution(sample_size=1000):  
    data = np.random.normal(loc=0, scale=1, size=sample_size)
    mean, std = np.mean(data), np.std(data)
    within_one_std = np.sum((mean - std <= data) & (data <= mean + std)) / sample_size
    
    plt.hist(data, bins=30, density=True, alpha=0.6, color='b')
    plt.title('Normal Distribution Histogram')
    plt.xlabel('X->Value')
    plt.ylabel('Y->Frequency')
    plt.show()
     
    return within_one_std


if __name__ == "__main__" :
    '''<<--Data Declaration Here-->>'''
    findThis = 6
    sensitivity = 95/100
    specificity = (90/100)
    prior_prob = 1/100
    
    ListData = [1, 6, 3, 6, 2, 6, 4, 5, 6, 1] 
    rain_data = [True, False, True, True, False, False, True, False] 
    cloud_data = [True, False, True, False, False, True, True, False]
    studied = [ True, True, False, False,True, True, False, True]  
    passed = [True, True, False, False, False, True, False, True] 
    
    
    '''<<--Function Call Here-->>'''
    sixProbResult = calculate_six_probability(ListData,findThis)
    jointProbResult = calculate_weather_joint_probability(rain_data,cloud_data)
    passingProbResult = calculate_passing_probability(studied,passed)
    diseaseResult = calculate_disease_probability(prior_prob, sensitivity, specificity)
    within_one_std = analyze_normal_distribution()
    
    
    '''<<--Print Here-->>'''
    print(f"The probability of rolling a six on a die based on experimental data is {sixProbResult}.")
    print(f"{jointProbResult}")
    print(f"{passingProbResult}")
    print(f"{diseaseResult} ({diseaseResult * 100})")
    print(f"A probability value close to {within_one_std}")
    
    
    
    
    
    
    
'''
OUTPUT:
PS D:\Pattern Recognisation Lab> python -u "d:\Pattern Recognisation Lab\Feb_Lab\2\A2.py"
The probability of rolling a six on a die based on experimental data is 0.4.
0.375 (3 days out of 8 were both rainy and cloudy)
0.83 (5 passed out of 6 who studied)
0.08755760368663597 (8.755760368663596)
A probability value close to 0.681
'''