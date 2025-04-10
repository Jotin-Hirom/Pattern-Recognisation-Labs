'''#Finding Length'''
def find_Length(data):
    count = 0
    for i in data:
        count+=1
    return count

'''#Finding Square root''' 
def find_Sqrt(data):
    return (data**0.5)

'''#Finding Mean'''
def find_Mean(data):
    numbers = find_Length(data)
    sum = 0
    for i in data:
        sum = sum + i
    meanValue = sum/numbers
    return meanValue

'''#Finding Median'''
def find_Median(data):
    sortList = sorted(data)
    numbers = find_Length(sortList)
    firstTerm = int((numbers/2)-1)
    secondTerm = int(((numbers/2)+1)-1)
    medianValue = 0
    if (numbers%2==0):
        medianValue = (sortList[firstTerm] + sortList[secondTerm])/2
    else:
        medianValue =sortList[int((numbers+1)/2)]
    return medianValue

'''#Finding Standard Deviation'''
def find_Std(data):
    numbers = find_Length(data)
    mean = find_Mean(data)
    sum = 0
    square = 0
    for i in data:
        sum = (i - mean) 
        square  = (sum**2) + square
    variance = (square)/(numbers - 1)
    return find_Sqrt(variance)
   
'''#Finding Maximum'''        
def find_Max(data):
    max = data[1] 
    for i in data:
        if max < i:
            max = i
    return max 

'''#Finding Minimum'''
def find_Min(data):
    min = data[1] 
    for i in data:
        if min > i:
            min = i
    return min 

'''#Finding Range'''
def find_Range(data):
    minimum = find_Min(data)
    maximum = find_Max(data)
    ranges = maximum - minimum
    return ranges

'''#Finding Single Mode'''
def find_Mode(data):
    maxCount = 0
    for i in data:
        count = data.count(i)
        if count > maxCount:
            maxCount = count
            return i

'''#Finding Multimode'''
def find_MultiMode(data):
    maxCounted = 0
    modes = []
    for i in data:
        count = data.count(i)
        if count > maxCounted:
            maxCounted = count
            modes.append(i)
        elif count == maxCounted and i not in modes:
            modes.append(i)
    return modes
    '''#Another way of finding
     temp_dic = {}
     for i in data:
         temp_dic[i] = temp_dic.get(i,0) +1
     maximum = max(temp_dic.values())
     modes = [key for key, value in temp_dic.items() if maximum == value ]
     return modes'''
     
'''#Finding no mode i.e. all the values have the same frequency
e.g. data = [1,2,3,4] => each number appears only once so, there is no mode
e.g. data = [1,1,2,2,3,3,4,4] => all have same frequency so, there is no mode
e.g. data = [1,1,2,2,2,3,3,3,4,4] => 2 & 3 have highest frequency with 3 so, there is a mode'''
def find_NoMode(data):
    maxCounted,check = 0,0
    modes = []
    for i in data:
        count = data.count(i)
        if count>maxCounted:
            maxCounted=count
            modes.append(i)
        elif count==maxCounted and i not in modes:
            modes.append(i)
    modes = set(modes)
    data = set(data)
    if(find_Length(data))==(find_Length(modes)):
       return "No mode."
    else:
        return f"Yes, {len(modes)} Mode available."
            
'''#Finding Moving average of time series data with specified window size. 
    data = [1,3,5,7,9,11,13,15]
    window_size = 3
    output: [3.0,5.0,7.0,9.0,11.0,13.0]
    (1+3+5)/window_size=3.0 <=MEAN
    (3+5+7)/window_size=5.0
    (5+7+9)/window_size=7.0
    (7+9+11)/window_size=9.0
    (9+11+13)/window_size=11.0
    (11+13+15)/window_size=13.0
    '''
def find_MovingAverage(data,windowSize):
    if windowSize<=0:
        raise ValueError("Window size should be greater than 0.")
    if find_Length(data)<=0 and find_Length(data)<windowSize:
        raise TypeError("Again put the value of Data.")
    # avg = []
    avg1 = []
    addedSum = sum(data[:windowSize]) #POS 0:3 (0->+1->+2->) X3
    avg1.append(addedSum/windowSize)
    for i in range(find_Length(data)- windowSize):
        addedSum += data[i + windowSize] - data [i] 
        addedValue = addedSum / windowSize
        avg1.append(addedValue)
    return (avg1) 
    
    '''Another way'''
    # for i in range(find_Length(data) - windowSize +1):
    #     value = data[i: i + windowSize] 
    #     finalValue = sum(value) / windowSize
    #     avg.append(finalValue)  
    # return (avg)
    

'''Main Program'''    
if __name__ == '__main__':
    '''Declaring variables''' 
    data = [85,92,78,65,88,72,90,85,83,72]
    # data = [1,2,3,4] 
    # data = [3,4,1,1,2,2,3,3,4,4,2,1]
    # data = [1,1,2,2,2,3,3,3,4,4] 
    
    '''Functions with variables''' 
    mean = find_Mean(data)
    median = find_Median(data)
    std = find_Std(data)
    maximum = find_Max(data)
    minimum = find_Min(data) 
    ranges = find_Range(data)
    mode = find_Mode(data)
    multimode = find_MultiMode(data)
    no_mode=find_NoMode(data)
    movingAvg=find_MovingAverage(data=[1,3,5,7,9,11,13,15],windowSize=3)

    '''PRINTING THE VALUES''' 
    print("<<--PRINTING THE VALUES-->>")
    print(f"Maximum value: {maximum}")
    print(f"Minimum value: {minimum}")
    print(f"Mean value: {mean}")
    print(f"Median value: {median}")
    print(f"Standard Deviation value: {round(std,2)}")
    print(f"Ranges value: {ranges}")
    print(f"Single Mode value: {mode}")
    print(f"Multi Mode value: {multimode}")
    print(f"Checking No-Mode or not :{no_mode}")
    print(f"Finding moving average of time series :{movingAvg}") 
    
    