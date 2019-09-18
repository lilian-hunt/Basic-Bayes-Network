import math
def parseInputLine(input_line):
    line = []
    for s in input_line:
        if s.endswith(":"):
            line.append(int(s[:-1]))
        else:
            line.append(int(s))
    return (line)

def calculateMLEestimates(container, num_movies):
    ret = [[[0 for x in range(2)] for y in range(2)] for i in range(num_movies)]
    for i in range(num_movies):
        box = container[i]
        ret_box = ret[i]
        for j in range(2):
            sum = box[j][0] + box[j][1]
            for k in range(2):
                if (sum == 0):
                    ret_box[j][k] = 0
                else:
                    ret_box[j][k]= box[j][k]/float(sum)
                
    return(ret)

def calculateMapestimates(container, num_movies):
    ret = [[[0 for x in range(2)] for y in range(2)] for i in range(num_movies)]
    for i in range(num_movies):
        box = container[i]
        ret_box = ret[i]
        for j in range(2):
            sum = box[j][0] + box[j][1]
            for k in range(2):   
                ret_box[j][k]= (box[j][k]+1)/float(sum+2)         
    return(ret)

#Calculatest the MLE estimates for Y
#P_iMLE = n_i / n
def calculateYMLE(prob_Y):
    ret = [0,0]
    sum = prob_Y[0] + prob_Y[1]
    if sum == 0:
        return ret
    for i in range(2):
        ret[i] = prob_Y[i]/float(sum)
    return ret

#Calculatest the MAP estimates for Y, since k = 2 for binary
#P_iMAP = n_i + 1 / (n + k)
def calculateYMAP(prob_Y):
    ret = [0,0]
    sum = prob_Y[0] + prob_Y[1]+2
    if sum == 0:
        return ret
    for i in range(2):
        ret[i] = (prob_Y[i]+1)/float(sum)
    return ret

def predict(train_file, test_file, useMLE):
    #Train the model
    file = open(train_file,"r")
    num_movies = int(file.readline().strip())
    num_users = int(file.readline().strip())
    
    #Initialise with 1 to account for zero frequency problem
    container = [[[1 for x in range(2)] for y in range(2)] for i in range(num_movies)]
    #Start from 2 * number of movies to account for zero frequency problem
    prob_Y = [2*num_movies,2*num_movies]
    for i in range(num_users):
        line = parseInputLine(file.readline().strip().split(" "))
        for x in range(num_movies):
            box = container[x]
            get_x = line[x]
            get_y = line[num_movies]
            box[get_y][get_x] += 1
        prob_Y[line[num_movies]] += 1
    prob_Y_MLE = calculateYMLE(prob_Y)
    
    # If using Maximum Likelihood Estimators
    if useMLE:
        estimates = calculateMLEestimates(container, num_movies)
    # Otherwise will calculate with Lapace MAP estimates
    else:
        estimates = calculateMapestimates(container, num_movies)
    
    # Try the testing data 
    file = open(test_file,"r")
    num_movies = int(file.readline().strip())
    num_users = int(file.readline().strip())
    count_accurate_predictions = 0
    for i in range(num_users):
        line = parseInputLine(file.readline().strip().split(" "))
        
        #Use Bayes Theorem to calculate the probability of 0/1
        #Use logs for numerical stability
        yIs0 = math.log(prob_Y_MLE[0])
        yIs1 = math.log(prob_Y_MLE[1])
        for movie in range(num_movies):
            box = estimates[movie]
            get_x = line[movie]     
            yIs0 += math.log(box[0][get_x])
            yIs1 += math.log(box[1][get_x])
        
        # Determine which has a higher argmax, will predict the outcome with the larger value
        if (yIs0 > yIs1):
            if line[num_movies] == 0:
                count_accurate_predictions += 1
        else:
            if line[num_movies] == 1:
                count_accurate_predictions += 1
    #Prints the percentage accuracy and the number of correctly identified cases
    print("Accurately identified: " + str(count_accurate_predictions) + "/" + str(num_users) + "%: "+ (str(count_accurate_predictions/float(num_users))))

# predict("netflix-train.txt", "netflix-test.txt", True)
# predict("netflix-train.txt", "netflix-test.txt", False)
# predict("heart-train.txt", "heart-test.txt", True)
# predict("heart-train.txt", "heart-test.txt", False)
