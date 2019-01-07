import numpy as np
UBIT = 'sahilsuh'
np.random.seed(sum([ord(c) for
c in UBIT]))
from matplotlib import pyplot as plt
X = np.array([[5.9, 3.2],
     [4.6, 2.9],
     [6.2, 2.8],
     [4.7, 3.2],
     [5.5, 4.2],
     [5.0, 3.0],
     [4.9, 3.1],
     [6.7, 3.1],
     [5.1, 3.8],
     [6.0, 3.0]])

#Initializing Clusters
k = 3

mean_new = np.array([[6.2, 3.2],
                [6.6, 3.7],
                [6.5, 3.0]])


def calculate_euclidean_distance(mean_new, X):
    group = np.zeros((k,X.shape[0]))
    for i in range(X.shape[0]):
        a = X[i]
        for j in range(k):
            b = mean_new[j]
            dist = np.linalg.norm(a-b)
            group[j][i] = dist
    return group
            
'''This function calculates the group matrix, basicallyi it is of the order,
(cluster_number, number of data points). For every row in group matrix, we get to know
which data points belongs to which cluster. i.e. for example if 
index[0][1] = 1, this means that the data point = 1 belongs to cluster number = 0.
The following code is modified where we are storing 1 for Cluster 1, 2 for Cluster 2 and
3 for Cluster 3'''
def calculate_group_matrix(distance, X):
    val = []
    index = np.zeros((k,X.shape[0]))
    for i in range(distance.shape[1]):
        for j in range(distance.shape[0]):
            val.append([distance[j][i]])
        a = val.index(min(val))
        index[a][i] = a + 1
        val = []
    return index

''' This function is similar to the above function, except that it calculates the group
matrix for image, baboon.jpg'''
def calculate_group_matrix_img(distance, X):
    val = []
    index = np.zeros((k,X.shape[0]))
    for i in range(distance.shape[1]):
        for j in range(distance.shape[0]):
            val.append([distance[j][i]])
        a = val.index(min(val))
        index[a][i] = 1
        val = []
    return index

#Calculate new means    
'''It calculates the updated mean using the group matrix'''
def calculate_updated_mean(index, X): 
    inter = []
    mean = []
    for i in range(k):
        for j in range(X.shape[0]):
            if(index[i][j] == 1 or index[i][j] == 2 or index[i][j] == 3):
                inter.append(X[j])
        mean.append(np.mean(inter, axis = 0))
        inter = []
    return mean

def plot_figure(X, mean_new, index, filename):
    
    #3.1
    #lst_colors stores the color values and lst_values stores the cluster values(Classification Vector)
    lst_colors = []
    lst_values = []
    for i in range(index.shape[1]):
        for j in range(index.shape[0]):
            if(j == 0 and index[j][i] == 1):
                lst_colors.append('Red')
                lst_values.append(1)
                continue
            elif(j == 1 and index[j][i] == 2):
                lst_colors.append('Green')
                lst_values.append(2)
                continue
            elif(j == 2 and index[j][i] == 3):
                lst_colors.append('Blue')
                lst_values.append(3)
                
    print("Classification Vector: \n",lst_values)            
                
    X_xcoord = X[:,:1].tolist()
    X_ycoord = X[:,1:].tolist()
        
    mean_new = np.array(mean_new)    
        
    mean_new_xcoord = mean_new[:,:1]
    mean_new_ycoord = mean_new[:,1:]
    
    mean_new_colors = ['Red', 'Green', 'Blue']
    
    plt.scatter(X_xcoord, X_ycoord, color=lst_colors, marker = '^')
    plt.scatter(mean_new_xcoord, mean_new_ycoord, color=mean_new_colors, marker = "o")
    
    s = str
    for i in range(X.shape[0]):
        s = "("+ str(X_xcoord[i][0]) +","+ str(X_ycoord[i][0]) + ")"
        plt.text(X_xcoord[i][0], X_ycoord[i][0],s)
        
    for i in range(mean_new.shape[0]):
        plt.text(mean_new_xcoord[i][0], mean_new_ycoord[i][0], s="(" + str(mean_new_xcoord[i][0]) +","+ str(mean_new_ycoord[i][0]) + ")")
       
    plt.savefig(filename)
    plt.clf()
    
mean_prev = mean_new
for i in range(3):
    
    if(i == 0): #Base Case
        str1 = "task3_iter1_a.jpg"
        group = calculate_euclidean_distance(mean_new, X)
        index = calculate_group_matrix(group, X)
        plot_figure(X, mean_new, index, str1)
        mean_new = calculate_updated_mean(index, X)
        
    if(i == 1):
        str2 = "task3_iter1_b.jpg"
        plot_figure(X, mean_new, index, str2)
        group = calculate_euclidean_distance(mean_new, X)
        index = calculate_group_matrix(group, X)
        str3 = "task3_iter2_a.jpg"
        plot_figure(X, mean_new, index, str3)
        mean_new = calculate_updated_mean(index, X)
    
    if(i == 2): 
        group = calculate_euclidean_distance(mean_new, X)
        index = calculate_group_matrix(group, X)
        mean_new = calculate_updated_mean(index, X)
        str4 = "task3_iter2_b.jpg"
        plot_figure(X, mean_new, index, str4)

   
#Task 3.4 - Color Quantization
import cv2
import random
image = cv2.imread("baboon.jpg")
(h, w) = image.shape[:2]
img_m = np.array(image)
'''
#clusters = [3, 5, 10, 20]
#a = int(input("Enter number of Clusters, eg. 3, 5, 10,  15"))
#if(a == 3): k = 3
#elif(a == 5): k = 5
#elif(a == 10): k = 10
#elif(a == 20): k = 20
'''
#Initializing Clusters
k = 3
mean_new = [] #for list
for i in range(k): 
   x = random.randint(0, 512)
   y = random.randint(0, 512) 
   mean_new.append(img_m[x][y])

print("Initial Mean Values: \n", mean_new)
mean_new = np.array(mean_new, dtype = float) 
'''Reference: (Concept of Reshaping, https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/)'''
img = image.reshape((image.shape[0] * image.shape[1], 3))

while(True):
    
    #To check whether the algorithm converged or not
    mean_prev = mean_new
    distance = calculate_euclidean_distance(mean_new, img)
    index = calculate_group_matrix_img(distance, img)
    mean_new = calculate_updated_mean(index, img)            
    for i in range(index.shape[1]):
        for j in range(index.shape[0]):
            if(index[j][i] == 1):
                img[i] = mean_new[j]
                continue
    if(np.array_equal(mean_prev, mean_new)):
        break
               
            
img = img.reshape((h, w, 3))
if(k == 3):
    cv2.imwrite("task3_baboon_3.jpg",img)
elif(k == 5):
    cv2.imwrite("task3_baboon_5.jpg",img)
elif(k == 10):
    cv2.imwrite("task3_baboon_10.jpg",img)
elif(k == 20):
    cv2.imwrite("task3_baboon_20.jpg",img)
