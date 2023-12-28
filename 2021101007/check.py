import sys
import numpy as np
import heapq as hq
from numpy.linalg import norm
from prettytable import PrettyTable
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class KNN:
    def __init__(self, data, k=7, encoderType='VIT', distanceMetric='manhattan'):
        self.k = k
        self.data = data
        self.encoderType = encoderType
        self.distanceMetric = distanceMetric

    def setK(self, k):
        self.k = k

    def setEncoderType(self, encoderType):
        self.encoderType = encoderType

    def setDistanceMetric(self, distanceMetric):
        self.distanceMetric = distanceMetric

    def getDistance(self, vec1, vec2):
        if self.distanceMetric=='manhattan':
            return np.sum(np.abs(vec1-vec2))
        if self.distanceMetric=='euclidean':
            return norm(vec1-vec2)
        if self.distanceMetric=='cosine':
            return 1-np.dot(vec1,vec2.T)/(norm(vec1)*norm(vec2))
        
    def getLabel(self, arr):
        freq = {}
        for pair in arr:
            freq[pair[1]] = freq.get(pair[1], 0) + 1
        if len(freq)==self.k:
            label = arr[0][1]
        else:
            label = max(zip(freq.values(), freq.keys()))[1]
        return label
        
    def compute(self, x):
        idx = 2 if self.encoderType=='VIT' else 1
        arr = []
        for y in self.data:
            distance = self.getDistance(x[idx], y[idx])
            if len(arr) < self.k:
                arr.append([-distance, y[3]])
                if len(arr)==self.k:
                    hq.heapify(arr)
            else:
                hq.heappushpop(arr, [-distance, y[3]])
        return self.getLabel(arr)
    
    def classify(self, x_test):
        predicted = []
        for x in x_test:
            label = self.compute(x)
            predicted.append(label)
        return predicted
    
def get_scores(actual, predicted):
    fone = f1_score(actual, predicted, zero_division=0, average='macro')
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, zero_division=0, average='macro')
    recall = recall_score(actual, predicted, zero_division=0, average='macro')
    return [fone, accuracy, precision, recall]

def print_results(actual, predicted):
    result = get_scores(actual, predicted)
    t = PrettyTable(['Measure', 'Value'])
    t.add_row(['F1-score', result[0]])
    t.add_row(['Accuracy', result[1]])
    t.add_row(['Precision', result[2]])
    t.add_row(['Recall', result[3]])
    print(t)

# print("2: ", sys.argv[1])
# print("3: ", sys.argv[2])

train_file = sys.argv[1]
test_file = sys.argv[2]

train = np.load(train_file, 'r', True)
test = np.load(test_file, 'r', True)
x_test = test[:,0:3]
y_test = test[:,3]

knn = KNN(train)
y_pred = knn.classify(x_test)
print_results(y_test, y_pred)
