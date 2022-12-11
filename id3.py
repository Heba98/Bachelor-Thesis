import pandas as pd
import math
import copy
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\hebam\Desktop\cancerdata.csv')
df = pd.DataFrame(data=dataset)
X = dataset.iloc[:, :].values

X_train,X_test= train_test_split(X, test_size=0.2, random_state=0)

for i in range(247):
    if X_train[i][1]>=65:
        X_train[i][1]="O"
    else:
        X_train[i][1]="Y"

for i in range(62):
    if X_test[i][1]>=65:
        X_test[i][1]="O"
    else:
        X_test[i][1]="Y"

X_t = pd.DataFrame(data=X_train)


with open('Data.txt', 'w') as f:
    print(X, file=f)

with open('Train.txt', 'w') as f:
    print(X_train, file=f)

with open('Test.txt', 'w') as f:
    print(X_test, file=f)


variables= dataset.columns[0:15] # to get all variables

attribute=['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
       'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
       'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
       'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']


correct=0


class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = None


def findEntropy(data, rows):
    yes = 0
    no = 0
    ans = -1
    idx = len(data[0]) - 1
    #print(idx)
    #print(data[0])
    #print(data[0][4])
    entropy = 0
    for i in rows:
        if data[i][idx] == 'YES':
            yes = yes + 1
        else:
            no = no + 1

    x = yes / (yes + no)
    y = no / (yes + no)
    if x != 0 and y != 0:
        entropy = -1 * (x * math.log2(x) + y * math.log2(y))
    if x == 1:  # when no=0
        ans = 1
    if y == 1:
        ans = 0  # when yes=0
    #print(entropy)
    #print(ans)
    return entropy, ans


def findMaxGain(data, rows, columns):
    maxGain = 0
    retidx = -1
    entropy, ans = findEntropy(data, rows)
    if entropy == 0:
       """"" if ans == 1:
            print("Yes")
        else:
            print("No")
        return maxGain, retidx, ans"""

    for j in columns:
        mydict = {}
        idx = j
        for i in rows:
            key = data[i][idx]
            #print(key)
            #print(mydict)
            if key not in mydict:
                mydict[key] = 1
            else:
                mydict[key] = mydict[key] + 1
        gain = entropy

        #print(mydict)
        for key in mydict:
            yes = 0
            no = 0
            for k in rows:
                if data[k][j] == key:
                    if data[k][-1] == 'YES':
                        yes = yes + 1
                    else:
                        no = no + 1
            #print(yes, no)
            x = yes/(yes+no)
            y = no/(yes+no)
            # print(x, y)
            if x != 0 and y != 0:
                gain += (mydict[key] * (x*math.log2(x) + y*math.log2(y)))/247
        # print(gain)
        if gain > maxGain:
            # print("hello")
            maxGain = gain
            retidx = j

    return maxGain, retidx, ans


def buildTree(data, rows, columns):

    maxGain, idx, ans = findMaxGain(X_train, rows, columns)
    #print("-----NODE:   ----> " , end=" ")
    #print(attribute[idx])
    root = Node()
    root.childs = []
    # print(maxGain
    #
    # )
    if maxGain == 0:
        if ans == 1:
            root.value = 'YES'
        else:
            root.value = 'NO'
        return root

    root.value = attribute[idx]
    mydict = {}
    for i in rows:
        key = data[i][idx]
        if key not in mydict:
            mydict[key] = 1
        else:
            mydict[key] += 1

    newcolumns = copy.deepcopy(columns)
    newcolumns.remove(idx) #tar bort den columen som är nod
    for key in mydict:
        newrows = []
        for i in rows:
            if data[i][idx] == key:
                newrows.append(i)
        #print(newrows)
        temp = buildTree(data, newrows, newcolumns)
        temp.decision = key
        root.childs.append(temp)

    return root



def traverse(root):
    #print(root.decision)
    #print(root.value)

    n = len(root.childs)
    if n > 0:
        for i in range(0, n):
            traverse(root.childs[i])


def get_answar(root,ind):
    #if root.decision != None:
     #   print(root.decision)
    value=root.value
    #print(value)
    if value=="YES" or value=="NO":
        return value

    else:
        j=attribute.index(value)
        n = len(root.childs)
        if n > 0:
            for i in range(0, n):
                if X_test[ind][j] == get_dec(root.childs[i]):
                    answar=get_answar(root.childs[i], ind)
                    if answar=="YES" or answar=="NO":
                        return answar


   # if value=="Yes" or value=="No":
    #    return value


def get_dec(root):
    return root.decision

def get_value(root):
    return root.value




def calculate():
    rows = [i for i in range(0, 247)]
    columns = [i for i in range(0, 15)]
    root = buildTree(X_train, rows, columns)
    root.decision = 'Start'
    traverse(root)
    ##print(len(X_train))
    #print(X_train[7][1])


    #gain=findMaxGain(X_train, rows, columns)
    #print(gain)

   # idx = len(X_train[0]) - 1
    #print(X_train[7])
    #print(X_train[7][4])
    #print(idx)
    #root = buildTree(X_train, rows, columns)
    #traverse(root)
    #print("-------")
    #print("Accuracy:")
    correct = 0
    predictedvlue=[]
    actualvalue=[]
    for i in range(0, 62):
        treeAnswar = get_answar(root, i)
        predictedvlue.append(treeAnswar)
        # print("Tree: ", end=" ")
        # print(treeAnswar)
        RightAnwar = X_test[i][-1]
        actualvalue.append(RightAnwar)
        # print("Right: ", end=" ")
        # print(RightAnwar)
        if treeAnswar == RightAnwar:
            correct = correct + 1
    acc=correct / 62 * 100


   # confusion_matrix = metrics.confusion_matrix(actualvalue,  predictedvlue)
    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["No", "Yes"])

    #cm_display.plot()
    #plt.title('Confusion Matrix for Decision tree')
    #plt.show()
    #return acc
    print("Accuracy =  {}".format(acc))





calculate()


def calc():
    rows = [i for i in range(0, 247)]
    columns = [i for i in range(0, 15)]
    root = buildTree(X_train, rows, columns)
    index=[]
    for i in range(0, 247):
        treeAnswar = get_answar(root, i)
        RightAnwar = X_train[i][-1]
        if treeAnswar!=RightAnwar:
            index.append(i)

    print(index)
    print(len(index))

#calc()
