import math
import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv(r'C:\Users\hebam\Desktop\cancerdata.csv')
df = pd.DataFrame(data=dataset) # read rows and columns

X = dataset.iloc[:,:-1]

y = dataset.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

with open('data.txt', 'w') as f:
    print(dataset, file=f)

with open('X_train.txt', 'w') as f:
    print(X_train, file=f)

with open('X_test.txt', 'w') as f:
    print(X_test, file=f)

with open('y_train.txt', 'w') as f:
    print(y_train, file=f)

with open('y_test.txt', 'w') as f:
    print(y_test, file=f)


    X_t = pd.DataFrame(data=X_train)
    Y_t=pd.DataFrame(data=X_test)
    index=0

    gender= X_t.iloc[:,0]
    age= X_t.iloc[:,1]
    smoking = X_t.iloc[:, 2]
    yellow_fingers = X_t.iloc[:, 3]
    anxiety = X_t.iloc[:, 4]
    deep_pressure = X_t.iloc[:, 5]
    chronic_diase = X_t.iloc[:, 6]
    fatigue = X_t.iloc[:, 7]
    allergy = X_t.iloc[:, 8]
    wheezing = X_t.iloc[:, 9]
    alcohol_consumption = X_t.iloc[:, 10]
    coughing = X_t.iloc[:, 11]
    shortness_of_breath = X_t.iloc[:, 12]
    swallow_difficulty = X_t.iloc[:, 13]
    chest_pain = X_t.iloc[:, 14]

    node=[]

    attributes=[gender ,smoking, yellow_fingers, anxiety,deep_pressure, chronic_diase, fatigue, allergy, wheezing,alcohol_consumption,coughing,shortness_of_breath, swallow_difficulty,chest_pain]

#print(df.iloc[0,0:15])
# #print(df.loc[2,3])
variables= dataset.columns[0:15] # to get all variables
current_variables= variables
variables_gain=[]
#get entropy for the whole table
def get_S_entropy():
    S_entropy=calc_entropy(get_Yes(),get_No())
    return S_entropy


# Get entropy
def calc_entropy(p_value, n_value):
   tot= p_value+ n_value
   s=-(p_value/tot)* math.log(p_value/tot, 2) - (n_value/tot)*math.log(n_value/tot, 2)
   return s

def get_Yes():
    yes_num = 0
    for x in y_train:
       if x == "YES":
           yes_num = yes_num+1
    return yes_num


def get_No():
    no_num = len(y_train)- get_Yes()
    return no_num

def get_variable(var_list,y_train):
    var_1=0
    var_2=0
    var_1_yes=0
    var_1_NO=0
    var_2_yes=0
    var_2_NO=0
    count1=0
    index1=0
    index2=0

    for x in var_list:
     if x==1:
         var_1 = var_1 + 1
         index1=X_t.index[count1]
         count1 = count1 + 1
         if y_train[index1]=="YES":
             var_1_yes=var_1_yes+1
     if x==2:
        var_2 = var_2 + 1
        index2 = X_t.index[count1]
        count1= count1 + 1
        if y_train[index2] == "YES":
            var_2_yes = var_2_yes + 1

    var_1_NO=var_1-var_1_yes
    var_2_NO = var_2 - var_2_yes
   # print(var_1)
    #print(var_1_yes)
    #print(var_1_NO)
    #print(var_2)
    #print(var_2_yes)
    #print(var_2_NO)
    entropy=[var_1,var_1_yes,var_1_NO, var_2, var_2_yes,var_2_NO]
    return entropy

def get_variable_gender(var_list,y_train):
    var_M = 0
    var_F = 0
    var_F_yes = 0
    var_F_NO = 0
    var_M_yes = 0
    var_M_NO = 0
    count = 0
    index_F = 0
    index_M= 0

    for x in var_list:
        if x == "F":
            var_F = var_F+ 1
            index_F= X_t.index[count]
            count = count + 1
            if y_train[index_F] == "YES":
                var_F_yes = var_F_yes + 1
        if x == "M":
            var_M= var_M + 1
            index_M = X_t.index[count]
            count = count + 1
            if y_train[index_M] == "YES":
                var_M_yes = var_M_yes + 1

    var_F_NO = var_F - var_F_yes
    var_M_NO = var_M - var_M_yes
    entropy_val = [var_F, var_F_yes, var_F_NO, var_M, var_M_yes, var_M_NO]
    return entropy_val

def calc_variable_gain(var_list):
    if ((var_list[0])==gender[0]):
        value=get_variable_gender(var_list,y_train)
    else:
        value=get_variable(var_list,y_train)
    var_1=value[0]
    var_1_yes= value[1]
    var_1_No=value[2]
    var_2=value[3]
    var_2_yes= value[4]
    var_2_No=value[5]
    s_entropy=get_S_entropy()
    entropy_1=calc_entropy(var_1_yes,var_1_No)
    entropy_2=calc_entropy(var_2_yes,var_2_No)
    sum= var_1+ var_2
    entropy_sum=((var_1/sum)*(entropy_1)) +((var_2/sum)*(entropy_2))
    gain= s_entropy-entropy_sum
    variables_gain.append(gain)
   # print("gain")
    #print(gain)
    return gain

def get_variables_gain():
    for x in attributes:
        calc_variable_gain(x)

def get_root_node():
    get_variables_gain()
    min_gain= min(variables_gain)
    min_index= variables_gain.index(min_gain)
    attributes.pop(min_index)
    if (min_index>0):
        min_index=min_index+1
    node.append(variables[min_index])
    print(node)

def num_of_age():
    age_numbers=[]
    main_age=0
    for x in age:
        main_age = main_age + x
        if x not in age_numbers:
            age_numbers.append(x)
    main_age=main_age/len(age)
    print(main_age)
    print(age_numbers)
    print(len(age_numbers))












      #      if y_train[count] == "YES":
     #           var_yes = var_yes + 1
   # print(var_yes)
    #print(var_1)

   # no_num=len(y_train)- yes_num
#get_variable(smoking,y_train)

#print(X_t.index[0])
#get_variable_gender(gender,y_train)
#calc_variable_gain(gender)
#print(variables)
#print(variables[2])
#calc_variable_gain(variables[2])
#print(variables_gain)
#get_root_node()
num_of_age()





















