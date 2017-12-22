
import os
import numpy as np
import pandas as pd
import pdb

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

class PA1:

    def __init__(self, estimator):
        self.data, self.label = self.preprocess_data(estimator)
        self.estimator = estimator    

    def preprocess_data(self, estimator):

        names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
             "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
             "hours-per-week", "native-country", "label"]
          
        f = open("adult.csv", "w")


        fdata = open("adult.data","r")
        ftest = open("adult_test.data")
        k=0
        

        

        for row in fdata:
            row = row.replace(" ", "")
            #print(row)
            f.write(row)
            k += 1
            #if(k == 20):
            #    break
        for row in ftest:
            row = row.replace(" ", "")
            f.write(row)

        #print(k)
        f.close()    
        
        datadf = pd.read_csv("adult.csv", header = None, na_values = ['?'], names = names)

        
        del datadf["workclass"]
        
        del datadf["race"]

        
        del datadf["native-country"]
        
        del datadf["fnlwgt"]    

        data = self.makeBinaryIfPosbl(datadf.dropna())
        label = data.pop(">50K")
        del data["<=50K"]  

        return data, label


    def makeBinaryIfPosbl(self, dframe):
        
        #print(dframe)
        binaryListForEachUniqueValue = pd.DataFrame()

        #get type of the columns and if its not float,
        #then we

        for curr in dframe.columns:
            ctype = dframe[curr].dtype
            #print(dframe[curr])
            #print(ctype) object or float
            if ctype != float:
                
                #print(dframe[curr].value_counts().index, "value")
                
                #go through each unique value in each of the classes
                #and make true for that value and false for all other values
                #i.e. a special list for each unique value in which if that 
                #value is present then true, else false.
                #Apparently thats what I got after searching online
                #Do this and feed to train function to estimate using sklearn      

                for c in dframe[curr].value_counts().index:

                    #print(dframe[curr], (dframe[curr] == c))
                    #print(curr, dframe[curr].value_counts().index, c," khatm")
                    #print(dframe[curr], dframe[curr]==c)
                    #print(c," c over \n")
                    binaryListForEachUniqueValue[c] = (dframe[curr] == c)

                #print(dframe[curr].value_counts().index)
                #print(curr,"currrrrrr")
            elif ctype == np.int or ctype == np.float:
                binaryListForEachUniqueValue[curr] = dframe[curr]
            else:
                print("unused curr: {}".format(curr))
        #print(binaryListForEachUniqueValue)
        return binaryListForEachUniqueValue



    def train(self, n_examples=None):

        X = self.data.values.astype(np.float32)
        y = self.label.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.estimator.fit(X_train, y_train)

        y_pred = self.estimator.predict(X_test)
        print( classification_report(y_test, y_pred, target_names=["<=50k", ">50k"]))

        y_score = self.estimator.predict_proba(X_test)
        print("roc: {}".format( roc_auc_score(y_test, y_score[:,1]) ))

if __name__ == "__main__":

    
    seed = np.random.randint(100000)


    #estimator = KNeighborsClassifier(n_neighbors=2)
    #estimator = KNeighborsClassifier(n_neighbors=5)
    #estimator = KNeighborsClassifier(n_neighbors=25)
    #estimator = KNeighborsClassifier(n_neighbors=50)
    estimator = KNeighborsClassifier(n_neighbors=1000)
    

    pa1 = PA1(estimator)    
    
    pa1.train()
