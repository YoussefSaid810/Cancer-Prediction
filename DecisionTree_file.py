from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
 #DecisionTree
class DecisionTree(object):
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test   
        self.y_test=y_test
       # self.yPred=yPred
    def Tree(self):
        dtc = DecisionTreeClassifier() # Create Decision Tree classifer object
        dtc.fit( self.X_train, self. y_train) # Train Data = Decision Tree Classifer
        self.yPred = dtc.predict( self.X_test) #Predict the response for test dataset
        dtc_acc=accuracy_score( self.y_test,self.yPred)*100
        print("Decision tree Accuracy:",dtc_acc, "%")
        # Output=not constant Change by run the program 
        # Model Precision: what percentage of positive tuples are labeled as such?
        print("Precision Decision tree:",metrics.precision_score( self.y_test, self.yPred)*100, "%")
        # Model Recall: what percentage of positive tuples are labelled as such?
        print("Recall Decision tree:",metrics.recall_score( self.y_test, self.yPred)*100, "%")
        # Model Accuracy, Correctness of the model
        print("Confusion_matrix :")
        print(confusion_matrix( self.y_test, self.yPred))
        DecisionTree_Report=classification_report( self.y_test, self.yPred)
        print("DecisionTree Report :")
        print(DecisionTree_Report, sep='\n')
        print(self.yPred)
        print('*'*60)
        return dtc
        
