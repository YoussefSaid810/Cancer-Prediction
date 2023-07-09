from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

class Logistic_Regression(object):
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test   
        self.y_test=y_test
        #self.yPred=yPred
    def logic(self):
        # implemented our model through logistic regression
        lr = LogisticRegression()
        lr.fit(self.X_train,self.y_train)  #Train data
        self.yPred = lr.predict(self.X_test)  #Test data
        # array containing the actual output
        #print( "Logistic Regression train Accuracy: ", accuracy , "%" )
        lr_acc=accuracy_score(self.y_test,self.yPred)*100
        print("Logistic Regression  Accuracy:",lr_acc, "%")
        # Output=0.9652777777777778 Change by change (random_state)
        # Model Precision: what percentage of positive tuples are labeled as such?
        print("Precision Logistic Regression:",metrics.precision_score(self.y_test,self.yPred)*100, "%")
        # Model Recall: what percentage of positive tuples are labelled as such?
        print("Recall Logistic Regression:",metrics.recall_score(self.y_test, self.yPred)*100, "%")
        # Model Accuracy, Correctness of the model
        print("Confusion_matrix :")
        print(confusion_matrix(self.y_test,self.yPred) )
        LogisticRegression_Report= classification_report(self.y_test,self.yPred) #Report
        print("Logistic Regression Report :")
        print(LogisticRegression_Report , sep='\n')
        print(self.yPred)
        print('*'*60)
        return lr
        
