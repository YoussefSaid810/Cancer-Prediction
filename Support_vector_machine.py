from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm
# from sklearn.svm import SVC

#SVM model
class SVM(object):
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test   
        self.y_test=y_test
        #self.yPred=yPred
    def svms(self):
        svc = svm.SVC( kernel = 'linear') #Using kernel function
        svc.fit(self.X_train,self.y_train)   #Train data    
        self.yPred = svc.predict(self.X_test)  #Test data
        svc_acc=accuracy_score(self.y_test,self.yPred )*100
        print("svm Accuracy:",svc_acc, "%")
        # Output=0.9652777777777778 =logistic regression [Change by change (random_state)] 
        # Model Precision: what percentage of positive tuples are labeled as such?
        print("Precision svm:",metrics.precision_score(self.y_test, self.yPred)*100, "%")
        # Model Recall: what percentage of positive tuples are labelled as such?
        print("Recall svm:",metrics.recall_score(self.y_test, self.yPred)*100, "%")
        print("Confusion_matrix :")
        print(confusion_matrix(self.y_test,self.yPred))
        SVM_Report=classification_report(self.y_test, self.yPred)
        print("SVM Report :")
        print(SVM_Report, sep='\n')
        print(self.yPred)
        print('*'*60)
        return svc
        

