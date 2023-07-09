import pickle
import logistic_Regression_file
import os
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


class lr:
    y_pred_1 = []

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test

    def load(self):
        if os.path.exists("file.pickle"):
            print("Loading trained model")
            model_lr = pickle.load(open("file.pickle", "rb"))

            self.y_pred_1 = model_lr.predict(self.x_test)
            lr_acc = accuracy_score(self.y_test, self.y_pred_1) * 100
            print("Logistic Regression  Accuracy:", lr_acc, "%")
            print("Precision Logistic Regression:", metrics.precision_score(self.y_test, self.y_pred_1) * 100, "%")
            # Model Recall: what percentage of positive tuples are labelled as such?
            print("Recall Logistic Regression:", metrics.recall_score(self.y_test, self.y_pred_1) * 100, "%")
            # Model Accuracy, Correctness of the model
            print("Confusion_matrix :")
            print(confusion_matrix(self.y_test, self.y_pred_1))
            LogisticRegression_Report = classification_report(self.y_test, self.y_pred_1)  # Report
            print("Logistic Regression Report :")
            print(LogisticRegression_Report, sep='\n')

            print("File LR prediction:")
            print(self.y_pred_1)

        else:
            print("training model")
            logicR = logistic_Regression_file.Logistic_Regression(self.X_train, self.y_train, self.x_test, self.y_test)
            model_lr = logicR.logic()
            with open("file.pickle", "wb") as file:
                pickle.dump(model_lr, file)
