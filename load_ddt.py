import os.path
import pickle
import DecisionTree_file
import os
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


class tree_load:
    y_pred_2 = []

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def load(self):
        if os.path.exists("file_dt.pickle"):
            print("Loading trained model")
            model_dt = pickle.load(open("file_dt.pickle", "rb"))

            self.y_pred_2 = model_dt.predict(self.X_test)
            self.y_pred_2 = model_dt.predict(self.X_test)
            dt_acc = accuracy_score(self.y_test, self.y_pred_2) * 100
            print("Decision Tree Accuracy:", dt_acc, "%")
            print("Precision Decision Tree:", metrics.precision_score(self.y_test, self.y_pred_2) * 100, "%")
            # Model Recall: what percentage of positive tuples are labelled as such?
            print("Recall Decision Tree:", metrics.recall_score(self.y_test, self.y_pred_2) * 100, "%")
            # Model Accuracy, Correctness of the model
            print("Confusion_matrix :")
            print(confusion_matrix(self.y_test, self.y_pred_2))
            dt_Report = classification_report(self.y_test, self.y_pred_2)  # Report
            print("Decision Tree Report :")
            print(dt_Report, sep='\n')

            print("File DT prediction:")
            print(self.y_pred_2)

        else:
            print("training model")
            Decision = DecisionTree_file.DecisionTree(self.X_train, self.y_train, self.X_test, self.y_test)
            model_dt = Decision.Tree()
            with open("file_dt.pickle", "wb") as file:
                pickle.dump(model_dt, file)
