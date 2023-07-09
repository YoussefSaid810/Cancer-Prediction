import os.path
import pickle
import Support_vector_machine
import os
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


class svm_load:
    y_pred_3 = []

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def load(self):

        if os.path.exists("file_svm.pickle"):
            print("Loading trained model")
            model_svm = pickle.load(open("file_svm.pickle", "rb"))
            self.y_pred_3 = model_svm.predict(self.X_test)
            svm_acc = accuracy_score(self.y_test, self.y_pred_3) * 100
            print("SVM Accuracy:", svm_acc, "%")
            print("Precision SVM:", metrics.precision_score(self.y_test, self.y_pred_3) * 100, "%")
            # Model Recall: what percentage of positive tuples are labelled as such?
            print("Recall SVM:", metrics.recall_score(self.y_test, self.y_pred_3) * 100, "%")
            # Model Accuracy, Correctness of the model
            print("Confusion_matrix :")
            print(confusion_matrix(self.y_test, self.y_pred_3))
            svm_Report = classification_report(self.y_test, self.y_pred_3)  # Report
            print("SVM Report :")
            print(svm_Report, sep='\n')

            print("File SVM prediction:")
            print(self.y_pred_3)
            print('*' * 60)

        else:
            print("training model")
            vector_machine = Support_vector_machine.SVM(self.X_train, self.y_train, self.X_test, self.y_test)
            model_svm = vector_machine.svms()
            with open("file_svm.pickle", "wb") as file:
                pickle.dump(model_svm, file)
