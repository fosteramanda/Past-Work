import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import Perceptron

iris_data = pd.read_csv("iris_data.csv" ) # , names=column_names )
titanic_data = pd.read_csv("titanic_train2.csv")
titanic_data_test = pd.read_csv("titanic_test2.csv")

titanic_data['Age'] = titanic_data['Age'].fillna(30)
titanic_data_test['Age'] = titanic_data_test.fillna(30)


"""Demonstrate Perception on Iris Data
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris-setosa
      -- Iris-versicolor
      -- Iris-virginica
British statistician and biologist Ronald Fisher
"The use of multiple measurements in taxonomic problems" 
    Annual Eugenics, 7, Part II, 179-188 (1936)
"""

iris_data.info()
print(iris_data.head())
print(iris_data['class'].value_counts())

def iris_to_color(iris):
   if iris=='Iris-setosa':
      return "red"
   elif iris=='Iris-versicolor':
      return "blue"
   else:
      return "green"


def print_conf_matrix(targets, outputs):
    cm = confusion_matrix(targets, outputs)
    print("Confusion Matrix:")
    print("     PN PP")
    print("AN: "+ str(cm[0]))
    print("AP: "+ str(cm[1]))


def class_to_targets(target, iris):
    if target == iris:
        return 1
    else:
        return 0


# Full fit with two inputs

""" This method trains a Perception to classify the Iris type passed in using the inputs passed in. 
"""
def classify_iris(iris_inputs, iris_type):
    iris_inputs = iris_data[ iris_inputs ]
    iris_targets = iris_targets = iris_data['class'].apply(lambda x: class_to_targets(x, iris_type))
    percey = Perceptron()

    print("Training a Perceptron to classify Iris-Virginica using petal inputs.")
    percey.fit(iris_inputs, iris_targets)
    percey_outputs = percey.predict(iris_inputs)

    print("Found weights=" + str(percey.coef_) + " and threshold: " + str(percey.intercept_) + " in " + str(percey.n_iter_) + " epochs.")
    print("Mean accuracy:", percey.score(iris_inputs, iris_targets))
    print_conf_matrix(iris_targets, percey_outputs)
    print("Precision = TP / (TP + FP) = ", precision_score(iris_targets, percey_outputs))
    print("Recall = TP / (TP + FN) = ", recall_score(iris_targets, percey_outputs))


    print("\n\nLet's make it train longer:")
    percey2 = Perceptron(n_iter_no_change=10)
    percey2.fit(iris_inputs, iris_targets)
    percey2_outputs = percey2.predict(iris_inputs)

    print("Found weights=" + str(percey2.coef_) + " and threshold: " + str(percey2.intercept_) + " in " + str(percey2.n_iter_) + " epochs.") 
    print("Mean accuracy:", percey2.score(iris_inputs, iris_targets))
    print_conf_matrix(iris_targets, percey2_outputs)
    print("Precision = TP / (TP + FP) = ", precision_score(iris_targets, percey2_outputs))
    print("Recall = TP / (TP + FN) = ", recall_score(iris_targets, percey2_outputs))

def change_it(value, variable):
    if value == variable:
        return 1
    else:
        return 0

def sex_to_Targets(value):
    if value == 'female':
        return 1
    else:
        return 0
def age_to_Targets(value):
    if value <= 18:
        return 1
    else:
        return 0
def class_to_Targets(value):
    if value == 1:
        return 1
    else: 
        return 0

def SibSp_to_Targets(value):
    if value == 0:
        return 1
    else:
        return 0
        
        
def titanic_predict(inputs):
    titanic_inputs = titanic_data[ inputs ]
    titanic_targets = titanic_data['Survived']
    percey = Perceptron()

    percey.fit(titanic_inputs, titanic_targets)
    percey_outputs = percey.predict(titanic_inputs)

    print("Found weights=" + str(percey.coef_) + " and threshold: " + str(percey.intercept_) + " in " + str(percey.n_iter_) + " epochs.")
    print("Mean accuracy:", percey.score(titanic_inputs, titanic_targets))
    print_conf_matrix(titanic_targets, percey_outputs)
    print("Precision = TP / (TP + FP) = ", precision_score(titanic_targets, percey_outputs))
    print("Recall = TP / (TP + FN) = ", recall_score(titanic_targets, percey_outputs))
    
    titanic_inputs_test = titanic_data_test[inputs]
    titanic_targets_test = titanic_data_test['Survived']
    percey = Perceptron()

    percey.fit(titanic_inputs_test, titanic_targets_test)
    percey_outputs_test = percey.predict(titanic_inputs_test)

    print("Found weights=" + str(percey.coef_) + " and threshold: " + str(percey.intercept_) + " in " + str(percey.n_iter_) + " epochs.")
    print("Mean accuracy:", percey.score(titanic_inputs_test, titanic_targets_test))
    print_conf_matrix(titanic_targets_test, percey_outputs_test)
    print("Precision = TP / (TP + FP) = ", precision_score(titanic_targets_test, percey_outputs_test))
    print("Recall = TP / (TP + FN) = ", recall_score(titanic_targets_test, percey_outputs_test))
   
    """
    print("\n\nLet's make it train longer:")
    percey2 = Perceptron(n_iter_no_change=10)
    percey2.fit(titanic_inputs, titanic_targets)
    percey2_outputs = percey2.predict(titanic_inputs)

    print("Found weights=" + str(percey2.coef_) + " and threshold: " + str(percey2.intercept_) + " in " + str(percey2.n_iter_) + " epochs.") 
    print("Mean accuracy:", percey2.score(titanic_inputs, titanic_targets))
    print(percey2_outputs)
    print(titanic_targets)
    print_conf_matrix(titanic_targets, percey2_outputs)
    print("Precision = TP / (TP + FP) = ", precision_score(titanic_targets, percey2_outputs))
    print("Recall = TP / (TP + FN) = ", recall_score(titanic_targets, percey2_outputs))
    """

""" Main Method 
"""
if __name__ == "__main__":
    classify_iris(['petal width', 'petal length'],"Iris-setosa")
    classify_iris(['sepal width', 'sepal length'], "Iris-virginica")
    classify_iris(['petal width', 'petal length'], "Iris-virginica")
    classify_iris(['petal width', 'petal length', 'sepal width', 'sepal length'], "Iris-virginica")
    
    titanic_data['Mod Sex'] = titanic_data['Sex'].apply(sex_to_Targets)
    titanic_data['Mod Age'] = titanic_data['Age'].apply(age_to_Targets)
    titanic_data['Mod Class'] = titanic_data['Pclass'].apply(class_to_Targets)
    titanic_data['Mod SibSp'] = titanic_data['SibSp'].apply(SibSp_to_Targets)  
    titanic_data_test['Mod Sex'] = titanic_data_test['Sex'].apply(sex_to_Targets)
    titanic_data_test['Mod Age'] = titanic_data_test['Age'].apply(age_to_Targets)
    titanic_data_test['Mod Class'] = titanic_data_test['Pclass'].apply(class_to_Targets)
    titanic_data_test['Mod SibSp'] = titanic_data_test['SibSp'].apply(SibSp_to_Targets)
    
    titanic_predict(['Mod Sex','Mod Age','Mod Class', 'Mod SibSp'])