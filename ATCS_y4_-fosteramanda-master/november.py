import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

def runTest(name, model, train_inputs, train_targets, test_inputs, test_targets, avg='binary'):
    model.fit(train_inputs, train_targets)
    test_outputs = model.predict(test_inputs)
    print("\n" + name + ", Mean test accuracy: ", model.score(test_inputs, test_targets))
    print("Precision = TP / (TP + FP) = ", precision_score(test_targets, test_outputs, average=avg))
    print("Recall = TP / (TP + FN) = ", recall_score(test_targets, test_outputs, average=avg))
    print("Confusion Matrix (test):")
    print(confusion_matrix(test_targets, test_outputs))

    
#MY WORK FOR QUESTION #1
print("\n\nQUESTION #1")
mnist_train = pd.read_csv("mnist_train.csv",index_col=False,header=None)
mnist_test = pd.read_csv("mnist_test.csv",index_col=False,header=None)


# Now let's shuffle the training set to reduce bias opportunities
train_targets, train_inputs = shuffle(mnist_train[0], mnist_train[mnist_train.columns[1:]], random_state=42)

#17,1
# for depth in range (10,41,10) --> values that had a depth of 20 had the best overall mean accuracy 
# for depth in range (15, 25) --> depth 16 has the best overall mean accuracy
# for depth in range (16, 18) --> depth 17 performed best and 1 leaf
# best combination I found was: 17 (depth), 1 (min samples leaves)

#for depth in range (16, 18):
 #   for leaf in range(1, 10):
tree = DecisionTreeClassifier( max_depth = 17, min_samples_leaf = 1)
runTest('Decision Tree',
        tree,
        train_inputs,
        train_targets,
        mnist_test[mnist_test.columns[1:]],
        mnist_test[0],
        'weighted')
print('Used depth = ', tree.get_depth(),' and num leaves = ',tree.get_n_leaves())


#WORK FOR QUESTION #2
print("\n\nQUESTION #2")
titanic_data = pd.read_csv("titanic_train2.csv")
titanic_data_test = pd.read_csv("titanic_test2.csv")

titanic_data['Age'] = titanic_data['Age'].fillna(30)
titanic_data_test['Age'] = titanic_data_test.fillna(30)

titanic_data['Mod Sex'] = titanic_data['Sex'].apply(lambda x: 1 if x=='female' else 0)
titanic_data['Mod Age'] = titanic_data['Age'].apply(lambda x: 1 if x<=18 else 0)
titanic_data['Mod Class'] = titanic_data['Pclass'].apply(lambda x: 1 if x==1 else 0)
titanic_data['Mod SibSp'] = titanic_data['SibSp'].apply(lambda x: 1 if x==0 else 0)  
titanic_data_test['Mod Sex'] = titanic_data_test['Sex'].apply(lambda x: 1 if x=='female' else 0)
titanic_data_test['Mod Age'] = titanic_data_test['Age'].apply(lambda x: 1 if x<=18 else 0)
titanic_data_test['Mod Class'] = titanic_data_test['Pclass'].apply(lambda x: 1 if x==1 else 0)
titanic_data_test['Mod SibSp'] = titanic_data_test['SibSp'].apply(lambda x: 1 if x==0 else 0)

train_columns = ['Mod Sex', 'Mod Age', 'Mod Class', 'Mod SibSp']
train_data = titanic_data[['Mod Sex', 'Mod Age', 'Mod Class', 'Mod SibSp']]

test_columns = ['Mod Sex', 'Mod Age', 'Mod Class', 'Mod SibSp']
test_data = titanic_data_test[['Mod Sex', 'Mod Age', 'Mod Class', 'Mod SibSp']]

tree = DecisionTreeClassifier( max_depth = 17, min_samples_leaf = 1)
runTest('Decision Tree classifier for Titanic',
        tree,
        pd.DataFrame(train_data, columns = train_columns),
        pd.Series(titanic_data['Survived']) ,
        pd.DataFrame(test_data, columns = test_columns),
        pd.Series(titanic_data_test['Survived']))
print('Used depth = ', tree.get_depth(),' and num leaves = ',tree.get_n_leaves())

# QUESTION #3
print("\n\nQUESTION #3")
train = pd.read_csv("pulsar_train.csv")
test = pd.read_csv("pulsar_test.csv")
# Useful model parameters to explore:
#     n_neighbors (defaults to 5)
#     weights (defaults to 'uniform', can also be 'distance')

for (name, model) in [
        ("K Nearest Neighbors classifier", KNeighborsClassifier()),
        ("Logistic Regression", LogisticRegression(solver ="saga")),
        ("Decision Tree", DecisionTreeClassifier(max_depth = 9, min_samples_leaf = 9))]:
    runTest(name,
            model,
            train.drop('Pulsar', axis=1),
            train['Pulsar'],
            test.drop('Pulsar',axis=1),
            test['Pulsar'])

#Question #4
print("\n\nQUESTION #4")
#CANCER DATA SET
train = pd.read_csv("cancer_train.csv")
test = pd.read_csv("cancer_test.csv")

train['Bare Nuclei'] = train['Bare Nuclei'].replace('?', '1')
test['Bare Nuclei'] = test['Bare Nuclei'].replace('?','1')

tree = DecisionTreeClassifier(max_depth = 9, min_samples_leaf = 9)

for (name, model) in [
        ("K Nearest Neighbors classifier", KNeighborsClassifier()),
        ("Logistic Regression", LogisticRegression(solver ="saga")),
        ("Decision Tree", tree)]:
    runTest(name,
            model,
            train.drop('Class', axis=1),
            train['Class'],
            test.drop('Class',axis=1),
            test['Class'],
            'weighted')

print('Decision Tree fit with depth = ', tree.get_depth(),' and num leaves = ',tree.get_n_leaves())

##Decision Tree Preformed Best
##Some features that are very influential and some that don't say much
##Decision Trees beats K Nearest Neighbors when there are obvious "swicthes" or turning points
##Theasholds that mean cancer is present --> swicthes when you reach a certain number existed a lot making Decision Tree Best













