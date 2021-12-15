import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math

# Helper functions

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

def print_conf_matrix(targets, outputs):
    cm = confusion_matrix(targets, outputs)
    print("Confusion Matrix:")
    print("     PN PP")
    print("AN: "+ str(cm[0]))
    print("AP: "+ str(cm[1]))

def print_class_results(targets, outputs):
	print_conf_matrix(targets, outputs)

	# Precision - How accurate are the positive predictions?
	print("Precision (TP / (TP + FP)):", precision_score(targets, outputs))

	# Recall - How correctly are positives predicted?
	print("Recall (TP / (TP + FN)):", recall_score(targets, outputs))

# Logistic Regression (even though it is a classifier)
def ink_Top_Third(row):
    return 1/ ( 1 + math.exp(row[:261].sum() /1000)) * 255 #784 (total rows)/3 
    
def ink_Total(row):
    return 1/ ( 1 + math.exp(row[:784].sum() /5000)) * 255


print("Original Mean Accuracy: 0.9385666666666667")
print("Original Confusion Matrix")
original_confusion_matrix = [[5788, 1, 13, 9, 10, 25, 29, 7, 37, 4],
                             [1, 6599, 30, 15, 7, 14, 3, 11, 50, 12],
                             [17, 16, 115, 5614, 6, 157, 14, 40, 108, 44],
                             [11, 20, 20, 9, 5521, 7, 43, 15, 32, 164],
                             [41, 16, 40, 152, 41, 4846, 81, 12, 151, 41],
                             [30, 9, 36, 2, 29, 53, 5733, 2, 21, 3],
                             [6, 17, 56, 23, 38, 7, 4, 5916, 18, 180],
                             [28, 84, 50, 121, 17, 141, 34, 19, 5298, 59],
                             [19, 21, 14, 67, 118, 32, 3, 131, 43, 5501]],
print(original_confusion_matrix)    



from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(penalty='none', solver='saga', random_state=42)

# MNIST data set of handwritten digits
mnist_train = pd.read_csv("mnist_train.csv",index_col=False,header=None)
mnist_test = pd.read_csv("mnist_test.csv",index_col=False,header=None)

mnist_train_targets = mnist_train[0]
mnist_train_inputs = mnist_train[mnist_train.columns[1:]]

mnist_test_targets = mnist_test[0]
mnist_test_inputs = mnist_test[mnist_test.columns[1:]]

# Now let's shuffle the training set to reduce bias opportunities
from sklearn.utils import shuffle
smn_train_targets, smn_train_inputs = shuffle(mnist_train_targets, mnist_train_inputs, random_state=42)

# Let's try our Logistic Classifier on the MNIST data, predicting digit 5
from sklearn.linear_model import LogisticRegression

smn_train_target5 = (smn_train_targets==5)
smn_test_target5 = (mnist_test_targets==5)

print("Training...")
log_reg.fit(smn_train_inputs, smn_train_target5)
smn_train_outputs5 = log_reg.predict(smn_train_inputs)
# Classification error metrics:
print("Mean accuracy:", log_reg.score(smn_train_inputs, smn_train_target5))
print_class_results(smn_train_target5, smn_train_outputs5)

# But how does it perform on the test set?
print()
print("And on the test set...")
smn_test_outputs5 = log_reg.predict(mnist_test_inputs)
print("Mean accuracy:", log_reg.score(mnist_test_inputs, smn_test_target5))
print_class_results(smn_test_target5, smn_test_outputs5)


# Softmax Regression or Multinomial Logistic Regression!
mnist_test_inputs['total ink'] = mnist_test_inputs.apply(ink_Total, axis= 1)
mnist_test_inputs['top third ink'] = mnist_test_inputs.apply(ink_Top_Third, axis= 1)
mnist_train_inputs['total ink'] = mnist_train_inputs.apply(ink_Total, axis= 1)
mnist_train_inputs['top third ink'] = mnist_train_inputs.apply(ink_Top_Third, axis= 1)

print("Training a Multinomial Logistic Regression classifier for ALL digits!")
softmax_reg = LogisticRegression(penalty="none",multi_class="multinomial", solver="saga")
softmax_reg.fit(mnist_train_inputs, mnist_train_targets)
softmax_outputs = softmax_reg.predict(mnist_test_inputs)
print("Mean accuracy:")
print(softmax_reg.score(mnist_test_inputs, mnist_test_targets))
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(mnist_test_targets, softmax_outputs))

print(mnist_test_inputs.iloc[0].sum())


#mnist_train_inputs['InkMe'] = mnist_train_inputs.apply(sum, axis = 1)
#print(mnist_train_inputs['InkMe'][:10],mnist_train_targets[:10])
#for (bar) in foo:
   # print(bar)
