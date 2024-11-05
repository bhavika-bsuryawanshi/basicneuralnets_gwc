# import libraries
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix, classification_report
from collections import defaultdict

# load the training dataset
train_data = pd.read_csv('fashion_mnist_20bal_train.csv')

# separate the data (features) and the classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

# load the testing dataset
validate_data = pd.read_csv('fashion_mnist_20bal_test.csv')

# split data for testing and training
X_train, X_validate, y_train, y_validate = train_test_split(
    X_train, y_train,
    test_size=20,       
    random_state=2,       
    stratify=y_train      
)

# separate the data (features) and the  classes
X_validate = validate_data.drop('class', axis=1)  # Features (all columns except the first one)
X_validate = X_validate / 255.0
y_validate = validate_data['class']   # Target (first column)

neural_net_model = MLPClassifier( hidden_layer_sizes=(20),random_state=42,tol=0.005)
neural_net_model.fit(X_train, y_train)

# determine model architecture
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Layer sizes: {layer_size_str}")

# predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_validate)

# rreate dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# count correct test predictions for each class
for true, pred in zip(y_validate, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# for comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1


# calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] *100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_validate)*100
print(f"Overall Test Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training*100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")
