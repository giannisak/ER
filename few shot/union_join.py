import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

candidate_pairs = '~/ollama/data/dt2/cp.csv'
groundtruth = '~/ollama/data/dt2/gt.csv'
cp = pd.read_csv(candidate_pairs)
gt = pd.read_csv(groundtruth)
cp = cp.to_numpy()
gt = gt.to_numpy()

file_path1 = 'mes_zephyr_e2/responses_TF_d2.txt'

with open(file_path1, 'r') as file1:
    responses1 = [line.strip() for line in file1.readlines()]

file_path2 = 'mes_zephyr_e2/responses_FT_d2.txt'

with open(file_path2, 'r') as file2:
    responses2 = [line.strip() for line in file2.readlines()]

# for i in range(len(responses1)):
#     if responses1[i]!='True' and responses1[i]!='False':
#         print(i)
#         print(responses1[i])
# print('')
# for i in range(len(responses2)):
#     if responses2[i]!='True' and responses2[i]!='False':
#         print(i)
#         print(responses2[i])
# print(len(responses1),len(responses2))

count_tt = sum(1 for r1, r2 in zip(responses1, responses2) if r1 == 'True' and r2 == 'True')
count_tf = sum(1 for r1, r2 in zip(responses1, responses2) if r1 == 'True' and r2 == 'False')
count_ft = sum(1 for r1, r2 in zip(responses1, responses2) if r1 == 'False' and r2 == 'True')


print('T-T:', count_tt)
print('T-F:', count_tf)
print('F-T:', count_ft)

union = []

for r1, r2 in zip(responses1, responses2):
    if r1 == 'True' and r2 == 'True':
        union.append('True')  
    elif r1 == 'True' and r2 == 'False':
        union.append('True')  
    elif r1 == 'False' and r2 == 'True':
        union.append('True')  
    else:
        union.append('False')  

count_true = union.count('True')
count_false = union.count('False')

print('Count of True:', count_true)
print('Count of False:', count_false)

join = []

for r1, r2 in zip(responses1, responses2):
    if r1 == 'True' and r2 == 'True':
        join.append('True')  
    elif r1 == 'True' and r2 == 'False':
        join.append('False')  
    elif r1 == 'False' and r2 == 'True':
        join.append('False')  
    else:
        join.append('False') 

count_true = join.count('True')
count_false = join.count('False')

print('Count of True:', count_true)
print('Count of False:', count_false)

true_labels = [1 if any((gt == pair).all(axis=1)) else 0 for pair in cp]
predicted_labels = [1 if resp == 'True' else 0 for resp in union] #or join

conf_matrix = confusion_matrix(true_labels, predicted_labels)

TP = conf_matrix[1, 1]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("\nTrue Positives:", TP)
print("False Positives:", FP)
print("True Negatives:", TN)
print("False Negatives:", FN)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print(" ")
