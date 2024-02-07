# import ollama

# r1 = 'Panasonic 2-Line Integrated Telephone System - KXTS208W Panasonic 2-Line Integrated Telephone System - KXTS208W/ 3-Way Conference/ One-Touch/Speed Dialer/ Speakerphone/ White Finish'
# r2 = 'Panasonic KX-TS208W Corded Phone 2 x Phone Line(s) - Headset - White'
# query = f"record 1: {r1}, record 2: {r2}. Answer with True. or False."

# stream = ollama.chat(
#     model = 'worker',
#     messages = [
#         {'role': 'user', 'content': query},
#     ],
#     stream = False
# )

# print(stream)

# stream = ollama.generate(
#     model = 'worker',
#     prompt = query,
#     stream = False,
# )

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import ollama

# paths for datasets, candidate pairs and groundtruth files
dataset_1 = 'data/dt2/abt.csv'
dataset_2 = 'data/dt2/buy.csv'
candidate_pairs = 'data/dt2/cp.csv'
groundtruth = 'data/dt2/gt.csv'

# read the files 
dt1 = pd.read_csv(dataset_1, sep='|')
dt2 = pd.read_csv(dataset_2, sep='|')
cp = pd.read_csv(candidate_pairs)
gt = pd.read_csv(groundtruth)

# convert to numpy arrays
dt1 = dt1.to_numpy()
dt2 = dt2.to_numpy()
cp = cp.to_numpy()
gt = gt.to_numpy()

# cut the indexes
dt1 = dt1[:, 1:]
dt2 = dt2[:, 1:]

# concatenate the strings in each column to a single string, omitting empty elements
dt1 = np.array([' '.join([x for x in row if isinstance(x, str)]) for row in dt1])
dt2 = np.array([' '.join([x for x in row if isinstance(x, str)]) for row in dt2])

#main loop: model iterates through each pair and returns its responses
start = time.time()
responses = []
# num_iterations = 400
num_iterations = len(cp)


for i in range(num_iterations):
    dt1_index = cp[i][0]
    dt2_index = cp[i][1]

    r1 = dt1[dt1_index]
    r2 = dt2[dt2_index]

    # print(f"candidate pair {i}")
    # print(f"record 1: {r1}")
    # print(f"record 2: {r2}")

    query = f"record 1: {r1}, record 2: {r2}. Answer with True. or False."

    resp = ollama.chat(
        model = 'worker',
        messages = [{'role': 'user', 'content': query}],
        stream = False
    )
    
    responses.append(resp['message']['content'])

    # print(f"worker's response: {resp['message']['content']}")

    gt_value = 'True' if any((gt == [dt1_index, dt2_index]).all(axis=1)) else 'False'

    # print(f"groundtruth value: {gt_value}")
    # print(f"pair: {[dt1_index, dt2_index]}")
    # print(" ")

end = time.time()

#model's response time
time_seconds = end - start  
hours, remainder = divmod(time_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print("Response Time: {:02}:{:02}:{:.2f}".format(int(hours), int(minutes), seconds))

#model's 'good behavior' response rate
good_responses = sum(response == 'True' or response == 'False' for response in responses)
good_behavior_rate = good_responses / len(responses)
print("Good Behavior Response Rate:", good_behavior_rate)

#model's conflict rate
record = cp[0][1]
count_true = 0
conflicts = 0
conflict_records = 0

for i in range(num_iterations):
    if record == cp[i][1]:
        if responses[i] == 'True':
            count_true += 1
    else:
        if count_true > 1:
            conflict_records += 1
            conflicts += count_true - 1
        record = cp[i][1]
        count_true = 0
        if responses[i] == 'True':
            count_true += 1

if count_true > 1:
    conflict_records += 1
    conflicts += count_true - 1

print("Conflicts:", conflicts)
print("Conflicted Records:", conflict_records)
conflict_rate = conflict_records / (cp[i][1]+1)
print("Conflict Rate per Record:", conflict_rate)

#evaluation metrics
true_labels = [1 if any((gt == pair).all(axis=1)) else 0 for pair in cp[:num_iterations]]
predicted_labels = [1 if resp == 'True' else 0 for resp in responses]

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

#model's responses
for i in range(len(responses)):
    if responses[i] != 'True' and responses[i] != 'False':
        responses[i] = 'False' 

file_path = 'responses_FT_d2.txt'
with open(file_path, 'w') as file:
    for response in responses:
        file.write(response + '\n')
