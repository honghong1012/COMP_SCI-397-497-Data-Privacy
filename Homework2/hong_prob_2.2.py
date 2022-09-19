
# coding: utf-8

# In[137]:


import diffprivlib.models as dp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[138]:


dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# # Train logistic regreesion model

# In[91]:


clf = LogisticRegression(solver="lbfgs", max_iter=10000)
clf.fit(X_train, y_train)


# In[92]:


from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression
art_classifier = ScikitlearnLogisticRegression(clf)

base_model_accuracy = clf.score(X_test, y_test)

print('Base model accuracy: ', base_model_accuracy)


# # Attack

# ## blackbox

# ### train attack model

# In[93]:


from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

attack_train_ratio = 0.5
attack_train_size = int(len(X_train) * attack_train_ratio)
attack_test_size = int(len(X_test) * attack_train_ratio)

attack = MembershipInferenceBlackBox(art_classifier, attack_model_type='rf') 

#train attack model
attack.fit(X_train[:attack_train_size], y_train[:attack_train_size],
           X_test[:attack_test_size], y_test[:attack_test_size])


# ### infer membership and check accuracy

# In[94]:


# infer attacked feature
inferred_train = attack.infer(X_train[attack_train_size:], y_train[attack_train_size:])
inferred_test = attack.infer(X_test[attack_test_size:], y_test[attack_test_size:])

# check accuracy
train_acc = np.sum(inferred_train) / len(inferred_train)
test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
print('attack accuracy on training data: ', train_acc)
print('attack accuracy on test data: ', test_acc)
print('overall attack accuracy: ', acc)


# this mieans that for 61% of the data, membership status is inferred correctly(a little bit better than a coin flip)

# In[95]:


def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1
    
    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall

# rule-based
print('precision and recall: ', calc_precision_recall(np.concatenate((inferred_train, inferred_test)), 
                            np.concatenate((np.ones(len(inferred_train)), np.zeros(len(inferred_test))))))


# # Train dfferentially private model

# In[117]:


import diffprivlib.models as dp

dp_model = dp.LogisticRegression(epsilon=2, data_norm=100,max_iter=10000)
dp_model.fit(X_train, y_train)
# print('norm: ', np.linalg.norm(x_train) )

dp_art_model = ScikitlearnLogisticRegression(dp_model)
print('DP model accuracy: ', dp_model.score(X_test, y_test))


# ## Black-box attack

# In[118]:


dp_attack = MembershipInferenceBlackBox(dp_art_model, attack_model_type='rf')

# train attack model
dp_attack.fit(X_train[:attack_train_size].astype(np.float32), (y_train[:attack_train_size]),
              X_test[:attack_test_size].astype(np.float32), (y_test[:attack_test_size]))
# infer 
dp_inferred_train = dp_attack.infer(X_train.astype(np.float32)[attack_train_size:], y_train[attack_train_size:])
dp_inferred_test = dp_attack.infer(X_test.astype(np.float32)[attack_test_size:], y_test[attack_test_size:])
# check accuracy
dp_train_acc = np.sum(dp_inferred_train) / len(dp_inferred_train)
dp_test_acc = 1 - (np.sum(dp_inferred_test) / len(dp_inferred_test))
dp_acc = (dp_train_acc * len(dp_inferred_train) + dp_test_acc * len(dp_inferred_test)) / (len(dp_inferred_train) + len(dp_inferred_test))
print('attack accuracy on training data: ', dp_train_acc)
print('attack accuracy on test data: ', dp_test_acc)
print('overall attack accuracy: ', dp_acc)

print('precision and recall: ', calc_precision_recall(np.concatenate((dp_inferred_train, dp_inferred_test)), 
                            np.concatenate((np.ones(len(dp_inferred_train)), np.zeros(len(dp_inferred_test))))))


# In[141]:


accuracy = []
attack_accuracy = []
epsilons = [ 0.5, 1.0, 5.0, 10.0, 25.0, 75.0, 100.0, 200.0]

for eps in epsilons:
    print(eps)
    dp_clf = dp.LogisticRegression(epsilon=eps, data_norm=100)
    dp_clf.fit(X_train, y_train)
    accuracy.append(dp_clf.score(X_test, y_test))
    
    dp_art_classifier = ScikitlearnLogisticRegression(dp_clf)
    dp_attack = MembershipInferenceBlackBox(dp_art_classifier, attack_model_type='rf')
    
    dp_attack.fit(X_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size].astype(np.float32),
                  X_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size].astype(np.float32))
    dp_inferred_train = dp_attack.infer(X_train.astype(np.float32)[attack_train_size:], y_train[attack_train_size:])
    dp_inferred_test = dp_attack.infer(X_test.astype(np.float32)[attack_test_size:], y_test[attack_test_size:])
    dp_train_acc = np.sum(dp_inferred_train) / len(dp_inferred_train)
    dp_test_acc = 1 - (np.sum(dp_inferred_test) / len(dp_inferred_test))
    
    dp_acc = (dp_train_acc * len(dp_inferred_train) + dp_test_acc * len(dp_inferred_test)) / (len(dp_inferred_train) + len(dp_inferred_test))
    attack_accuracy.append(dp_acc)


# In[143]:


import matplotlib.pyplot as plt

plt.plot(epsilons, accuracy)
plt.plot(epsilons, np.ones_like(epsilons) * base_model_accuracy, dashes=[2,2], label="base model")
plt.title("Differentially private logistic regression")
plt.xlabel("epsilon")
plt.ylabel("Model accuracy")
plt.ylim(0, 1)
plt.xlim(0.1, 200)
plt.show() 


# In[144]:


plt.plot(epsilons, attack_accuracy)
plt.plot(epsilons, np.ones_like(epsilons) * acc, dashes=[2,2], label="base model")
plt.title("Differentially private logistic regression")
plt.xlabel("epsilon")
plt.ylabel("Attack accuracy")
plt.ylim(0, 1)
plt.xlim(0.1, 200)
plt.show()


# In[147]:


dp_model = dp.LogisticRegression(epsilon=100, data_norm=100)
dp_model.fit(X_train, y_train)

dp_art_classifier = ScikitlearnLogisticRegression(dp_model)
print('DP model accuracy with eps=100: ', dp_model.score(X_test, y_test))

dp_attack = MembershipInferenceBlackBox(dp_art_classifier, attack_model_type='rf')
dp_attack.fit(X_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size].astype(np.float32),
              X_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size].astype(np.float32))
dp_inferred_train = dp_attack.infer(X_train.astype(np.float32)[attack_train_size:], y_train.astype(np.float32)[attack_train_size:])
dp_inferred_test = dp_attack.infer(X_test.astype(np.float32)[attack_test_size:], y_test.astype(np.float32)[attack_test_size:])
dp_train_acc = np.sum(dp_inferred_train) / len(dp_inferred_train)
dp_test_acc = 1 - (np.sum(dp_inferred_test) / len(dp_inferred_test))
dp_acc = (dp_train_acc * len(dp_inferred_train) + dp_test_acc * len(dp_inferred_test)) / (len(dp_inferred_train) + len(dp_inferred_test))
    
print('DP model attack accuracy with eps=100: ', dp_acc)

