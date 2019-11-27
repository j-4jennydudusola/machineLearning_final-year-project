import matplotlib
matplotlib.use('MacOSX')

from nilearn.masking import *
from nilearn.image import smooth_img

filename = '~/PycharmProjects/project3/data/swarfMP_BBCI_VPTAQ-0002-*.img'
results_img = smooth_img([filename],5)
mask_img = compute_epi_mask(results_img)
masked_data = apply_mask(results_img, mask_img)

# import matplotlib.pyplot as plt
# plt.plot(masked_data[:, :3])
# plt.show()

import pandas as pd
behavioural = pd.read_csv("labels.csv", sep=",")
conditions = behavioural['labels']

condition_mask = conditions.isin(["rest", "task"])
masked_data = masked_data[condition_mask[1:]]
print(masked_data.shape)

conditions = conditions[condition_mask]

print(conditions.shape) # there is one extra line in conditions for titles
# perfect svm
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(masked_data, conditions[1:])
#prediction = svc.predict(masked_data)
# print(prediction)

# use some of the data for prediction and some for training
svc.fit(masked_data[:-30], conditions[1:-30])
prediction = svc.predict(masked_data[-30:])
# print(prediction)
# accuracy
# print((prediction == conditions[-30:]).sum() / float(len(conditions[-30:])))

# implementing Kfold loop - unecessary
# from sklearn.model_selection import KFold
# #
# # cv = KFold(n_splits=5)
# for train, test in cv.split(X=masked_data):
#     conditions_masked = conditions[1:].values[train]
#     svc.fit(masked_data[train], conditions_masked)
#     prediction = svc.predict(masked_data[test])
#     print((prediction == conditions[1:].values[test]).sum() / float(len(conditions[1:].values[test])))

# cross validation with scikit-learn
# by default uses a 3 fold Kfold
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(svc, masked_data, conditions[1:])
# print(cv_score)
total_cv_score = 0.0
for i in cv_score:
    total_cv_score += i

mean_cv_score = total_cv_score / 3
print("svm accuracy: " + str(mean_cv_score))



