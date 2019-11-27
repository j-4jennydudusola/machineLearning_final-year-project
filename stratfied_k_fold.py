import matplotlib

matplotlib.use('MacOSX')

from nilearn.masking import *
from nilearn.image import smooth_img

filename = '~/PycharmProjects/project3/data/swarfMP_BBCI_VPTAQ-0002-*.img'
results_img = smooth_img([filename], 5)
mask_img = compute_epi_mask(results_img)
masked_data = apply_mask(results_img, mask_img)

import pandas as pd

behavioural = pd.read_csv("labels.csv", sep=",")
conditions = behavioural['labels']

condition_mask = conditions.isin(["rest", "task"])
masked_data = masked_data[condition_mask[1:]]
conditions = conditions[condition_mask]

X = masked_data
Y = conditions[1:]

from sklearn.svm import SVC

svc = SVC(kernel='linear')

from sklearn.feature_selection import SelectPercentile, f_classif

feature_selection = SelectPercentile(f_classif, percentile=5)
from sklearn.pipeline import Pipeline

anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=3)

cv_score = cross_val_score(anova_svc, X, Y, cv=cv)
total_cv_score = 0.0
for i in cv_score:
    total_cv_score += i

mean_cv_score = total_cv_score / 3
print("svm accuracy with stratified 3k fold: " + str(mean_cv_score))

print()
cv_score2 = cross_val_score(anova_svc, X, Y)
total_cv_score2 = 0.0
for i in cv_score2:
    total_cv_score2 += i

mean_cv_score2 = total_cv_score2 / 3
print("svm accuracy with normal 3k fold: " + str(mean_cv_score2))

from sklearn.model_selection import KFold
cv3 = KFold(n_splits=5)
total_cv_scores3 = 0.0
for train, test in cv3.split(X=X, y=Y):
    conditions_masked = conditions[1:].values[train]
    svc.fit(masked_data[train], conditions_masked)
    prediction = svc.predict(masked_data[test])
    total_cv_scores3 += ((prediction == conditions[1:].values[test]).sum() / float(len(conditions[1:].values[test])))

mean_cv_score3 = total_cv_scores3 / 5
print()
print("svm accuracy with normal 5k fold: " + str(mean_cv_score3))

cv4 = StratifiedKFold(n_splits=5)
cv_score4 = cross_val_score(anova_svc, X, Y, cv=cv4)
total_cv_score4 = 0.0
for i in cv_score4:
    total_cv_score4 += i

mean_cv_score4 = total_cv_score4 / 5
print()
print("svm accuracy with stratified 5k fold: " + str(mean_cv_score4))

