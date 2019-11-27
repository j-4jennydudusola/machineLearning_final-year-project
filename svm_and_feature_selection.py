import matplotlib
matplotlib.use('MacOSX')

from nilearn.masking import *
from nilearn.image import smooth_img

filename = '~/PycharmProjects/project3/data/swarfMP_BBCI_VPTAQ-0002-*.img'
results_img = smooth_img([filename],5)
mask_img = compute_epi_mask(results_img)
masked_data = apply_mask(results_img, mask_img)


import pandas as pd
behavioural = pd.read_csv("labels.csv", sep=",")
conditions = behavioural['labels']

condition_mask = conditions.isin(["rest", "task"])
masked_data = masked_data[condition_mask[1:]]
print(masked_data.shape)

conditions = conditions[condition_mask]

# from nilearn.input_data import NiftiMasker
# filename = './data/swarfMP_BBCI_VPTAQ-0002-*.img'
#
# from os import listdir
# import re
# files = [f for f in listdir('/Users/jennydudusola/PycharmProjects/ce301_dudusola_j/data') if re.findall(r'.img$', f)]
# for file in files:
#     functional_file = "./data/" + file
#
#     nifti_masker = NiftiMasker(mask_img=functional_file, smoothing_fwhm=4, standardize=True)
#     func_filename = functional_file
#     X = apply_mask(files, compute_epi_mask(func_filename))
#     print(X)
#

X = masked_data
Y = conditions[1:]

from sklearn.svm import SVC
svc = SVC(kernel='linear')

from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
feature_selection = SelectPercentile(f_classif, percentile=0.8)
from sklearn.pipeline import Pipeline
anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])


# anova_svc.fit(X, Y)
# y_pred = anova_svc.predict(X)

from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(anova_svc, X, Y)
total_cv_score = 0.0
for i in cv_score:
    total_cv_score += i

mean_cv_score = total_cv_score / 3
print("svm accuracy: " + str(mean_cv_score))



# # Use the mean image as a background to avoid relying on anatomical data
# from nilearn import image
# mean_img = image.mean_img(filename)
# # Create the figure
# from nilearn.plotting import plot_stat_map, show
# plot_stat_map(mean_img, title='SVM weights')
# show()

feature_selection2 = SelectKBest(f_classif, k=10)
anova_svc2 = Pipeline([('anova', feature_selection2), ('svc', svc)])

cv_score2 = cross_val_score(anova_svc2, X, Y)
total_cv_score2 = 0.0
for i in cv_score2:
    total_cv_score2 += i

mean_cv_score2 = total_cv_score2 / 3
print("svm accuracy: " + str(mean_cv_score2))



import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, feature_selection
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

transform = feature_selection.SelectPercentile(feature_selection.f_classif)
clf = Pipeline([('anova', transform), ('svc', svm.SVC(C=1.0))])
# Plot the cross-validation score as a function of percentile of features
score_means = list()
score_stds = list()
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    # Compute cross-validation score using 1 CPU
    this_scores = cross_val_score(clf, X, Y, n_jobs=1)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

plt.errorbar(percentiles, score_means, np.array(score_stds))

plt.title(
    'Performance of the SVM-Anova varying the percentile of features selected')
plt.xlabel('Percentile')
plt.ylabel('Prediction rate')

plt.axis('tight')
plt.show()
