import nilearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib.use('MacOSX')
import itertools

from nilearn.image import mean_img
from nilearn.input_data import NiftiMasker
from nilearn import plotting
from sklearn.feature_selection import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def retrieve_and_mask_data(func_filename):
    # create a masker, which normalises the data transforms the 4D image to a 2D image
    masker = NiftiMasker(standardize=True, mask_strategy='template')
    masker.fit(func_filename)
    fmri_masked = masker.fit_transform(func_filename)
    timepoint, n_features = fmri_masked.shape
    print("Dataset summary: \n" + "%d timepoints, %d features" % (timepoint, n_features))
    X = fmri_masked

    return X


def label_data(labels_file):
    # label data as rest or task
    behavioural = pd.read_csv(labels_file, sep=",")
    conditions = behavioural['labels']
    condition_mask = conditions.isin(["rest", "task"])
    # mask the data
    conditions = conditions[condition_mask]
    print("Data has been labelled rest or task")
    Y = conditions[1:]

    return Y


def svm_classification(x, y):
    # define classifier
    svc = SVC(kernel='linear')
    # previous testing showed selection of 3100 best features/ voxels
    feature_selection = SelectKBest(f_classif, k=3100)
    cv = StratifiedKFold(n_splits=15)
    feature_selection2 = SelectPercentile(f_classif, percentile=100)

    # we can plug classifier and feature selection methods together in a *pipeline* that performs the two operations
    anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])

    # Obtain prediction scores via cross validation
    cv_scores = cross_val_score(anova_svc, x, y, cv=cv)
    y_pred = cross_val_predict(anova_svc, x, y, cv=cv)

    # compute mean prediction accuracy of model
    classification_accuracy = cv_scores.mean()
    print("Classification accuracy: %.4f" %
          classification_accuracy)

    return y_pred


def plot_confusion_matrix(x, y, y_pred):
    cm = confusion_matrix(y, y_pred)
    tn = float(cm[0][0])
    tp = float(cm[1][1])
    actual = tn + tp
    print('confusion matrix accuracy: ' + str((actual / float(x.shape[0]))))

    classess = set(y)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classess))
    plt.xticks(tick_marks, classess, rotation=45)
    plt.yticks(tick_marks, classess)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", verticalalignment="center", color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure()
    plt.show()


def main():
    dataset = "M:\ce301\ce301_dudusola\data\subject1\swarfMP_BBCI_VPTAQ-0002-*.img"
    labels = "M:\ce301\ce301_dudusola\labels.csv"

    # # Visualising the fmri volume data of the patient (4D brain) in a single plot
    plotting.view_img(mean_img(dataset), threshold=None, output_file='mean _brain_image.png')
    plotting.show()

    # # Compute the mean of the images (in the time dimension of 4th dimension)
    mean = nilearn.image.mean_img(dataset)
    plotting.plot_epi(mean, title='Mean EPI Image')
    plotting.show()

    # # observe mask as a plot which cuts the image into 3 - frontal, axial, lateral
    # plotting.plot_roi(masker.mask_img_, mean,
    #          title="Mask from already masked data")
    # plotting.show()

    x = retrieve_and_mask_data(dataset)
    y = label_data(labels)
    y_pred = svm_classification(x, y)
    # visual analysis graph
    plot_confusion_matrix(x, y, y_pred)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)

# ##########################################################################################
# GRAPHS TO EVALUATE BEST K FOLD AND BEST PERCENTILE

# scale = [n for n in range(1, 65797) if n % 1000 == 0]
# voxels = [n for n in range(3000, 3300) if n % 10 == 0]
# cv_scores = []
# scoring = {}
# #
# for i in voxels:
#     X_new = SelectKBest(f_classif, k=i).fit_transform(X, Y)
#     print(X_new.shape)
#     cv_score = cross_val_score(svc, X, Y, cv=cv)
#     print(cv_score.mean())
#     cv_scores.append(cv_score.mean())
#     scoring[X_new.shape] = cv_score.mean()
#
# # best k value is 3100
# scoring = sorted(scoring.items(), key=lambda x: x[1], reverse=True)
# print(next(iter(scoring)))

# # plot graph of cross validation score against number of features for X
# plt.plot(cv_scores)
# plt.title(
#     'Performance of the SVM varying the number of features/voxels selected')
# plt.xlabel('Number of features x1000')
# plt.ylabel('Prediction score')
# plt.show()

# # Plot the cross-validation score as a function of percentile of features
# score_means = list()
# score_stds = list()
# percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
#
# for percentile in percentiles:
#     anova_svc.set_params(anova__percentile=percentile)
#     # Compute cross-validation score using 1 CPU
#     this_scores = cross_val_score(anova_svc, X, Y, n_jobs=1)
#     score_means.append(this_scores.mean())
#     score_stds.append(this_scores.std())
#
# plt.errorbar(percentiles, score_means, np.array(score_stds), ecolor='red')
#
# plt.title(
#     'Performance of the SVM-Anova varying the percentile of features selected')
# plt.xlabel('Percentile')
# plt.ylabel('Prediction score')
#
# plt.axis('tight')
# plt.show()

# ##########################################################################################
# # plot svm model weights in the brain
# coef_ = svc.coef_
# coef_img = masker.inverse_transform(coef_)
#coef_img.to_filename('model1_svc_weights.nii.gz')
#
# # plot which cuts the image into 3 - frontal, axial, lateral
# plotting.plot_stat_map(coef_img, bg_img=mean, title="SVM weights", display_mode="yx")
# plotting.show()
#
