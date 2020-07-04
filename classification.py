import matplotlib
import nilearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('MacOSX')
import itertools
import glob
import sklearn
import pickle

from nilearn.image import mean_img
from nilearn.input_data import NiftiMasker
from nilearn import plotting
from sklearn import feature_selection
from sklearn.feature_selection import *
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR


def retrieve_and_mask_data(func_filename):
    # create a masker, which normalises the data transforms the 4D image to series of 2D images
    masker = NiftiMasker(standardize=True, mask_strategy='template')
    # fit masker object to data
    masker.fit(func_filename)
    # produce a mask image
    mask_img = masker.mask_img_
    # produce a mean image of all 3D files for a subject
    mean_functional_img = mean_img(func_filename)
    fmri_masked = masker.fit_transform(func_filename)
    timepoint, n_features = fmri_masked.shape
    print("Dataset 2D summary: \n" + "%d timepoints, %d features" % (timepoint, n_features))
    X = fmri_masked

    return X, mask_img, mean_functional_img

def plot_masks(mask, mean):
    # plot mask image
    plotting.plot_roi(mask, mean, display_mode='y', title='Mask', cut_coords=1)
    # plot mean image
    plotting.plot_epi(mean, title="Mean EPI Image")
    plotting.show()

    return None

def label_data(labels_file):
    # read csv file containing true labels for each image
    behavioural = pd.read_csv(labels_file, sep=",")
    # assign to conditions list
    conditions = behavioural['labels']
    print("Data has been labelled")
    Y = conditions[1:]

    return Y

def feature_Selection(n):
    # set feature selection method to specific k value
    k_best_features = SelectKBest(f_classif, k=n)
    percentage_features = SelectPercentile(f_classif, percentile=100)

    return k_best_features

def leave_one_out_classification(classifier, x, y, classifier_name):
    print("{} using 'leave one group out' cross validation".format(classifier_name))
    # initiate leave one out object
    loo_cv = LeaveOneOut()
    # compute cross validation score  - accuracy of classifier after leave one out cv has occurred
    loocv_scores = cross_val_score(classifier, x, y, cv=loo_cv)
    # obtain the predicted class labels as a list of labels
    y_pred = cross_val_predict(classifier, x, y, cv=loo_cv)
    # mean accuracy of each score for each fold of cross validation
    loo_clf_accuracy = loocv_scores.mean() * 100.0
    # standard deviation term obtained as a percentage
    loo_clf_std = loocv_scores.std() * 100.0
    print("Classification accuracy: {}, {}".format(loo_clf_accuracy, loo_clf_std))

    return loo_clf_accuracy, loo_clf_std, y_pred

def stratifiedkfold(classifier, x, y, classifier_name):
    print("{} using 'stratified fold' cross validation".format(classifier_name))
    # initiate stratified 15 fold validation technique
    cv = StratifiedKFold(n_splits=15)
    # compute cross validation score  - accuracy of classifier after leave one out cv has occurred
    cv_scores = cross_val_score(classifier, x, y, cv=cv)
    # obtain the predicted class labels as a list of labels
    y_pred = cross_val_predict(classifier, x, y, cv=cv)
    # mean accuracy of each score for each fold of cross validation
    sf_clf_accuracy = cv_scores.mean() * 100.0
    # standard deviation term obtained as a percentage
    sf_clf_std = cv_scores.std() * 100.0
    print("Classification accuracy: {}, {}".format(sf_clf_accuracy, sf_clf_std))

    return sf_clf_accuracy, sf_clf_std, y_pred


# GRAPHS TO EVALUATE BEST K FOLD AND BEST PERCENTILE
def feature_plot (classifier, x, y, classifier_name, x_name):
    # a list of numbers to represent the number of features increasing in hundreds
    voxels = [n for n in range(1, 10000) if n % 100 == 0]
    cv_scores = []
    scoring = {}
    cv = StratifiedKFold(n_splits=15)
    # for each 100 group of voxels run stratified 15 fold cross validation and obtain mean accuracy score
    for i in voxels:
        # f_classif - > univariate feature selection based on F-test
        X_new = SelectKBest(f_classif, k=i).fit_transform(x, y)
        cv_score = cross_val_score(classifier, X_new, y, cv=cv)
        cv_scores.append(cv_score.mean())
        scoring[X_new.shape] = cv_score.mean()
    # sort dictionary of number of features -> mean accuracy to find the best number of features by its accuracy
    scoring = sorted(scoring.items(), key=lambda x: x[1], reverse=True)
    # best k value is printed in console
    best_k_features = next(iter(scoring))

    # plot graph of cross validation score against number of features for X
    plt.plot(cv_scores, voxels)
    plt.title(
        'Performance of the ' + str(classifier_name) + ' varying the number of features/voxels selected for '+ x_name)
    plt.xlabel('Number of features x1000')
    plt.ylabel('Prediction score %')
    plt.show()

    return best_k_features


def percentile_plot(x,y):
    score_means = list()
    score_stds = list()
    percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
    # define feature selection technique - univariate feature selection based on F-test
    f_selection = feature_selection.SelectPercentile(feature_selection.f_classif)
    # pipeline, combing SVM classifier and feature selection
    anova_svc = Pipeline([('anova', f_selection), ('svc', SVC(kernel='linear'))])

    # for each percentile the SVM ANOVA pipeline parameter is reset, cross validation score for each fold is
    # computed and appended to list of scores to be plot, along with the standard deviation for error bars
    for percentile in percentiles:
        anova_svc.set_params(anova__percentile=percentile)
        # Compute cross-validation score using 1 CPU
        this_scores = cross_val_score(anova_svc, x, y, n_jobs=1)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())
    # Plot the cross-validation score as a function of percentile of features using one way ANOVA to comapare each percentile
    plt.errorbar(percentiles, score_means, np.array(score_stds), ecolor='blue', capsize=3)
    plt.title(
        'Performance of the SVM-Anova varying the percentile of features selected')
    plt.xlabel('Percentile')
    plt.ylabel('Prediction score %')
    plt.show()

    return None


def main():
    # initiate Support Vector Machine classifier
    SVM = SVC(kernel='linear')
    # initiate Logistic Regression classifier
    logreg = LR(penalty='l2', C=0.1)


    print('CLASSIFICATION OF SUBJ VPTAQ')
    # retrieve functional mri image and mask image for participant
    X = retrieve_and_mask_data("~/PycharmProjects/mydisso/data/sub3/swarfMP_BBCI_VPTAQ-0002-*.img")
    # save X object into a pickle file
    # with open('vptaq.pkl', 'wb') as fpicks:
    #     pickle.dump(X[0], fpicks)
    # produce plot of mask image and mean image
    plot_masks(X[1], X[2])
    # retrieve true rest and task class labels for participant
    Y = label_data('~/PycharmProjects/mydisso/data/labels.csv')
    # retrieve task true class labels for participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptaq.csv')
    # plot line graph of cross validation score for each percentile for SVM classifier
    # percentile_plot(X, Y)
    # plot line graph of variation in cross validation score by number of features selected for SVM classifier
    # feature_plot(SVM, X, Y, "SVM", 'VPTAQ')
    # set k value for number of features to be used by classifiers
    fs = feature_Selection(8400)

    # initiate SVM and LR classifier pipelines - > feature selection and classification performed successively
    pipeline_svm = Pipeline([('selection', fs), ('clf', SVM)])
    pipeline_lr = Pipeline([('selection', fs), ('clf', logreg)])

    # leave one out cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    loo_taq1 = leave_one_out_classification(pipeline_svm, X[0], Y, "Support Vector Machine")
    loo_taq2 = leave_one_out_classification(pipeline_svm, X[0], Y_task, "Support Vector Machine")
    loo_taq3 = leave_one_out_classification(pipeline_lr, X[0], Y, "Logistic Regression")
    loo_taq4 = leave_one_out_classification(pipeline_lr, X[0], Y_task, "Logistic Regression")
    # results from classification are dumped into pickle files for easy access to data object of results
    # with open('loo_taq_svm.pkl', 'wb') as pickle_file:
    #     pickle.dump(loo_taq1, pickle_file)
    # with open('loo_taq_lr.pkl', 'wb') as pickle_file:
    #     pickle.dump(loo_taq3, pickle_file)
    # with open('loo_taq_svm_task.pkl', 'wb') as pickle_file:
    #     pickle.dump(loo_taq2, pickle_file)
    # with open('loo_taq_lr_task.pkl', 'wb') as pickle_file:
    #     pickle.dump(loo_taq4, pickle_file)

    # stratified 15 fold cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    sk_taq1 = stratifiedkfold(pipeline_svm, X[0], Y, 'Support Vector Machine')
    sk_taq2 = stratifiedkfold(pipeline_svm, X[0], Y_task, 'Support Vector Machine')
    sk_taq3 = stratifiedkfold(pipeline_lr, X[0], Y, 'Logistic Regression')
    sk_taq4 = stratifiedkfold(pipeline_lr, X[0], Y_task, 'Logistic Regression')
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('sk_taq_svm.pkl', 'wb') as pickle_file:
        pickle.dump(sk_taq1, pickle_file)
    with open('sk_taq_lr.pkl', 'wb') as pickle_file:
        pickle.dump(sk_taq3, pickle_file)
    with open('sk_taq_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_taq2, pickle_file)
    with open('sk_taq_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_taq4, pickle_file)

    print('CLASSIFICATION OF SUBJ VPTAO')
    # retrieve functional mri image and mask image for participant
    X = retrieve_and_mask_data("~/PycharmProjects/mydisso/data/sub1/swarfMP_BBCI_VPtao-0002-*.img")
    # save X object into a pickle file
    with open('vptao.pkl', 'wb') as fpicks:
        pickle.dump(X[0], fpicks)
    # retrieve true rest and task class labels for participant
    Y = label_data('~/PycharmProjects/mydisso/data/labels.csv')
    # retrieve task true class labels for participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptao.csv')
    # plot line graph of cross validation score for each percentile for SVM classifier
    percentile_plot(X, Y)
    # plot line graph of variation in cross validation score by number of features selected for SVM classifier
    feature_plot(SVM, X, Y, "SVM", 'VPTAO')
    # set k value for number of features to be used by classifiers
    fs = feature_Selection(100)

    # initiate SVM and LR classifier pipelines - > feature selection and classification performed successively
    pipeline_svm = Pipeline([('selection', fs), ('clf', SVM)])
    pipeline_lr = Pipeline([('selection', fs), ('clf', logreg)])

    # leave one out cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    loo_tao1 = leave_one_out_classification(pipeline_svm, X[0], Y, "Support Vector Machine")
    loo_tao2 = leave_one_out_classification(pipeline_svm, X[0], Y_task, "Support Vector Machine")
    loo_tao3 = leave_one_out_classification(pipeline_lr, X[0], Y, "Logistic Regression")
    loo_tao4 = leave_one_out_classification(pipeline_lr, X[0], Y_task, "Logistic Regression")
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('loo_tao_svm.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tao1, pickle_file)
    with open('loo_tao_lr.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tao3, pickle_file)
    with open('loo_tao_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tao2, pickle_file)
    with open('loo_tao_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tao4, pickle_file)

    # stratified 15 fold cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    sk_tao1 = stratifiedkfold(pipeline_svm, X[0], Y, 'Support Vector Machine')
    sk_tao2 = stratifiedkfold(pipeline_svm, X[0], Y_task, 'Support Vector Machine')
    sk_tao3 = stratifiedkfold(pipeline_lr, X[0], Y, 'Logistic Regression')
    sk_tao4 = stratifiedkfold(pipeline_lr, X[0], Y_task, 'Logistic Regression')
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('sk_tao_svm.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tao1, pickle_file)
    with open('sk_tao_lr.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tao3, pickle_file)
    with open('sk_tao_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tao2, pickle_file)
    with open('sk_tao_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tao4, pickle_file)

    print('CLASSIFICATION OF SUBJ VPTAT')
    # retrieve functional mri image and mask image for participant
    X = retrieve_and_mask_data("~/PycharmProjects/mydisso/data/sub2/swarfMP_BBCI_VPTAT-0002-*.img")
    # save X object into a pickle file
    with open('vptat.pkl', 'wb') as fpicks:
        pickle.dump(X[0], fpicks)
    # retrieve true rest and task class labels for participant
    Y = label_data('~/PycharmProjects/mydisso/data/labels.csv')
    # retrieve task true class labels for participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptat.csv')
    # set k value for number of features to be used by classifiers
    fs = feature_Selection(100)

    # initiate SVM and LR classifier pipelines - > feature selection and classification performed successively
    pipeline_svm = Pipeline([('selection', fs), ('clf', SVM)])
    pipeline_lr = Pipeline([('selection', fs), ('clf', logreg)])

    # leave one out cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    loo_tat1 = leave_one_out_classification(pipeline_svm, X[0], Y, "Support Vector Machine")
    loo_tat2 = leave_one_out_classification(pipeline_svm, X[0], Y_task, "Support Vector Machine")
    loo_tat3 = leave_one_out_classification(pipeline_lr, X[0], Y, "Logistic Regression")
    loo_tat4 = leave_one_out_classification(pipeline_lr, X[0], Y_task, "Logistic Regression")
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('loo_tat_svm.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tat1, pickle_file)
    with open('loo_tat_lr.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tat3, pickle_file)
    with open('loo_tat_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tat2, pickle_file)
    with open('loo_tat_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tat4, pickle_file)

    # stratified 15 fold cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    sk_tat1 = stratifiedkfold(pipeline_svm, X[0], Y, 'Support Vector Machine')
    sk_tat2 = stratifiedkfold(pipeline_svm, X[0], Y_task, 'Support Vector Machine')
    sk_tat3 = stratifiedkfold(pipeline_lr, X[0], Y, 'Logistic Regression')
    sk_tat4 = stratifiedkfold(pipeline_lr, X[0], Y_task, 'Logistic Regression')
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('sk_tat_svm.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tat1, pickle_file)
    with open('sk_tat_lr.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tat3, pickle_file)
    with open('sk_tat_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tat2, pickle_file)
    with open('sk_tat_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tat4, pickle_file)

    print('CLASSIFICATION OF SUBJ VPTBE')
    # retrieve functional mri image and mask image for participant
    X = retrieve_and_mask_data("~/PycharmProjects/mydisso/data/sub4/swarfMP_BBCI_VPTBE-0002-*.img")
    # save X object into a pickle file
    with open('vptbe.pkl', 'wb') as fpicks:
        pickle.dump(X[0], fpicks)
    # retrieve true rest and task class labels for participant
    Y = label_data('~/PycharmProjects/mydisso/data/labels_vptbe.csv')
    # retrieve task true class labels for participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptbe.csv')
    # set k value for number of features to be used by classifiers
    fs = feature_Selection(1900)

    # initate SVM and LR classifier pipelines - > feature selection and classification performed successively
    pipeline_svm = Pipeline([('selection', fs), ('clf', SVM)])
    pipeline_lr = Pipeline([('selection', fs), ('clf', logreg)])

    # leave one out cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    loo_tbe1 = leave_one_out_classification(pipeline_svm, X[0], Y, "Support Vector Machine")
    loo_tbe2 = leave_one_out_classification(pipeline_svm, X[0], Y_task, "Support Vector Machine")
    loo_tbe3 = leave_one_out_classification(pipeline_lr, X[0], Y, "Logistic Regression")
    loo_tbe4 = leave_one_out_classification(pipeline_lr, X[0], Y_task, "Logistic Regression")
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('loo_tbe_svm.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tbe1, pickle_file)
    with open('loo_tbe_lr.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tbe3, pickle_file)
    with open('loo_tbe_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tbe2, pickle_file)
    with open('loo_tbe_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tbe4, pickle_file)

    # stratified 15 fold cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    sk_tbe1 = stratifiedkfold(pipeline_svm, X[0], Y, 'Support Vector Machine')
    sk_tbe2 = stratifiedkfold(pipeline_svm, X[0], Y_task, 'Support Vector Machine')
    sk_tbe3 = stratifiedkfold(pipeline_lr, X[0], Y, 'Logistic Regression')
    sk_tbe4 = stratifiedkfold(pipeline_lr, X[0], Y_task, 'Logistic Regression')
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('sk_tbe_svm.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tbe1, pickle_file)
    with open('sk_tbe_lr.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tbe3, pickle_file)
    with open('sk_tbe_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tbe2, pickle_file)
    with open('sk_tbe_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tbe4, pickle_file)

    print('CLASSIFICATION OF SUBJ VPTAK')
    # retrieve functional mri image and mask image for participant
    X = retrieve_and_mask_data("~/PycharmProjects/mydisso/data/sub5/swarfMP_BBCI_AK-0002-*.img")
    # save X object into a pickle file
    with open('vptak.pkl', 'wb') as fpicks:
        pickle.dump(X[0], fpicks)
    # retrieve true rest and task class labels for participant
    Y = label_data('~/PycharmProjects/mydisso/data/labels.csv')
    # retrieve task true class labels for participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptak.csv')
    # set k value for number of features to be used by classifiers
    fs = feature_Selection(9300)

    # initiate SVM and LR classifier pipelines - > feature selection and classification performed successively
    pipeline_svm = Pipeline([('selection', fs), ('clf', SVM)])
    pipeline_lr = Pipeline([('selection', fs), ('clf', logreg)])

    # leave one out cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    loo_tak1 = leave_one_out_classification(pipeline_svm, X[0], Y, "Support Vector Machine")
    loo_tak2 = leave_one_out_classification(pipeline_svm, X[0], Y_task, "Support Vector Machine")
    loo_tak3 = leave_one_out_classification(pipeline_lr, X[0], Y, "Logistic Regression")
    loo_tak4 = leave_one_out_classification(pipeline_lr, X[0], Y_task, "Logistic Regression")
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('loo_tak_svm.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tak1, pickle_file)
    with open('loo_tak_lr.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tak3, pickle_file)
    with open('loo_tak_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tak2, pickle_file)
    with open('loo_tak_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tak4, pickle_file)

    # stratified 15 fold cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    sk_tak1 = stratifiedkfold(pipeline_svm, X[0], Y, 'Support Vector Machine')
    sk_tak2 = stratifiedkfold(pipeline_svm, X[0], Y_task, 'Support Vector Machine')
    sk_tak3 = stratifiedkfold(pipeline_lr, X[0], Y, 'Logistic Regression')
    sk_tak4 = stratifiedkfold(pipeline_lr, X[0], Y_task, 'Logistic Regression')
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('sk_tak_svm.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tak1, pickle_file)
    with open('sk_tak_lr.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tak3, pickle_file)
    with open('sk_tak_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tak2, pickle_file)
    with open('sk_tak_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tak4, pickle_file)

    print('CLASSIFICATION OF SUBJ VPTAJ')
    # retrieve functional mri image and mask image for participant
    X = retrieve_and_mask_data("~/PycharmProjects/mydisso/data/sub6/swarfMP_BBCI_AJ-0002-*.img")
    # save X object into a pickle file
    with open('vptaj.pkl', 'wb') as fpicks:
        pickle.dump(X[0], fpicks)
    # retrieve true rest and task class labels for participant
    Y = label_data('~/PycharmProjects/mydisso/data/labels.csv')
    # retrieve task true class labels for participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptaj.csv')
    # set k value for number of features to be used by classifiers
    fs = feature_Selection(9800)

    # initiate SVM and LR classifier pipelines - > feature selection and classification performed successively
    pipeline_svm = Pipeline([('selection', fs), ('clf', SVM)])
    pipeline_lr = Pipeline([('selection', fs), ('clf', logreg)])

    # leave one out cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    loo_taj1 = leave_one_out_classification(pipeline_svm, X[0], Y, "Support Vector Machine")
    loo_taj2 = leave_one_out_classification(pipeline_svm, X[0], Y_task, "Support Vector Machine")
    loo_taj3 = leave_one_out_classification(pipeline_lr, X[0], Y, "Logistic Regression")
    loo_taj4 = leave_one_out_classification(pipeline_lr, X[0], Y_task, "Logistic Regression")
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('loo_taj_svm.pkl', 'wb') as pickle_file:
        pickle.dump(loo_taj1, pickle_file)
    with open('loo_taj_lr.pkl', 'wb') as pickle_file:
        pickle.dump(loo_taj3, pickle_file)
    with open('loo_taj_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_taj2, pickle_file)
    with open('loo_taj_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_taj4, pickle_file)

    # stratified 15 fold cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    sk_taj1 = stratifiedkfold(pipeline_svm, X[0], Y, 'Support Vector Machine')
    sk_taj2 = stratifiedkfold(pipeline_svm, X[0], Y_task, 'Support Vector Machine')
    sk_taj3 = stratifiedkfold(pipeline_lr, X[0], Y, 'Logistic Regression')
    sk_taj4 = stratifiedkfold(pipeline_lr, X[0], Y_task, 'Logistic Regression')
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('sk_taj_svm.pkl', 'wb') as pickle_file:
        pickle.dump(sk_taj1, pickle_file)
    with open('sk_taj_lr.pkl', 'wb') as pickle_file:
        pickle.dump(sk_taj3, pickle_file)
    with open('sk_taj_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_taj2, pickle_file)
    with open('sk_taj_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_taj4, pickle_file)

    print('CLASSIFICATION OF VPTAE')
    # retrieve functional mri image and mask image for participant
    X = retrieve_and_mask_data("~/PycharmProjects/mydisso/data/sub7/swarfMP_BBCI_TAE-0002-*.img")
    # save X object into a pickle file
    with open('vptae.pkl', 'wb') as fpicks:
        pickle.dump(X[0], fpicks)
    # retrieve true rest and task class labels for participant
    Y = label_data('~/PycharmProjects/mydisso/data/labels.csv')
    # retrieve task true class labels for participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptae.csv')
    # set k value for number of features to be used by classifiers
    fs = feature_Selection(100)

    # initiate SVM and LR classifier pipelines - > feature selection and classification performed successively
    pipeline_svm = Pipeline([('selection', fs), ('clf', SVM)])
    pipeline_lr = Pipeline([('selection', fs), ('clf', logreg)])

    # leave one out cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    loo_tae1 = leave_one_out_classification(pipeline_svm, X[0], Y, "Support Vector Machine")
    loo_tae2 = leave_one_out_classification(pipeline_svm, X[0], Y_task, "Support Vector Machine")
    loo_tae3 = leave_one_out_classification(pipeline_lr, X[0], Y, "Logistic Regression")
    loo_tae4 = leave_one_out_classification(pipeline_lr, X[0], Y_task, "Logistic Regression")
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('loo_tae_svm.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tae1, pickle_file)
    with open('loo_tae_lr.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tae3, pickle_file)
    with open('loo_tae_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tae2, pickle_file)
    with open('loo_tae_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tae4, pickle_file)

    # stratified 15 fold cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    sk_tae1 = stratifiedkfold(pipeline_svm, X[0], Y, 'Support Vector Machine')
    sk_tae2 = stratifiedkfold(pipeline_svm, X[0], Y_task, 'Support Vector Machine')
    sk_tae3 = stratifiedkfold(pipeline_lr, X[0], Y, 'Logistic Regression')
    sk_tae4 = stratifiedkfold(pipeline_lr, X[0], Y_task, 'Logistic Regression')
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('sk_tae_svm.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tae1, pickle_file)
    with open('sk_tae_lr.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tae3, pickle_file)
    with open('sk_tae_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tae2, pickle_file)
    with open('sk_tae_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tae4, pickle_file)

    print('CLASSIFICATION OF SUBJ VPTAD')
    # retrieve functional mri image and mask image for participant
    X = retrieve_and_mask_data("~/PycharmProjects/mydisso/data/sub8/swarfMP_BBCI_TAD-0002-*.img")
    # save X object into a pickle file
    with open('vptad.pkl', 'wb') as fpicks:
        pickle.dump(X[0], fpicks)
    # retrieve true rest and task class labels for participant
    Y = label_data('~/PycharmProjects/mydisso/data/labels.csv')
    # retrieve task true class labels for participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptad.csv')
    # plot line graph of variation in cross validation score by number of features selected for SVM classifier
    feature_plot(SVM, X, Y, "SVM", 'VPTAD')
    # set k value for number of features to be used by classifiers
    fs = feature_Selection(7000)

    # initiate SVM and LR classifier pipelines - > feature selection and classification performed successively
    pipeline_svm = Pipeline([('selection', fs), ('clf', SVM)])
    pipeline_lr = Pipeline([('selection', fs), ('clf', logreg)])

    # leave one out cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    loo_tad1 = leave_one_out_classification(pipeline_svm, X[0], Y, "Support Vector Machine")
    loo_tad2 = leave_one_out_classification(pipeline_svm, X[0], Y_task, "Support Vector Machine")
    loo_tad3 = leave_one_out_classification(pipeline_lr, X[0], Y, "Logistic Regression")
    loo_tad4 = leave_one_out_classification(pipeline_lr, X[0], Y_task, "Logistic Regression")
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('loo_tad_svm.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tad1, pickle_file)
    with open('loo_tad_lr.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tad3, pickle_file)
    with open('loo_tad_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tad2, pickle_file)
    with open('loo_tad_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(loo_tad4, pickle_file)

    # stratified 15 fold cross validation is performed by both classifiers and assigned to variables
    # for both classification problems - > rest vs task, task 1 vs task 2
    sk_tad1 = stratifiedkfold(pipeline_svm, X[0], Y, 'Support Vector Machine')
    sk_tad2 = stratifiedkfold(pipeline_svm, X[0], Y_task, 'Support Vector Machine')
    sk_tad3 = stratifiedkfold(pipeline_lr, X[0], Y, 'Logistic Regression')
    sk_tad4 = stratifiedkfold(pipeline_lr, X[0], Y_task, 'Logistic Regression')
    # results from classification are dumped into pickle files for easy access to data object of results
    with open('sk_tad_svm.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tad1, pickle_file)
    with open('sk_tad_lr.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tad3, pickle_file)
    with open('sk_tad_svm_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tad2, pickle_file)
    with open('sk_tad_lr_task.pkl', 'wb') as pickle_file:
        pickle.dump(sk_tad4, pickle_file)

##########################################################################################################

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)

# ##########################################################################################

