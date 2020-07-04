import glob
import itertools
import matplotlib
import nilearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import mean_img
import pickle
from nilearn.input_data import NiftiMasker
from sklearn import metrics, model_selection
from nilearn.masking import *
from sklearn.metrics import confusion_matrix

matplotlib.use('MacOSX')

 
def label_data(label_file):
    behavioural = pd.read_csv(label_file, sep=",")
    conditions = behavioural['labels']
    Y = conditions[1:]

    return Y


def return_confusion_matrix(y_prediction, y, x_name):
    cnf_matrix = confusion_matrix(y, y_prediction)

    return cnf_matrix, metrics.accuracy_score(y, y_prediction)


def plot_confusion_matrix(y_prediction, y, x_name):
    cnf_matrix = confusion_matrix(y, y_prediction)
    classess = set(y)
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classess))
    plt.xticks(tick_marks, classess)
    plt.yticks(tick_marks, classess)

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j], horizontalalignment="center", verticalalignment="center", color="black")

    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.suptitle('Confusion Matrix for subject: ' + x_name)
    plt.show()

    return cnf_matrix, metrics.accuracy_score(y, y_prediction)


def main():
    print('CLASSIFICATION FOR SUBJ VPTAQ')
    # retrieve true rest and task class labels for participant and assign to Y variable, same for all participant
    Y = label_data('~/PycharmProjects/mydisso/data/labels.csv')
    # retrieve task true class labels for current participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptaq.csv')
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the stratified 15 fold cross validation method
    with open('sk_taq_svm.pkl', 'rb') as pickle_file:
        sk_taq1 = pickle.load(pickle_file)
    with open('sk_taq_lr.pkl', 'rb') as pickle_file:
        sk_taq3 = pickle.load(pickle_file)
    with open('sk_taq_svm_task.pkl', 'rb') as pickle_file:
        sk_taq2 = pickle.load(pickle_file)
    with open('sk_taq_lr_task.pkl', 'rb') as pickle_file:
        sk_taq4 = pickle.load(pickle_file)

    # plots of confusion matrix for each classification object result in pickle file with stratified cross validation
    cm_taqa = return_confusion_matrix(sk_taq1[2], Y, "VPTAQ")
    cm_taq1 = return_confusion_matrix(sk_taq2[2], Y_task, "VPTAQ")
    cm_taq = return_confusion_matrix(sk_taq3[2], Y, "VPTAQ")
    cm_taq2 = return_confusion_matrix(sk_taq4[2], Y_task, "VPTAQ")
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the leaveoneout cross validation method
    with open('loo_taq_svm.pkl', 'rb') as pickle_file:
        loo_taq1 = pickle.load(pickle_file)
    with open('loo_taq_lr.pkl', 'rb') as pickle_file:
        loo_taq3 = pickle.load(pickle_file)
    with open('loo_taq_svm_task.pkl', 'rb') as pickle_file:
        loo_taq2 = pickle.load(pickle_file)
    with open('loo_taq_lr_task.pkl', 'rb') as pickle_file:
        loo_taq4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with leaveoneout cross validation
    l_cm_taqa = return_confusion_matrix(loo_taq1[2], Y, "VPTAQ")
    l_cm_taq1 = return_confusion_matrix(loo_taq2[2], Y_task, "VPTAQ")
    l_cm_taq = return_confusion_matrix(loo_taq3[2], Y, "VPTAQ")
    l_cm_taq2 = return_confusion_matrix(loo_taq4[2], Y_task, "VPTAQ")

    print('CLASSIFICATION FOR SUBJ VPTAO')
    # retrieve task true class labels for current participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptao.csv')
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the stratified 15 fold cross validation method
    with open('sk_tao_svm.pkl', 'rb') as pickle_file:
        sk_tao1 = pickle.load(pickle_file)
    with open('sk_tao_lr.pkl', 'rb') as pickle_file:
        sk_tao3 = pickle.load(pickle_file)
    with open('sk_tao_svm_task.pkl', 'rb') as pickle_file:
        sk_tao2 = pickle.load(pickle_file)
    with open('sk_tao_lr_task.pkl', 'rb') as pickle_file:
        sk_tao4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with stratified cross validation
    cm_taoa = return_confusion_matrix(sk_tao1[2], Y, "VPTAO")
    cm_tao1 = return_confusion_matrix(sk_tao2[2], Y_task, "VPTAO")
    cm_tao = return_confusion_matrix(sk_tao3[2], Y, "VPTAO")
    cm_tao2 = plot_confusion_matrix(sk_tao4[2], Y_task, "VPTAO")
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the leaveoneout cross validation method
    with open('loo_tao_svm.pkl', 'rb') as pickle_file:
        loo_tao1 = pickle.load(pickle_file)
    with open('loo_tao_lr.pkl', 'rb') as pickle_file:
        loo_tao3 = pickle.load(pickle_file)
    with open('loo_tao_svm_task.pkl', 'rb') as pickle_file:
        loo_tao2 = pickle.load(pickle_file)
    with open('loo_tao_lr_task.pkl', 'rb') as pickle_file:
        loo_tao4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with leaveoneout cross validation
    l_cm_taoa = return_confusion_matrix(loo_tao1[2], Y, "VPTAO")
    l_cm_tao1 = return_confusion_matrix(loo_tao2[2], Y_task, "VPTAO")
    l_cm_tao = return_confusion_matrix(loo_tao3[2], Y, "VPTAO")
    l_cm_tao2 = return_confusion_matrix(loo_tao4[2], Y_task, "VPTAO")

    print('CLASSIFICATION FOR SUBJ VPTAT')
    # retrieve task true class labels for current participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptat.csv')
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the stratified 15 fold cross validation method
    with open('sk_tat_svm.pkl', 'rb') as pickle_file:
        sk_tat1 = pickle.load(pickle_file)
    with open('sk_tat_lr.pkl', 'rb') as pickle_file:
        sk_tat3 = pickle.load(pickle_file)
    with open('sk_tat_svm_task.pkl', 'rb') as pickle_file:
        sk_tat2 = pickle.load(pickle_file)
    with open('sk_tat_lr_task.pkl', 'rb') as pickle_file:
        sk_tat4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with stratified cross validation
    cm_tata = return_confusion_matrix(sk_tat1[2], Y, "VPTAT")
    cm_tat1 = return_confusion_matrix(sk_tat2[2], Y_task, "VPTAT")
    cm_tat = plot_confusion_matrix(sk_tat3[2], Y, "VPTAT")
    cm_tat2 = return_confusion_matrix(sk_tat4[2], Y_task, "VPTAT")
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the leaveoneout cross validation method
    with open('loo_tat_svm.pkl', 'rb') as pickle_file:
        loo_tat1 = pickle.load(pickle_file)
    with open('loo_tat_lr.pkl', 'rb') as pickle_file:
        loo_tat3 = pickle.load(pickle_file)
    with open('loo_tat_svm_task.pkl', 'rb') as pickle_file:
        loo_tat2 = pickle.load(pickle_file)
    with open('loo_tat_lr_task.pkl', 'rb') as pickle_file:
        loo_tat4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with leaveoneout cross validation
    l_cm_tata = return_confusion_matrix(loo_tat1[2], Y, "VPTAT")
    l_cm_tat1 = return_confusion_matrix(loo_tat2[2], Y_task, "VPTAT")
    l_cm_tat = return_confusion_matrix(loo_tat3[2], Y, "VPTAT")
    l_cm_tat2 = return_confusion_matrix(loo_tat4[2], Y_task, "VPTAT")

    print('CLASSIFICATION FOR SUBJ VPTBE')
    # retrieve true rest and task class labels for vptbe participant and assign to new Y variable
    Y_tb = label_data('~/PycharmProjects/mydisso/data/labels_vptbe.csv')
    # retrieve task true class labels for current participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptbe.csv')
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the stratified 15 fold cross validation method
    with open('sk_tbe_svm.pkl', 'rb') as pickle_file:
        sk_tbe1 = pickle.load(pickle_file)
    with open('sk_tbe_lr.pkl', 'rb') as pickle_file:
        sk_tbe3 = pickle.load(pickle_file)
    with open('sk_tbe_svm_task.pkl', 'rb') as pickle_file:
        sk_tbe2 = pickle.load(pickle_file)
    with open('sk_tbe_lr_task.pkl', 'rb') as pickle_file:
        sk_tbe4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with stratified cross validation
    cm_tbea = return_confusion_matrix(sk_tbe1[2], Y_tb, "VPTBE")
    cm_tbe1 = return_confusion_matrix(sk_tbe2[2], Y_task, "VPTBE")
    cm_tbe = plot_confusion_matrix(sk_tbe3[2], Y_tb, "VPTBE")
    cm_tbe2 = return_confusion_matrix(sk_tbe4[2], Y_task, "VPTBE")
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the leaveoneout cross validation method
    with open('loo_tbe_svm.pkl', 'rb') as pickle_file:
        loo_tbe1 = pickle.load(pickle_file)
    with open('loo_tbe_lr.pkl', 'rb') as pickle_file:
        loo_tbe3 = pickle.load(pickle_file)
    with open('loo_tbe_svm_task.pkl', 'rb') as pickle_file:
        loo_tbe2 = pickle.load(pickle_file)
    with open('loo_tbe_lr_task.pkl', 'rb') as pickle_file:
        loo_tbe4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with leaveoneout cross validation
    l_cm_tbea = return_confusion_matrix(loo_tbe1[2], Y_tb, "VPTBE")
    l_cm_tbe1 = return_confusion_matrix(loo_tbe2[2], Y_task, "VPTBE")
    l_cm_tbe = return_confusion_matrix(loo_tbe3[2], Y_tb, "VPTBE")
    l_cm_tbe2 = return_confusion_matrix(loo_tbe4[2], Y_task, "VPTBE")

    print('CLASSIFICATION FOR SUBJ VPTAK')
    # retrieve task true class labels for current participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptak.csv')
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the stratified 15 fold cross validation method
    with open('sk_tak_svm.pkl', 'rb') as pickle_file:
        sk_tak1 = pickle.load(pickle_file)
    with open('sk_tak_lr.pkl', 'rb') as pickle_file:
        sk_tak3 = pickle.load(pickle_file)
    with open('sk_tak_svm_task.pkl', 'rb') as pickle_file:
        sk_tak2 = pickle.load(pickle_file)
    with open('sk_tak_lr_task.pkl', 'rb') as pickle_file:
        sk_tak4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with stratified cross validation
    cm_taka = return_confusion_matrix(sk_tak1[2], Y, "VPTAK")
    cm_tak1 = return_confusion_matrix(sk_tak2[2], Y_task, "VPTAK")
    cm_tak = return_confusion_matrix(sk_tak3[2], Y, "VPTAK")
    cm_tak2 = return_confusion_matrix(sk_tak4[2], Y_task, "VPTAK")
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the leaveoneout cross validation method
    with open('loo_tak_svm.pkl', 'rb') as pickle_file:
        loo_tak1 = pickle.load(pickle_file)
    with open('loo_tak_lr.pkl', 'rb') as pickle_file:
        loo_tak3 = pickle.load(pickle_file)
    with open('loo_tak_svm_task.pkl', 'rb') as pickle_file:
        loo_tak2 = pickle.load(pickle_file)
    with open('loo_tak_lr_task.pkl', 'rb') as pickle_file:
        loo_tak4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with leaveoneout cross validation
    l_cm_taka = return_confusion_matrix(loo_tak1[2], Y, "VPTAK")
    l_cm_tak1 = return_confusion_matrix(loo_tak2[2], Y_task, "VPTAK")
    l_cm_tak = return_confusion_matrix(loo_tak3[2], Y, "VPTAK")
    l_cm_tak2 = return_confusion_matrix(loo_tak4[2], Y_task, "VPTAK")

    print('CLASSIFICATION FOR SUBJ VPTAJ')
    # retrieve task true class labels for current participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptaj.csv')
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the stratified 15 fold cross validation method
    with open('sk_taj_svm.pkl', 'rb') as pickle_file:
        sk_taj1 = pickle.load(pickle_file)
    with open('sk_taj_lr.pkl', 'rb') as pickle_file:
        sk_taj3 = pickle.load(pickle_file)
    with open('sk_taj_svm_task.pkl', 'rb') as pickle_file:
        sk_taj2 = pickle.load(pickle_file)
    with open('sk_taj_lr_task.pkl', 'rb') as pickle_file:
        sk_taj4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with stratified cross validation
    cm_taja = return_confusion_matrix(sk_taj1[2], Y, "VPTAJ")
    cm_taj1 = return_confusion_matrix(sk_taj2[2], Y_task, "VPTAJ")
    cm_taj = return_confusion_matrix(sk_taj3[2], Y, "VPTAJ")
    cm_taj2 = return_confusion_matrix(sk_taj4[2], Y_task, "VPTAJ")
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the leaveoneout cross validation method
    with open('loo_taj_svm.pkl', 'rb') as pickle_file:
        loo_taj1 = pickle.load(pickle_file)
    with open('loo_taj_lr.pkl', 'rb') as pickle_file:
        loo_taj3 = pickle.load(pickle_file)
    with open('loo_taj_svm_task.pkl', 'rb') as pickle_file:
        loo_taj2 = pickle.load(pickle_file)
    with open('loo_taj_lr_task.pkl', 'rb') as pickle_file:
        loo_taj4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with leaveoneout cross validation
    l_cm_taja = return_confusion_matrix(loo_taj1[2], Y, "VPTAJ")
    l_cm_taj1 = return_confusion_matrix(loo_taj2[2], Y_task, "VPTAJ")
    l_cm_taj = return_confusion_matrix(loo_taj3[2], Y, "VPTAJ")
    l_cm_taj2 = return_confusion_matrix(loo_taj4[2], Y_task, "VPTAJ")

    print('CLASSIFICATION FOR SUBJ VPTAE')
    # retrieve task true class labels for current participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptae.csv')
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the stratified 15 fold cross validation method
    with open('sk_tae_svm.pkl', 'rb') as pickle_file:
        sk_tae1 = pickle.load(pickle_file)
    with open('sk_tae_lr.pkl', 'rb') as pickle_file:
        sk_tae3 = pickle.load(pickle_file)
    with open('sk_tae_svm_task.pkl', 'rb') as pickle_file:
        sk_tae2 = pickle.load(pickle_file)
    with open('sk_tae_lr_task.pkl', 'rb') as pickle_file:
        sk_tae4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with stratified cross validation
    cm_taea = return_confusion_matrix(sk_tae1[2], Y, "VPTAE")
    cm_tae1 = return_confusion_matrix(sk_tae2[2], Y_task, "VPTAE")
    cm_tae = return_confusion_matrix(sk_tae3[2], Y, "VPTAE")
    cm_tae2 = return_confusion_matrix(sk_tae4[2], Y_task, "VPTAE")
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the leaveoneout cross validation method
    with open('loo_tae_svm.pkl', 'rb') as pickle_file:
        loo_tae1 = pickle.load(pickle_file)
    with open('loo_tae_lr.pkl', 'rb') as pickle_file:
        loo_tae3 = pickle.load(pickle_file)
    with open('loo_tae_svm_task.pkl', 'rb') as pickle_file:
        loo_tae2 = pickle.load(pickle_file)
    with open('loo_tae_lr_task.pkl', 'rb') as pickle_file:
        loo_tae4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with leaveoneout cross validation
    l_cm_taea = return_confusion_matrix(loo_tae1[2], Y, "VPTAE")
    l_cm_tae1 = return_confusion_matrix(loo_tae2[2], Y_task, "VPTAE")
    l_cm_tae = return_confusion_matrix(loo_tae3[2], Y, "VPTAE")
    l_cm_tae2 = return_confusion_matrix(loo_tae4[2], Y_task, "VPTAE")

    print('CLASSIFICATION FOR SUBJ VPTAD')
    # retrieve task true class labels for current participant
    Y_task = label_data('~/PycharmProjects/mydisso/data/l_vptad.csv')
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the stratified 15 fold cross validation method
    with open('sk_tad_svm.pkl', 'rb') as pickle_file:
        sk_tad1 = pickle.load(pickle_file)
    with open('sk_tad_lr.pkl', 'rb') as pickle_file:
        sk_tad3 = pickle.load(pickle_file)
    with open('sk_tad_svm_task.pkl', 'rb') as pickle_file:
        sk_tad2 = pickle.load(pickle_file)
    with open('sk_tad_lr_task.pkl', 'rb') as pickle_file:
        sk_tad4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with stratified cross validation
    cm_tada = return_confusion_matrix(sk_tad1[2], Y, "VPTAD")
    cm_tad1 = return_confusion_matrix(sk_tad2[2], Y_task, "VPTAD")
    cm_tad = return_confusion_matrix(sk_tad3[2], Y, "VPTAD")
    cm_tad2 = plot_confusion_matrix(sk_tad4[2], Y_task, "VPTAD")
    # open each pickle file containing classification result object for participant using LR ans SVM for
    # using the leaveoneout cross validation method
    with open('loo_tad_svm.pkl', 'rb') as pickle_file:
        loo_tad1 = pickle.load(pickle_file)
    with open('loo_tad_lr.pkl', 'rb') as pickle_file:
        loo_tad3 = pickle.load(pickle_file)
    with open('loo_tad_svm_task.pkl', 'rb') as pickle_file:
        loo_tad2 = pickle.load(pickle_file)
    with open('loo_tad_lr_task.pkl', 'rb') as pickle_file:
        loo_tad4 = pickle.load(pickle_file)
    # plots of confusion matrix for each classification object result in pickle file with leaveoneout cross validation
    l_cm_tada = return_confusion_matrix(loo_tad1[2], Y, "VPTAD")
    l_cm_tad1 = return_confusion_matrix(loo_tad2[2], Y_task, "VPTAD")
    l_cm_tad = return_confusion_matrix(loo_tad3[2], Y, "VPTAD")
    l_cm_tad2 = return_confusion_matrix(loo_tad4[2], Y_task, "VPTAD")

    # STRATIFIED FOLD BAR PLOT overall accuracy rest vs task
    data = [[sk_tad1[0], sk_tae1[0], sk_taj1[0], sk_tak1[0], sk_tao1[0], sk_taq1[0], sk_tat1[0], sk_tbe1[0]],
            [sk_tad3[0], sk_tae3[0], sk_taj3[0], sk_tak3[0], sk_tao3[0], sk_taq3[0], sk_tat3[0], sk_tbe3[0]]]
    arr = np.arange(8)
    err = [[sk_tad1[1], sk_tae1[1], sk_taj1[1], sk_tak1[1], sk_tao1[1], sk_taq1[1], sk_tat1[1], sk_tbe1[1]],
           [sk_tad3[1], sk_tae3[1], sk_taj3[1], sk_tak3[1], sk_tao3[1], sk_taq3[1], sk_tat3[1], sk_tbe3[1]]]
    plt.xlabel('Subjects')
    plt.ylabel('Prediction score (%)')
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25, yerr=err[0], capsize=3)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25, yerr=err[1], capsize=3)
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'], loc=4)
    plt.suptitle('Prediction scores for each subject comparing SVM and LR classifiers')
    plt.show()

    # STRATIFIED FOLD BAR PLOT overall accuracy task 1 vs task 2
    data = [[sk_tad2[0], sk_tae2[0], sk_taj2[0], sk_tak2[0], sk_tao2[0], sk_taq2[0], sk_tat2[0], sk_tbe2[0]],
            [sk_tad4[0], sk_tae4[0], sk_taj4[0], sk_tak4[0], sk_tao4[0], sk_taq4[0], sk_tat4[0], sk_tbe4[0]]]
    arr = np.arange(8)
    err = [[sk_tad2[1], sk_tae2[1], sk_taj2[1], sk_tak2[1], sk_tao2[1], sk_taq2[1], sk_tat2[1], sk_tbe2[1]],
           [sk_tad4[1], sk_tae4[1], sk_taj4[1], sk_tak4[1], sk_tao4[1], sk_taq4[1], sk_tat4[1], sk_tbe4[1]]]

    plt.xlabel('Subjects')
    plt.ylabel('Prediction score (%)')
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25, yerr=err[0], capsize=3)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25, yerr=err[1], capsize=3)
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'], loc=4)
    plt.suptitle('Prediction scores for each subject comparing SVM and LR classifiers')
    plt.show()

    # STRATIFIED FOLD BAR PLOT overall task 1 vs task 2 vs task 3 accuracy PLOTS
    data = [[((cm_tad1[1] + cm_tak1[1] + cm_tao1[1] + cm_tat1[1]) / 4.0) * 100,
             ((cm_tae1[1] + cm_taj1[1] + cm_taq1[1]) / 3.0) * 100, cm_tbe1[1] * 100], [((cm_tad2[1] + cm_tak2[1] +
            cm_tao2[1] + cm_tat2[1]) / 4.0) * 100, ((cm_tae2[1] +cm_taj2[1] +cm_taq2[1]) / 3.0) * 100,cm_tbe2[1] * 100]]
    arr = np.arange(3)
    plt.xlabel('Motor Task Classification Problem in order 1 vs 3, 1 vs 2, 2 vs 3')
    plt.ylabel('Prediction score (%)')
    plt.xticks([])
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'], loc=4)
    plt.suptitle(
        'Plot of variation in classification accuracy against task performed: left hand(1), right hand(2) or feet(3)')
    plt.show()

    # STRATIFIED FOLD BAR PLOT of prediction accuracy rest vs task and task 1 vs task 2 for each subject
    data = [[sk_tad1[0], sk_tae1[0], sk_taj1[0], sk_tak1[0], sk_tao1[0], sk_taq1[0], sk_tat1[0], sk_tbe1[0]],
            [sk_tad2[0], sk_tae2[0], sk_taj2[0], sk_tak2[0], sk_tao2[0], sk_taq2[0], sk_tat2[0], sk_tbe2[0]]]
    arr = np.arange(8)
    plt.xlabel('Subjects')
    plt.ylabel('Prediction score (%)')
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)
    plt.legend(labels=['rest vs task', 'task 1 vs task 2'], loc=4)
    plt.suptitle(
        'Plot of variation in SVM classification accuracy against classification problems one and two')
    plt.show()

    data = [[sk_tad3[0], sk_tae3[0], sk_taj3[0], sk_tak3[0], sk_tao3[0], sk_taq3[0], sk_tat3[0], sk_tbe3[0]],
            [sk_tad4[0], sk_tae4[0], sk_taj4[0], sk_tak4[0], sk_tao4[0], sk_taq4[0], sk_tat4[0], sk_tbe4[0]]]
    arr = np.arange(8)
    plt.xlabel('Subjects')
    plt.ylabel('Prediction score (%)')
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)
    plt.legend(labels=['rest vs task', 'task 1 vs task 2'], loc=4)
    plt.suptitle(
        'Plot of variation in LR classification accuracy against classification problems one and two')
    plt.show()

    ## STRATIFIED FOLD BAR PLOT of recognition rate of classifiers

    # overall rest vs task recognition rate
    data = [[(((cm_tada[0][0][0] / (cm_tada[0][0][0] + cm_tada[0][1][0])) + (
            cm_taka[0][0][0] / (cm_taka[0][0][0] + cm_taka[0][1][0])) +
               (cm_taoa[0][0][0] / (cm_taoa[0][0][0] + cm_taoa[0][1][0])) + (
                       cm_tata[0][0][0] / (cm_tata[0][0][0] + cm_tata[0][1][0])) +
               (cm_taea[0][0][0] / (cm_taea[0][0][0] + cm_taea[0][1][0])) + (
                       cm_taja[0][0][0] / (cm_taja[0][0][0] + cm_taja[0][1][0])) +
               (cm_taqa[0][0][0] / (cm_taqa[0][0][0] + cm_taqa[0][1][0])) + (
                       cm_tbea[0][0][0] / (cm_tbea[0][0][0] + cm_tbea[0][1][0]))) * 100 / 8.0),
             (((cm_tad[0][0][0] / (cm_tad[0][0][0] + cm_tad[0][1][0])) + (
                         cm_tak[0][0][0] / (cm_tak[0][0][0] + cm_tak[0][1][0])) +
               (cm_tao[0][0][0] / (cm_tao[0][0][0] + cm_tao[0][1][0])) + (
                           cm_tat[0][0][0] / (cm_tat[0][0][0] + cm_tat[0][1][0])) +
               (cm_tae[0][0][0] / (cm_tae[0][0][0] + cm_tae[0][1][0])) + (
                           cm_taj[0][0][0] / (cm_taj[0][0][0] + cm_taj[0][1][0])) +
               (cm_taq[0][0][0] / (cm_taq[0][0][0] + cm_taq[0][1][0])) + (
                           cm_tbe[0][0][0] / (cm_tbe[0][0][0] + cm_tbe[0][1][0]))) * 100 / 8.0)],
            [(((cm_tada[0][1][1] / (cm_tada[0][1][0] + cm_tada[0][1][1])) + (
                    cm_taka[0][1][1] / (cm_taka[0][1][0] + cm_taka[0][1][1])) +
               (cm_taoa[0][1][1] / (cm_taoa[0][1][0] + cm_taoa[0][1][1])) + (
                       cm_tata[0][1][1] / (cm_tata[0][1][0] + cm_tata[0][1][1])) +
               (cm_taea[0][1][1] / (cm_taea[0][1][0] + cm_taea[0][1][1])) + (
                       cm_taja[0][1][1] / (cm_taja[0][1][0] + cm_taja[0][1][1])) +
               (cm_taqa[0][1][1] / (cm_taqa[0][1][0] + cm_taqa[0][1][1])) + (
                       cm_tbea[0][1][1] / (cm_tbea[0][1][0] + cm_tbea[0][1][1]))) * 100 / 8.0),
             (((cm_tad[0][1][1] / (cm_tad[0][1][0] + cm_tad[0][1][1])) + (
                     cm_tak[0][1][1] / (cm_tak[0][1][0] + cm_tak[0][1][1])) +
               (cm_tao[0][1][1] / (cm_tao[0][1][0] + cm_tao[0][1][1])) + (
                       cm_tat[0][1][1] / (cm_tat[0][1][0] + cm_tat[0][1][1])) +
               (cm_tae[0][1][1] / (cm_tae[0][1][0] + cm_tae[0][1][1])) + (
                       cm_taj[0][1][1] / (cm_taj[0][1][0] + cm_taj[0][1][1])) +
               (cm_taq[0][1][1] / (cm_taq[0][1][0] + cm_taq[0][1][1])) + (
                       cm_tbe[0][1][1] / (cm_tbe[0][1][0] + cm_tbe[0][1][1]))) * 100 / 8.0)
             ]]
    arr = np.arange(2)
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)
    plt.xticks([])
    plt.xlabel('Motor task: rest, task')
    plt.ylabel('Recognition rate of classifier %')
    plt.suptitle('Plot of recognition rate of classifier for class problem: rest or task')
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'], loc=4)
    plt.show()

    data = [[(((cm_taq1[0][2][2] / (cm_taq1[0][2][1] + cm_taq1[0][2][2] + cm_taq1[0][2][0])) * 100) + (
            (cm_tao1[0][2][2] / (cm_tao1[0][2][1] + cm_tao1[0][2][2] + cm_tao1[0][2][0])) * 100) +
              ((cm_tat1[0][2][2] / (cm_tat1[0][2][1] + cm_tat1[0][2][2] + cm_tat1[0][2][0])) * 100) + (
                      (cm_tat1[0][2][2] / (cm_tak1[0][2][1] + cm_tak1[0][2][2] + cm_tak1[0][2][0])) * 100) +
              ((cm_taj1[0][1][1] / (cm_taj1[0][1][1] + cm_taj1[0][1][2] + cm_taj1[0][1][0])) * 100) + (
                      (cm_tae1[0][1][1] / (cm_tae1[0][1][1] + cm_tae1[0][1][2] + cm_tae1[0][1][0])) * 100) +
              ((cm_tad1[0][2][2] / (cm_tad1[0][2][1] + cm_tad1[0][2][2] + cm_tad1[0][2][0])) * 100)) / 7.0,
             (((cm_taq2[0][2][2] / (cm_taq2[0][2][1] + cm_taq2[0][2][2] + cm_taq2[0][2][0])) * 100) +
              ((cm_tao2[0][2][2] / (cm_tao2[0][2][1] + cm_tao2[0][2][2] + cm_tao2[0][2][0])) * 100) +
              ((cm_tat2[0][2][2] / (cm_tat2[0][2][1] + cm_tat2[0][2][2] + cm_tat2[0][2][0])) * 100) +
              ((cm_tat2[0][2][2] / (cm_tak2[0][2][1] + cm_tak2[0][2][2] + cm_tak2[0][2][0])) * 100) +
              ((cm_taj2[0][1][1] / (cm_taj2[0][1][1] + cm_taj2[0][1][2] + cm_taj2[0][1][0])) * 100) +
              ((cm_tae2[0][1][1] / (cm_tae2[0][1][1] + cm_tae2[0][1][2] + cm_tae2[0][1][0])) * 100) +
              ((cm_tad2[0][2][2] / (cm_tad2[0][2][1] + cm_tad2[0][2][2] + cm_tad2[0][2][0])) * 100)) / 7.0],
            [(((cm_taq1[0][0][0] / (cm_taq1[0][0][1] + cm_taq1[0][0][2] + cm_taq1[0][0][0])) * 100) + (
                    (cm_tbe1[0][2][2] / (cm_tbe1[0][2][1] + cm_tbe1[0][2][2] + cm_tbe1[0][2][0])) * 100) +
              ((cm_taj1[0][2][2] / (cm_taj1[0][2][1] + cm_taj1[0][2][2] + cm_taj1[0][2][0])) * 100) + (
                      (cm_tae1[0][2][2] / (cm_tae1[0][2][1] + cm_tae1[0][2][2] + cm_tae1[0][2][0])) * 100)) / 4.0,
             (((cm_taq2[0][0][0] / (cm_taq2[0][0][1] + cm_taq2[0][0][2] + cm_taq2[0][0][0])) * 100) +
              ((cm_tbe2[0][2][2] / (cm_tbe2[0][2][1] + cm_tbe2[0][2][2] + cm_tbe2[0][2][0])) * 100) +
              ((cm_taj2[0][2][2] / (cm_taj2[0][2][1] + cm_taj2[0][2][2] + cm_taj2[0][2][0])) * 100) +
              ((cm_tae2[0][2][2] / (cm_tae2[0][2][1] + cm_tae2[0][2][2] + cm_tae2[0][2][0])) * 100)) / 4.0]]
    arr = np.arange(2)
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)
    plt.xticks([])
    plt.xlabel('Motor task: left hand, right hand')
    plt.ylabel('Recognition rate of classifier %')
    plt.suptitle('Plot of recognition rate of classifier for class problem: task 1 or task 2')
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'], loc=4)
    plt.show()

    data = [[(((cm_taq1[0][2][2] / (cm_taq1[0][2][1] + cm_taq1[0][2][2] + cm_taq1[0][2][0])) * 100) + (
            (cm_tao1[0][2][2] / (cm_tao1[0][2][1] + cm_tao1[0][2][2] + cm_tao1[0][2][0])) * 100) +
              ((cm_tat1[0][2][2] / (cm_tat1[0][2][1] + cm_tat1[0][2][2] + cm_tat1[0][2][0])) * 100) + (
                      (cm_tat1[0][2][2] / (cm_tak1[0][2][1] + cm_tak1[0][2][2] + cm_tak1[0][2][0])) * 100) +
              ((cm_taj1[0][1][1] / (cm_taj1[0][1][1] + cm_taj1[0][1][2] + cm_taj1[0][1][0])) * 100) + (
                      (cm_tae1[0][1][1] / (cm_tae1[0][1][1] + cm_tae1[0][1][2] + cm_tae1[0][1][0])) * 100) +
              ((cm_tad1[0][2][2] / (cm_tad1[0][2][1] + cm_tad1[0][2][2] + cm_tad1[0][2][0])) * 100)) / 7.0,
             (((cm_taq2[0][2][2] / (cm_taq2[0][2][1] + cm_taq2[0][2][2] + cm_taq2[0][2][0])) * 100) +
              ((cm_tao2[0][2][2] / (cm_tao2[0][2][1] + cm_tao2[0][2][2] + cm_tao2[0][2][0])) * 100) +
              ((cm_tat2[0][2][2] / (cm_tat2[0][2][1] + cm_tat2[0][2][2] + cm_tat2[0][2][0])) * 100) +
              ((cm_tat2[0][2][2] / (cm_tak2[0][2][1] + cm_tak2[0][2][2] + cm_tak2[0][2][0])) * 100) +
              ((cm_taj2[0][1][1] / (cm_taj2[0][1][1] + cm_taj2[0][1][2] + cm_taj2[0][1][0])) * 100) +
              ((cm_tae2[0][1][1] / (cm_tae2[0][1][1] + cm_tae2[0][1][2] + cm_tae2[0][1][0])) * 100) +
              ((cm_tad2[0][2][2] / (cm_tad2[0][2][1] + cm_tad2[0][2][2] + cm_tad2[0][2][0])) * 100)) / 7.0],
            [(((cm_tao1[0][0][0] / (cm_tao1[0][0][1] + cm_tao1[0][0][2] + cm_tao1[0][0][0])) * 100) + (
                    (cm_tat1[0][0][0] / (cm_tat1[0][0][1] + cm_tat1[0][0][2] + cm_tat1[0][0][0])) * 100) +
              ((cm_tbe1[0][0][0] / (cm_tbe1[0][0][1] + cm_tbe1[0][0][2] + cm_tbe1[0][0][0])) * 100) + (
                      (cm_tak1[0][0][0] / (cm_tak1[0][0][1] + cm_tak1[0][0][2] + cm_tak1[0][0][0])) * 100) +
              ((cm_tad1[0][0][0] / (cm_tad1[0][0][1] + cm_tad1[0][0][2] + cm_tad1[0][0][0])) * 100)) / 5.0,
             (((cm_tao2[0][0][0] / (cm_tao2[0][0][1] + cm_tao2[0][0][2] + cm_tao2[0][0][0])) * 100) + (
                     (cm_tat2[0][0][0] / (cm_tat2[0][0][1] + cm_tat2[0][0][2] + cm_tat2[0][0][0])) * 100) +
              ((cm_tbe2[0][0][0] / (cm_tbe2[0][0][1] + cm_tbe2[0][0][2] + cm_tbe2[0][0][0])) * 100) + (
                      (cm_tak2[0][0][0] / (cm_tak2[0][0][1] + cm_tak2[0][0][2] + cm_tak2[0][0][0])) * 100) +
              ((cm_tad2[0][0][0] / (cm_tad2[0][0][1] + cm_tad2[0][0][2] + cm_tad2[0][0][0])) * 100)) / 5.0]]
    arr = np.arange(2)
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)

    plt.xticks([])
    plt.xlabel('Motor task: left hand, feet')
    plt.ylabel('Recognition rate of classifier %')
    plt.suptitle('Plot of recognition rate of classifier for class problem: task 1 or task 2')
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'], loc=4)
    plt.show()

    ##########################################################################################################
    ##########################################################################################################


    # LEAVE ONE OUT CV BAR PLOT overall accuracy rest vs task
    data = [[loo_tad1[0], loo_tae1[0], loo_taj1[0], loo_tak1[0], loo_tao1[0], loo_taq1[0], loo_tat1[0], loo_tbe1[0]],
            [loo_tad3[0], loo_tae3[0], loo_taj3[0], loo_tak3[0], loo_tao3[0], loo_taq3[0], loo_tat3[0], loo_tbe3[0]]]
    arr = np.arange(8)
    err = [[loo_tad1[1], loo_tae1[1], loo_taj1[1], loo_tak1[1], loo_tao1[1], loo_taq1[1], loo_tat1[1], loo_tbe1[1]],
           [loo_tad3[1], loo_tae3[1], loo_taj3[1], loo_tak3[1], loo_tao3[1], loo_taq3[1], loo_tat3[1], loo_tbe3[1]]]
    plt.xlabel('Subjects')
    plt.ylabel('Prediction score (%)')
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25, yerr=err[0], capsize=3)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25, yerr=err[1], capsize=3)
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'])
    plt.suptitle('Prediction scores for each subject comparing SVM and LR classifiers')
    plt.show()

    # LEAVE ONE OUT CV BAR PLOT overall accuracy task 1 vs task 2
    data = [[loo_tad2[0], loo_tae2[0], loo_taj2[0], loo_tak2[0], loo_tao2[0], loo_taq2[0], loo_tat2[0], loo_tbe2[0]],
            [loo_tad4[0], loo_tae4[0], loo_taj4[0], loo_tak4[0], loo_tao4[0], loo_taq4[0], loo_tat4[0], loo_tbe4[0]]]
    arr = np.arange(8)
    err = [[loo_tad2[1], loo_tae2[1], loo_taj2[1], loo_tak2[1], loo_tao2[1], loo_taq2[1], loo_tat2[1], loo_tbe2[1]],
           [loo_tad4[1], loo_tae4[1], loo_taj4[1], loo_tak4[1], loo_tao4[1], loo_taq4[1], loo_tat4[1], loo_tbe4[1]]]
    plt.xlabel('Subjects')
    plt.ylabel('Prediction score (%)')
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25, yerr=err[0], capsize=3)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25, yerr=err[1], capsize=3)
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'])
    plt.suptitle('Prediction scores for each subject comparing SVM and LR classifiers')
    plt.show()

    # LEAVE ONE OUT CV BAR PLOT overall task 1 vs task 2 vs task 3 accuracy PLOTS
    data = [[((l_cm_tad1[1] + l_cm_tak1[1] + l_cm_tao1[1] + l_cm_tat1[1]) / 4.0) * 100,
             ((l_cm_tae1[1] + l_cm_taj1[1] + l_cm_taq1[1]) / 3.0) * 100, l_cm_tbe1[1] * 100],
            [((l_cm_tad2[1] + l_cm_tak2[1] +
               l_cm_tao2[1] + l_cm_tat2[1]) / 4.0) * 100, ((l_cm_tae2[1] + l_cm_taj2[1] + l_cm_taq2[1]) / 3.0) * 100,
             l_cm_tbe2[1] * 100]]
    arr = np.arange(3)
    plt.xlabel('Motor Task Classification Problem in order 1 vs 3, 1 vs 2, 2 vs 3')
    plt.ylabel('Prediction score (%)')
    plt.xticks([])
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'], loc=4)
    plt.suptitle(
        'Plot of variation in classification accuracy against task performed: left hand(1), right hand(2) or feet(3)')
    plt.show()

    # LEAVE ONE OUT CV BAR PLOT prediction accuracy rest vs task and task 1 vs task 2 for each subject for SVM classifier
    data = [[loo_tad1[0], loo_tae1[0], loo_taj1[0], loo_tak1[0], loo_tao1[0], loo_taq1[0], loo_tat1[0], loo_tbe1[0]],
            [loo_tad2[0], loo_tae2[0], loo_taj2[0], loo_tak2[0], loo_tao2[0], loo_taq2[0], loo_tat2[0], loo_tbe2[0]]]
    arr = np.arange(8)
    plt.xlabel('Subjects')
    plt.ylabel('Prediction score (%)')
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)
    plt.legend(labels=['rest vs task', 'task 1 vs task 2'], loc=4)
    plt.suptitle(
        'Plot of variation in SVM classification accuracy against classification problems one and two')
    plt.show()

    # LEAVE ONE OUT CV BAR PLOT prediction accuracy rest vs task and task 1 vs task 2 for each subject for LR classifier
    data = [[loo_tad3[0], loo_tae3[0], loo_taj3[0], loo_tak3[0], loo_tao3[0], loo_taq3[0], loo_tat3[0], loo_tbe3[0]],
            [loo_tad4[0], loo_tae4[0], loo_taj4[0], loo_tak4[0], loo_tao4[0], loo_taq4[0], loo_tat4[0], loo_tbe4[0]]]
    arr = np.arange(8)
    plt.xlabel('Subjects')
    plt.ylabel('Prediction score (%)')
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)
    plt.legend(labels=['rest vs task', 'task 1 vs task 2'], loc=4)
    plt.suptitle(
        'Plot of variation in LR classification accuracy against classification problems one and two')
    plt.show()

    ## LEAVE ONE OUT CV BAR PLOT of recognition rate of classifiers
    # overall rest vs task recognition rate
    data = [[(((l_cm_tada[0][0][0] / (l_cm_tada[0][0][0] + l_cm_tada[0][1][0])) + (
            l_cm_taka[0][0][0] / (l_cm_taka[0][0][0] + l_cm_taka[0][1][0])) +
               (l_cm_taoa[0][0][0] / (l_cm_taoa[0][0][0] + l_cm_taoa[0][1][0])) + (
                       l_cm_tata[0][0][0] / (l_cm_tata[0][0][0] + l_cm_tata[0][1][0])) +
               (l_cm_taea[0][0][0] / (l_cm_taea[0][0][0] + l_cm_taea[0][1][0])) + (
                       l_cm_taja[0][0][0] / (l_cm_taja[0][0][0] + l_cm_taja[0][1][0])) +
               (l_cm_taqa[0][0][0] / (l_cm_taqa[0][0][0] + l_cm_taqa[0][1][0])) + (
                       l_cm_tbea[0][0][0] / (l_cm_tbea[0][0][0] + l_cm_tbea[0][1][0]))) * 100 / 8.0),
             (((l_cm_tad[0][0][0] / (l_cm_tad[0][0][0] + l_cm_tad[0][1][0])) + (
                     l_cm_tak[0][0][0] / (l_cm_tak[0][0][0] + l_cm_tak[0][1][0])) +
               (l_cm_tao[0][0][0] / (l_cm_tao[0][0][0] + l_cm_tao[0][1][0])) + (
                       l_cm_tat[0][0][0] / (l_cm_tat[0][0][0] + l_cm_tat[0][1][0])) +
               (l_cm_tae[0][0][0] / (l_cm_tae[0][0][0] + l_cm_tae[0][1][0])) + (
                       l_cm_taj[0][0][0] / (l_cm_taj[0][0][0] + l_cm_taj[0][1][0])) +
               (l_cm_taq[0][0][0] / (l_cm_taq[0][0][0] + l_cm_taq[0][1][0])) + (
                       l_cm_tbe[0][0][0] / (l_cm_tbe[0][0][0] + l_cm_tbe[0][1][0]))) * 100 / 8.0)],
            [(((l_cm_tada[0][1][1] / (l_cm_tada[0][1][0] + l_cm_tada[0][1][1])) + (
                    l_cm_taka[0][1][1] / (l_cm_taka[0][1][0] + l_cm_taka[0][1][1])) +
               (l_cm_taoa[0][1][1] / (l_cm_taoa[0][1][0] + l_cm_taoa[0][1][1])) + (
                       l_cm_tata[0][1][1] / (l_cm_tata[0][1][0] + l_cm_tata[0][1][1])) +
               (l_cm_taea[0][1][1] / (l_cm_taea[0][1][0] + l_cm_taea[0][1][1])) + (
                       l_cm_taja[0][1][1] / (l_cm_taja[0][1][0] + l_cm_taja[0][1][1])) +
               (l_cm_taqa[0][1][1] / (l_cm_taqa[0][1][0] + l_cm_taqa[0][1][1])) + (
                       l_cm_tbea[0][1][1] / (l_cm_tbea[0][1][0] + l_cm_tbea[0][1][1]))) * 100 / 8.0),
             (((l_cm_tad[0][1][1] / (l_cm_tad[0][1][0] + l_cm_tad[0][1][1])) + (
                     l_cm_tak[0][1][1] / (l_cm_tak[0][1][0] + l_cm_tak[0][1][1])) +
               (l_cm_tao[0][1][1] / (l_cm_tao[0][1][0] + l_cm_tao[0][1][1])) + (
                       l_cm_tat[0][1][1] / (l_cm_tat[0][1][0] + l_cm_tat[0][1][1])) +
               (l_cm_tae[0][1][1] / (l_cm_tae[0][1][0] + l_cm_tae[0][1][1])) + (
                       l_cm_taj[0][1][1] / (l_cm_taj[0][1][0] + l_cm_taj[0][1][1])) +
               (l_cm_taq[0][1][1] / (l_cm_taq[0][1][0] + l_cm_taq[0][1][1])) + (
                       l_cm_tbe[0][1][1] / (l_cm_tbe[0][1][0] + l_cm_tbe[0][1][1]))) * 100 / 8.0)
             ]]
    arr = np.arange(2)
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)
    plt.xticks([])
    plt.xlabel('Motor task: rest, task')
    plt.ylabel('Recognition rate of classifier %')
    plt.suptitle('Plot of recognition rate of classifier for class problem: rest or task')
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'], loc=4)
    plt.show()

    data = [[(((l_cm_taq1[0][2][2] / (l_cm_taq1[0][2][1] + l_cm_taq1[0][2][2] + l_cm_taq1[0][2][0])) * 100) + (
            (l_cm_tao1[0][2][2] / (l_cm_tao1[0][2][1] + l_cm_tao1[0][2][2] + l_cm_tao1[0][2][0])) * 100) +
              ((l_cm_tat1[0][2][2] / (l_cm_tat1[0][2][1] + l_cm_tat1[0][2][2] + l_cm_tat1[0][2][0])) * 100) + (
                      (l_cm_tat1[0][2][2] / (l_cm_tak1[0][2][1] + l_cm_tak1[0][2][2] + l_cm_tak1[0][2][0])) * 100) +
              ((l_cm_taj1[0][1][1] / (l_cm_taj1[0][1][1] + l_cm_taj1[0][1][2] + l_cm_taj1[0][1][0])) * 100) + (
                      (l_cm_tae1[0][1][1] / (l_cm_tae1[0][1][1] + l_cm_tae1[0][1][2] + l_cm_tae1[0][1][0])) * 100) +
              ((l_cm_tad1[0][2][2] / (l_cm_tad1[0][2][1] + l_cm_tad1[0][2][2] + l_cm_tad1[0][2][0])) * 100)) / 7.0,
             (((l_cm_taq2[0][2][2] / (l_cm_taq2[0][2][1] + l_cm_taq2[0][2][2] + l_cm_taq2[0][2][0])) * 100) +
              ((l_cm_tao2[0][2][2] / (l_cm_tao2[0][2][1] + l_cm_tao2[0][2][2] + l_cm_tao2[0][2][0])) * 100) +
              ((l_cm_tat2[0][2][2] / (l_cm_tat2[0][2][1] + l_cm_tat2[0][2][2] + l_cm_tat2[0][2][0])) * 100) +
              ((l_cm_tat2[0][2][2] / (l_cm_tak2[0][2][1] + l_cm_tak2[0][2][2] + l_cm_tak2[0][2][0])) * 100) +
              ((l_cm_taj2[0][1][1] / (l_cm_taj2[0][1][1] + l_cm_taj2[0][1][2] + l_cm_taj2[0][1][0])) * 100) +
              ((l_cm_tae2[0][1][1] / (l_cm_tae2[0][1][1] + l_cm_tae2[0][1][2] + l_cm_tae2[0][1][0])) * 100) +
              ((l_cm_tad2[0][2][2] / (l_cm_tad2[0][2][1] + l_cm_tad2[0][2][2] + l_cm_tad2[0][2][0])) * 100)) / 7.0],
            [(((l_cm_taq1[0][0][0] / (l_cm_taq1[0][0][1] + l_cm_taq1[0][0][2] + l_cm_taq1[0][0][0])) * 100) + (
                    (l_cm_tbe1[0][2][2] / (l_cm_tbe1[0][2][1] + l_cm_tbe1[0][2][2] + l_cm_tbe1[0][2][0])) * 100) +
              ((l_cm_taj1[0][2][2] / (l_cm_taj1[0][2][1] + l_cm_taj1[0][2][2] + l_cm_taj1[0][2][0])) * 100) + (
                      (l_cm_tae1[0][2][2] / (
                                  l_cm_tae1[0][2][1] + l_cm_tae1[0][2][2] + l_cm_tae1[0][2][0])) * 100)) / 4.0,
             (((l_cm_taq2[0][0][0] / (l_cm_taq2[0][0][1] + l_cm_taq2[0][0][2] + l_cm_taq2[0][0][0])) * 100) +
              ((l_cm_tbe2[0][2][2] / (l_cm_tbe2[0][2][1] + l_cm_tbe2[0][2][2] + l_cm_tbe2[0][2][0])) * 100) +
              ((l_cm_taj2[0][2][2] / (l_cm_taj2[0][2][1] + l_cm_taj2[0][2][2] + l_cm_taj2[0][2][0])) * 100) +
              ((l_cm_tae2[0][2][2] / (l_cm_tae2[0][2][1] + l_cm_tae2[0][2][2] + l_cm_tae2[0][2][0])) * 100)) / 4.0]]
    arr = np.arange(2)
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)
    plt.xticks([])
    plt.xlabel('Motor task: left hand, right hand')
    plt.ylabel('Recognition rate of classifier %')
    plt.suptitle('Plot of recognition rate of classifier for class problem: task 1 or task 2')
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'], loc=4)
    plt.show()

    data = [[(((l_cm_taq1[0][2][2] / (l_cm_taq1[0][2][1] + l_cm_taq1[0][2][2] + l_cm_taq1[0][2][0])) * 100) + (
            (l_cm_tao1[0][2][2] / (l_cm_tao1[0][2][1] + l_cm_tao1[0][2][2] + l_cm_tao1[0][2][0])) * 100) +
              ((l_cm_tat1[0][2][2] / (l_cm_tat1[0][2][1] + l_cm_tat1[0][2][2] + l_cm_tat1[0][2][0])) * 100) + (
                      (l_cm_tat1[0][2][2] / (l_cm_tak1[0][2][1] + l_cm_tak1[0][2][2] + l_cm_tak1[0][2][0])) * 100) +
              ((l_cm_taj1[0][1][1] / (l_cm_taj1[0][1][1] + l_cm_taj1[0][1][2] + l_cm_taj1[0][1][0])) * 100) + (
                      (l_cm_tae1[0][1][1] / (l_cm_tae1[0][1][1] + l_cm_tae1[0][1][2] + l_cm_tae1[0][1][0])) * 100) +
              ((l_cm_tad1[0][2][2] / (l_cm_tad1[0][2][1] + l_cm_tad1[0][2][2] + l_cm_tad1[0][2][0])) * 100)) / 7.0,
             (((l_cm_taq2[0][2][2] / (l_cm_taq2[0][2][1] + l_cm_taq2[0][2][2] + l_cm_taq2[0][2][0])) * 100) +
              ((l_cm_tao2[0][2][2] / (l_cm_tao2[0][2][1] + l_cm_tao2[0][2][2] + l_cm_tao2[0][2][0])) * 100) +
              ((l_cm_tat2[0][2][2] / (l_cm_tat2[0][2][1] + l_cm_tat2[0][2][2] + l_cm_tat2[0][2][0])) * 100) +
              ((l_cm_tat2[0][2][2] / (l_cm_tak2[0][2][1] + l_cm_tak2[0][2][2] + l_cm_tak2[0][2][0])) * 100) +
              ((l_cm_taj2[0][1][1] / (l_cm_taj2[0][1][1] + l_cm_taj2[0][1][2] + l_cm_taj2[0][1][0])) * 100) +
              ((l_cm_tae2[0][1][1] / (l_cm_tae2[0][1][1] + l_cm_tae2[0][1][2] + l_cm_tae2[0][1][0])) * 100) +
              ((l_cm_tad2[0][2][2] / (l_cm_tad2[0][2][1] + l_cm_tad2[0][2][2] + l_cm_tad2[0][2][0])) * 100)) / 7.0],
            [(((l_cm_tao1[0][0][0] / (l_cm_tao1[0][0][1] + l_cm_tao1[0][0][2] + l_cm_tao1[0][0][0])) * 100) + (
                    (l_cm_tat1[0][0][0] / (l_cm_tat1[0][0][1] + l_cm_tat1[0][0][2] + l_cm_tat1[0][0][0])) * 100) +
              ((l_cm_tbe1[0][0][0] / (l_cm_tbe1[0][0][1] + l_cm_tbe1[0][0][2] + l_cm_tbe1[0][0][0])) * 100) + (
                      (l_cm_tak1[0][0][0] / (l_cm_tak1[0][0][1] + l_cm_tak1[0][0][2] + l_cm_tak1[0][0][0])) * 100) +
              ((l_cm_tad1[0][0][0] / (l_cm_tad1[0][0][1] + l_cm_tad1[0][0][2] + l_cm_tad1[0][0][0])) * 100)) / 5.0,
             (((l_cm_tao2[0][0][0] / (l_cm_tao2[0][0][1] + l_cm_tao2[0][0][2] + l_cm_tao2[0][0][0])) * 100) + (
                     (l_cm_tat2[0][0][0] / (l_cm_tat2[0][0][1] + l_cm_tat2[0][0][2] + l_cm_tat2[0][0][0])) * 100) +
              ((l_cm_tbe2[0][0][0] / (l_cm_tbe2[0][0][1] + l_cm_tbe2[0][0][2] + l_cm_tbe2[0][0][0])) * 100) + (
                      (l_cm_tak2[0][0][0] / (l_cm_tak2[0][0][1] + l_cm_tak2[0][0][2] + l_cm_tak2[0][0][0])) * 100) +
              ((l_cm_tad2[0][0][0] / (l_cm_tad2[0][0][1] + l_cm_tad2[0][0][2] + l_cm_tad2[0][0][0])) * 100)) / 5.0]]
    arr = np.arange(2)
    plt.bar(arr + 0.00, data[0], color='blue', width=0.25)
    plt.bar(arr + 0.25, data[1], color='grey', width=0.25)
    plt.xticks([])
    plt.xlabel('Motor task: left hand, feet')
    plt.ylabel('Recognition rate of classifier %')
    plt.suptitle('Plot of recognition rate of classifier for class problem: task 1 or task 2')
    plt.legend(labels=['Support Vector Machine', 'Logistic Regression'], loc=4)
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
