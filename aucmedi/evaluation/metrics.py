#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss, cohen_kappa_score, balanced_accuracy_score

#-----------------------------------------------------#
#         Computation: Classification Metrics         #
#-----------------------------------------------------#
def compute_metrics(preds, labels, n_labels, threshold=None):
    """ Function for computing various classification metrics.

    !!! info "Computed Metrics"
        F1, Accuracy, Sensitivity, Specificity, AUROC (AUC), Precision, FPR, FNR,
        FDR, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

    Args:
        preds (numpy.ndarray):          A NumPy array of predictions formatted with shape (n_samples, n_labels). Provided by
                                        [NeuralNetwork][aucmedi.neural_network.model].
        labels (numpy.ndarray):         Classification list with One-Hot Encoding. Provided by
                                        [input_interface][aucmedi.data_processing.io_data.input_interface].
        n_labels (int):                 Number of classes. Provided by [input_interface][aucmedi.data_processing.io_data.input_interface].
        threshold (float):              Only required for multi_label data. Threshold value if prediction is positive.

    Returns:
        metrics (pandas.DataFrame):     Dataframe containing all computed metrics (except ROC).
    """
    def compute_metrics(preds, labels, n_labels, threshold=None):
        df_list = []
        for c in range(n_labels):
            data_dict = {}
            truth = labels[:, c]
            if threshold is None:
                pred_argmax = np.argmax(preds, axis=-1)
                pred = (pred_argmax == c).astype(int)
            else:
                pred = np.where(preds[:, c] >= threshold, 1, 0)
            pred_prob = preds[:, c]

            # Compute the confusion matrix elements
            tp, tn, fp, fn = compute_CM(truth, pred)

            # Metrics from confusion matrix
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            precision = tp / (tp + fp)
            fpr = fp / (fp + tn)
            fnr = fn / (fn + tp)
            fdr = fp / (fp + tp)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * tp / (2 * tp + fp + fn)

            # Additional Metrics
            try:
                auc = roc_auc_score(truth, pred_prob)
            except ValueError:
                auc = float('nan')

            brier_score = brier_score_loss(truth, pred_prob)
            kappa = cohen_kappa_score(truth, pred)
            balanced_acc = balanced_accuracy_score(truth, pred)

            # Likelihood Ratio Positive (LR+)
            lr_plus = sensitivity / (1 - specificity) if specificity != 1 else float('inf')

            # Constructing the DataFrame
            data_dict = {
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Precision": precision,
                "FPR": fpr,
                "FNR": fnr,
                "FDR": fdr,
                "Accuracy": accuracy,
                "F1": f1,
                "AUC": auc,
                "Brier Score": brier_score,
                "Cohen Kappa": kappa,
                "Balanced Accuracy": balanced_acc,
                "LR+": lr_plus
            }

            df = pd.DataFrame(data_dict, index=[0])
            df['Class'] = c
            df_list.append(df)

        df_final = pd.concat(df_list, ignore_index=True)
        return df_final

#-----------------------------------------------------#
#            Computation: Confusion Matrix            #
#-----------------------------------------------------#
def compute_confusion_matrix(preds, labels, n_labels):
    """ Function for computing a confusion matrix.

    Args:
        preds (numpy.ndarray):          A NumPy array of predictions formatted with shape (n_samples, n_labels). Provided by
                                        [NeuralNetwork][aucmedi.neural_network.model].
        labels (numpy.ndarray):         Classification list with One-Hot Encoding. Provided by
                                        [input_interface][aucmedi.data_processing.io_data.input_interface].
        n_labels (int):                 Number of classes. Provided by [input_interface][aucmedi.data_processing.io_data.input_interface].

    Returns:
        rawcm (numpy.ndarray):          NumPy matrix with shape (n_labels, n_labels).
    """
    preds_argmax = np.argmax(preds, axis=-1)
    labels_argmax = np.argmax(labels, axis=-1)
    rawcm = np.zeros((n_labels, n_labels))
    for i in range(0, labels.shape[0]):
        rawcm[labels_argmax[i]][preds_argmax[i]] += 1
    return rawcm

#-----------------------------------------------------#
#             Computation: ROC Coordinates            #
#-----------------------------------------------------#
def compute_roc(preds, labels, n_labels):
    """ Function for computing the data data of a ROC curve (FPR and TPR).

    Args:
        preds (numpy.ndarray):          A NumPy array of predictions formatted with shape (n_samples, n_labels). Provided by
                                        [NeuralNetwork][aucmedi.neural_network.model].
        labels (numpy.ndarray):         Classification list with One-Hot Encoding. Provided by
                                        [input_interface][aucmedi.data_processing.io_data.input_interface].
        n_labels (int):                 Number of classes. Provided by [input_interface][aucmedi.data_processing.io_data.input_interface].
    Returns:
        fpr_list (list of list):        List containing a list of false positive rate points for each class. Shape: (n_labels, tpr_coords).
        tpr_list (list of list):        List containing a list of true positive rate points for each class. Shape: (n_labels, fpr_coords).
    """
    fpr_list = []
    tpr_list = []
    for i in range(0, n_labels):
        truth_class = labels[:, i].astype(int)
        pdprob_class = preds[:, i]
        fpr, tpr, _ = roc_curve(truth_class, pdprob_class)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    return fpr_list, tpr_list

#-----------------------------------------------------#
#                     Subroutines                     #
#-----------------------------------------------------#
# Compute confusion matrix
def compute_CM(gt, pd):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(gt)):
        if gt[i] == 1 and pd[i] == 1 : tp += 1
        elif gt[i] == 1 and pd[i] == 0 : fn += 1
        elif gt[i] == 0 and pd[i] == 0 : tn += 1
        elif gt[i] == 0 and pd[i] == 1 : fp += 1
        else : print("ERROR at confusion matrix", i)
    return tp, tn, fp, fn
