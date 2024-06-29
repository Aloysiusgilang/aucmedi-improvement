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
from sklearn.metrics import roc_curve, roc_auc_score

#-----------------------------------------------------#
#         Computation: Classification Metrics         #
#-----------------------------------------------------#
def compute_metrics(preds, labels, n_labels, threshold=None):
    """ Function for computing various classification metrics.

    !!! info "Computed Metrics"
        F1, Accuracy, Sensitivity, Specificity, AUROC (AUC), Precision, FPR, FNR,
        FDR, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Balanced Accuracy

    Args:
        preds (numpy.ndarray):          A NumPy array of predictions formatted with shape (n_samples, n_labels).
        labels (numpy.ndarray):         Classification list with One-Hot Encoding.
        n_labels (int):                 Number of classes.
        threshold (float):              Only required for multi_label data. Threshold value if prediction is positive.

    Returns:
        metrics (pandas.DataFrame):     Dataframe containing all computed metrics (except ROC).
    """
    # Perclass metrics
    per_class_metrics = [compute_class_metrics(preds, labels, c, threshold) for c in range(n_labels)]
    df_per_class = pd.concat(per_class_metrics, axis=0, ignore_index=True) 
    
    # Balanced Accuracy
    balanced_accuracy = compute_balanced_accuracy(per_class_metrics)
    df_balanced_accuracy = pd.DataFrame({
        "metric": ["Balanced Accuracy"],
        "score": [balanced_accuracy],
        "class": ["All"]
    })
    
    df_final = pd.concat([df_per_class, df_balanced_accuracy], axis=0, ignore_index=True)
    return df_final

def compute_class_metrics(preds, labels, class_index, threshold=None):
    """ Function for computing metrics for a specific class.
    
    Args:
        preds (numpy.ndarray):          A NumPy array of predictions.
        labels (numpy.ndarray):         Classification list with One-Hot Encoding.
        class_index (int):              Index of the class to compute metrics for.
        threshold (float):              Threshold value if prediction is positive.
        
    Returns:
        metrics (pandas.DataFrame):     Dataframe containing metrics for the specified class.
    """
    data_dict = {}
    
    truth, pred, pred_prob = get_class_predictions(preds, labels, class_index, threshold)
    
    # Compute confusion matrix
    tp, tn, fp, fn = compute_CM(truth, pred)
    data_dict["TP"] = tp
    data_dict["TN"] = tn
    data_dict["FP"] = fp
    data_dict["FN"] = fn

    # Compute metrics based on confusion matrix
    data_dict.update(compute_confusion_metrics(tp, tn, fp, fn))
    
    # Compute AUROC
    data_dict["AUC"] = compute_auc(truth, pred_prob)
    
    # Parse metrics to dataframe
    df = pd.DataFrame.from_dict(data_dict, orient="index", columns=["score"]).reset_index()
    df.rename(columns={"index": "metric"}, inplace=True)
    df["class"] = class_index
    
    return df

def get_class_predictions(preds, labels, class_index, threshold=None):
    """ Helper function to get truth labels, predictions and probabilities for a specific class.
    
    Args:
        preds (numpy.ndarray):          A NumPy array of predictions.
        labels (numpy.ndarray):         Classification list with One-Hot Encoding.
        class_index (int):              Index of the class.
        threshold (float):              Threshold value if prediction is positive.
        
    Returns:
        truth (numpy.ndarray):          Truth labels for the class.
        pred (numpy.ndarray):           Predictions for the class.
        pred_prob (numpy.ndarray):      Prediction probabilities for the class.
    """
    truth = labels[:, class_index]
    if threshold is None:
        pred_argmax = np.argmax(preds, axis=-1)
        pred = (pred_argmax == class_index).astype(np.int8)
    else:
        pred = np.where(preds[:, class_index] >= threshold, 1, 0)
    pred_prob = preds[:, class_index]
    
    return truth, pred, pred_prob

def compute_confusion_metrics(tp, tn, fp, fn):
    """ Helper function to compute metrics based on confusion matrix.
    
    Args:
        tp (int):                       True Positives.
        tn (int):                       True Negatives.
        fp (int):                       False Positives.
        fn (int):                       False Negatives.
        
    Returns:
        metrics (dict):                 Dictionary containing computed metrics.
    """
    metrics = {
        "Sensitivity": np.divide(tp, tp + fn),
        "Specificity": np.divide(tn, tn + fp),
        "Precision": np.divide(tp, tp + fp),
        "FPR": np.divide(fp, fp + tn),
        "FNR": np.divide(fn, fn + tp),
        "FDR": np.divide(fp, fp + tp),
        "Accuracy": np.divide(tp + tn, tp + tn + fp + fn),
        "F1": np.divide(2 * tp, 2 * tp + fp + fn),
        "LR+": np.divide(np.divide(tp, tp + fn), np.divide(fp, fp + tn)),
    }
    return metrics

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

#-----------------------------------------------------#
#                     Additional Metrics              #
#-----------------------------------------------------#

def compute_auc(truth, pred_prob):
    """ Helper function to compute AUROC.
    
    Args:
        truth (numpy.ndarray):          Truth labels.
        pred_prob (numpy.ndarray):      Prediction probabilities.
        
    Returns:
        auc (float):                    Computed AUROC value.
    """
    try:
        auc = roc_auc_score(truth, pred_prob)
    except ValueError:
        auc = None
        print("ROC AUC score is not defined.")
    return auc

def compute_balanced_accuracy(metrics_per_class):
    """ Function to compute balanced accuracy.
    
    Args:
        metrics_per_class (list):       List of dataframes containing metrics for each class.
        
    Returns:
        balanced_accuracy (float):      Computed balanced accuracy.
    """
    sensitivities = [df[df['metric'] == 'Sensitivity']['score'].values[0] for df in metrics_per_class]
    balanced_accuracy = np.mean(sensitivities)
    return balanced_accuracy
