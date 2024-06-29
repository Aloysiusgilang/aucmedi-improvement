# External Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss

#-----------------------------------------------------#
#         Computation: Classification Metrics         #
#-----------------------------------------------------#
def compute_metrics(preds, labels, n_labels, threshold=None, metrics=None):
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
        metrics (list):                 List of metrics to compute. If None, all metrics are computed.

    Returns:
        metrics (pandas.DataFrame):     Dataframe containing all computed metrics (except ROC).
    """
    # Default list of metrics
    default_metrics = ["TP", "TN", "FP", "FN", "Sensitivity", "Specificity", 
                       "Precision", "FPR", "FNR", "FDR", "Accuracy", "F1", "AUC", "LR+", "Brier Score"]
    if metrics is None:
        metrics = default_metrics
    
    df_list = []
    for c in range(0, n_labels):
        # Initialize variables
        data_dict = {}

        # Identify truth and prediction for class c
        truth = labels[:, c]
        if threshold is None:
            pred_argmax = np.argmax(preds, axis=-1)
            pred = (pred_argmax == c).astype(np.int8)
        else:
            pred = np.where(preds[:, c] >= threshold, 1, 0)
        # Obtain prediction confidence (probability)
        pred_prob = preds[:, c]

        # Compute the confusion matrix
        tp, tn, fp, fn = compute_CM(truth, pred)
        if "TP" in metrics:
            data_dict["TP"] = tp
        if "TN" in metrics:
            data_dict["TN"] = tn
        if "FP" in metrics:
            data_dict["FP"] = fp
        if "FN" in metrics:
            data_dict["FN"] = fn

        # Compute several metrics based on confusion matrix
        if "Sensitivity" in metrics:
            data_dict["Sensitivity"] = np.divide(tp, tp + fn)
        if "Specificity" in metrics:
            data_dict["Specificity"] = np.divide(tn, tn + fp)
        if "Precision" in metrics:
            data_dict["Precision"] = np.divide(tp, tp + fp)
        if "FPR" in metrics:
            data_dict["FPR"] = np.divide(fp, fp + tn)
        if "FNR" in metrics:
            data_dict["FNR"] = np.divide(fn, fn + tp)
        if "FDR" in metrics:
            data_dict["FDR"] = np.divide(fp, fp + tp)
        if "Accuracy" in metrics:
            data_dict["Accuracy"] = np.divide(tp + tn, tp + tn + fp + fn)
        if "F1" in metrics:
            data_dict["F1"] = np.divide(2 * tp, 2 * tp + fp + fn)
        if "LR+" in metrics:
            data_dict["LR+"] = np.divide(np.divide(tp, tp + fn), np.divide(fp, fp + tn))

        # Compute area under the ROC curve
        if "AUC" in metrics:
            try:
                data_dict["AUC"] = roc_auc_score(truth, pred_prob)
            except:
                print("ROC AUC score is not defined.")

        if "Brier Score" in metrics:
            try:
                data_dict["Brier Score"] = brier_score_loss(truth, pred_prob)
            except:
                print("Brier score is not defined.")
        

        # Parse metrics to dataframe
        df = pd.DataFrame.from_dict(data_dict, orient="index", columns=["score"])
        df = df.reset_index()
        df.rename(columns={"index": "metric"}, inplace=True)
        df["class"] = c

        # Append dataframe to list
        df_list.append(df)

    # Combine dataframes
    df_final = pd.concat(df_list, axis=0, ignore_index=True)
    print(df_final)
    # Return final dataframe
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
