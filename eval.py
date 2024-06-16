# Copyright (C) 2019  National Center of Tumor Diseases (NCT) Dresden, Division of Translational Surgical Oncology

import numpy as np
import os
import argparse

import matplotlib
# matplotlib.use('Agg')  # uncomment for remote execution (e.g. using ssh)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

splits_LOSO = ['data_1.csv', 'data_2.csv', 'data_3.csv', 'data_4.csv', 'data_5.csv']
splits_LOUO = ['data_B.csv', 'data_C.csv', 'data_D.csv', 'data_E.csv', 'data_F.csv', 'data_G.csv', 'data_H.csv', 'data_I.csv']
splits_LOUO_NP = ['data_B.csv', 'data_C.csv', 'data_D.csv', 'data_E.csv', 'data_F.csv', 'data_H.csv', 'data_I.csv']


def calc_metrics(num_class, results):
    eval = np.zeros([num_class, 3], dtype=np.int64)
    for d in results:
        pred = int(d[0])
        label = int(d[1])

        if pred == label:
            eval[pred, 0] += 1  # True positive
        else:
            eval[pred, 1] += 1  # False positive
            eval[label, 2] += 1  # False negative

    acc = np.sum(eval[:, 0]) / len(results)

    avg_recall = []
    avg_precision = []
    avg_f1 = []
    for p in range(num_class):
        TP = eval[p, 0]
        FP = eval[p, 1]
        FN = eval[p, 2]
        if TP + FN > 0:
            recall = TP / (TP + FN)
            precision = 0
            f1 = 0
            if TP > 0:
                precision = TP / (TP + FP)
                f1 = (2 * precision * recall) / (precision + recall)

            avg_recall.append(recall)
            avg_precision.append(precision)
            avg_f1.append(f1)

    avg_recall_score = np.mean(avg_recall)
    avg_precision_score = np.mean(avg_precision)
    avg_f1_score = np.mean(avg_f1)

    return acc, avg_recall_score, avg_precision_score, avg_f1_score


def create_confusion_matrix():
    global args

    epoch = args.model_no

    out_dir = os.path.join(args.model_dir, "ConfMat")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    conf_mat = np.zeros([3, 3], dtype=np.int64)
    for exp in args.exp:
        predictions_file = os.path.join(args.model_dir, exp, args.eval_scheme, "eval_" + str(epoch) + ".csv")
        if not os.path.isfile(predictions_file):
            print("Cannot find predictions file " + predictions_file)
            return
        data = np.genfromtxt(predictions_file, delimiter=",")
        for d in data:
            pred = int(d[1])
            label = int(d[2])
            conf_mat[label][pred] += 1
    print(conf_mat)

    df_cm = pd.DataFrame(conf_mat, index=["Novice", "Interm.", "Expert"], columns=["Novice", "Interm.", "Expert"])
    plt.figure(figsize=(10, 8))
    sn.set(font_scale=2.2)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 24},  cmap=plt.get_cmap('Blues'), cbar=False, square=True)

    plt.yticks(rotation=0)
    plt.ylabel("Actual labels")
    plt.xlabel("Predicted labels")

    # plt.show()
    plt.savefig(os.path.join(out_dir, args.out_file))
    plt.close()


def calc_average():
    global args

    epoch = args.model_no

    acc = []
    avg_recall = []
    avg_precision = []
    avg_f1 = []
    for exp in args.exp:
        results_file = os.path.join(args.model_dir, "eval_{}_{}_{}.csv".format(exp, args.eval_scheme, epoch))
        if not os.path.isfile(results_file):
            print("Cannot find file " + results_file)
            return
        f = open(results_file, 'r')
        f.readline()  # skip header
        results = f.readline()
        f.close()
        results = results.split(' ')
        assert(int(results[0]) == epoch)

        acc.append(float(results[1]))
        avg_recall.append(float(results[2]))
        avg_precision.append(float(results[3]))
        avg_f1.append(float(results[4]))

    results_log = open(os.path.join(args.model_dir, args.out_file), "w")
    results_log.write("epoch: " + str(epoch) + os.linesep)
    results_log.write("--- acc avg_recall avg_precision avg_f1" + os.linesep)
    msg = "avg {:.4f} {:.4f} {:.4f} {:.4f}"\
        .format(np.mean(acc), np.mean(avg_recall), np.mean(avg_precision), np.mean(avg_f1))
    print(msg)
    results_log.write(msg + os.linesep)
    msg = "std {:.4f} {:.4f} {:.4f} {:.4f}"\
        .format(np.std(acc, ddof=1), np.std(avg_recall, ddof=1), np.std(avg_precision, ddof=1), np.std(avg_f1, ddof=1))
    print(msg)
    results_log.write(msg + os.linesep)
    results_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--eval_mode', type=str, choices=['avg', 'conf_mat'], default='conf_mat',
                        help="Choose \'avg\' to compute mean experimental results or \'conf_mat\' to generate "
                             "confusion matrices.")
    parser.add_argument('--exp', type=str, nargs='+', required=True,
                        help="Name(s) of the experiment(s) to evaluate (including auto-generated timestamp). "
                             "If more than one experiment is given, results will be accumulated over all experiments.")
    parser.add_argument('--eval_scheme', type=str, choices=['LOSO', 'LOUO'], default='LOSO',
                        help="Cross-validation scheme to use: Leave one supertrial out (LOSO) or Leave one user out ("
                             "LOUO).")
    parser.add_argument('--model_dir', type=str, default="?",
                        help="Path to the folder where the models for the relevant experiment(s) are stored. "
                             "Usually identical to <out> as specified during training.")
    parser.add_argument('--model_no', type=int, default=1199,
                        help="Number of the model to evaluate (= number of epochs for which the model has been "
                             "trained - 1).")
    parser.add_argument('--out_file', type=str, required=True, help="Name of the file that will be generated.")
    args = parser.parse_args()

    if args.eval_mode == 'conf_mat':
        create_confusion_matrix()
    elif args.eval_mode == 'avg':
        calc_average()
