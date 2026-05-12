import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import seaborn as sn
import pandas as pd
from itertools import cycle
from sklearn.metrics import roc_curve, confusion_matrix


def Plot_Results():
    # New color palette and new markers
    color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']  # Updated colors
    bar_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']  # Same colors for bar plot
    markers = ['*', 'H', 'v', 'X', 'P']
    # Load evaluation data
    for a in range(2):
        Eval = np.load('Eval_Batch.npy', allow_pickle=True)[a]

        # Metrics list
        Terms = ['Accuracy', 'MD', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'ONE-NORM', 'TWO-NORM', 'MAPE']

        # Batch sizes and indices for the terms to plot
        learn = [1, 2, 3, 4, 5]
        Graph_Term = [0,1,2,3,4,5,6,7,8,9]

        for j in range(len(Graph_Term)):
            # Initialize graph array
            Graph = np.zeros((Eval.shape[0], Eval.shape[1]))

            # Populate Graph array with evaluation data
            for k in range(Eval.shape[0]):
                for l in range(Eval.shape[1]):
                    Graph[k, l] = Eval[k, l, Graph_Term[j]]

            # Line Plot
            plt.figure(figsize=(10, 6))
            for idx, (color, marker) in enumerate(zip(color_palette, markers)):
                plt.plot(learn, np.sort(Graph[:, idx]), color=color, linewidth=4, marker=marker,
                         markerfacecolor='white', markersize=8,
                         label=["CSA-HDA-ADDNet", "NGO-HDA-ADDNet", "CWO-HDA-ADDNet",
                     "SCOA-HDA-ADDNet", "MRSCO-HDA-ADDNet"][idx])
            plt.xticks(learn, ['4', '8', '16', '32', '48'], fontsize=10)
            plt.xlabel('Batch Size', fontsize=12)
            plt.ylabel(Terms[Graph_Term[j]], fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            plt.tight_layout()
            path1 = f"./Results/Dataset_{a+1}_{Terms[Graph_Term[j]]}_Line.png"
            plt.savefig(path1)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            # Bar Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            X = np.arange(5)
            for idx, color in enumerate(bar_palette):
                ax.bar(X + idx * 0.15, np.sort(Graph[:, idx + 5]), color=color, edgecolor='k', width=0.15,
                       label=[ "ANN", "LSTM-GRU", "YOLOv3-Mask RCNN", "DDNet", "MRSCO-HDA-ADDNet"][idx])
            ax.set_xticks(X + 0.3)
            ax.set_xticklabels(['4', '8', '16', '32', '48'], rotation=7)
            ax.set_xlabel('Batch Size', fontsize=12)
            ax.set_ylabel(Terms[Graph_Term[j]], fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            plt.tight_layout()
            path2 = f"./Results/Dataset_{a+1}_{Terms[Graph_Term[j]]}_Bar.png"
            plt.savefig(path2)
            plt.show(block=False)
            plt.pause(1)
            plt.close()


def Plot_table():
    for b in range(2):
        Eval = np.load('Eval_Epoch.npy', allow_pickle=True)[b]
        Terms = ['Accuracy', 'MD', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'ONE-NORM', 'TWO-NORM', 'MAPE']

        Algorithm = ['TERMS', "CSA-HDA-ADDNet", "NGO-HDA-ADDNet", "CWO-HDA-ADDNet",
                     "SCOA-HDA-ADDNet", "MRSCO-HDA-ADDNet"]
        Classifier = ['TERMS',  "ANN", "LSTM-GRU", "YOLOv3-Mask RCNN", "DDNet", "MRSCO-HDA-ADDNet"]
        value = Eval[:, :, :]

        Graph_Term = [0, 6]
        for a in range(len(Graph_Term)):
            variation = ['50', '100', '150', '200', '250']

            Table = PrettyTable()
            Table.add_column('Epochs /Algorithm', variation[0:])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Term[a]])
            print('---------------------------------------------------- Dataset '+str(b+1)+' Algorithm Comparison -',
                  Terms[Graph_Term[a]],
                  '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column('Epochs /Classifier', variation[0:])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j + 5, Graph_Term[a]])
            print('---------------------------------------------------  Dataset '+str(b+1)+' Method Comparison -',
                  Terms[Graph_Term[a]],
                  '--------------------------------------------------')
            print(Table)


def Plot_Fitness():
    for a in range(2):
        conv = np.load('Fitness.npy', allow_pickle=True)[a]

        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ["CSA-HDA-ADDNet", "NGO-HDA-ADDNet", "CWO-HDA-ADDNet",
                     "SCOA-HDA-ADDNet", "MRSCO-HDA-ADDNet"]
        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print(
            '--------------------------------------------------Dataset '+str(a+1)+' Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='r', linewidth=3, marker='>', markerfacecolor='blue', markersize=8,
                 label="CSA-HDA-ADDNet")
        plt.plot(iteration, conv[1, :], color='g', linewidth=3, marker='>', markerfacecolor='red', markersize=8,
                 label="NGO-HDA-ADDNet")
        plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='>', markerfacecolor='green', markersize=8,
                 label="CWO-HDA-ADDNet")
        plt.plot(iteration, conv[3, :], color='m', linewidth=3, marker='>', markerfacecolor='yellow', markersize=8,
                 label="SCOA-HDA-ADDNet")
        plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=8,
                 label="MRSCO-HDA-ADDNet")
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        path1 = f"./Results/Dataset_{a+1}_conv.png"
        plt.savefig(path1)
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a


if __name__ == '__main__':
    Plot_Results()
    Plot_table()
    Plot_Fitness()
