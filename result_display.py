import os
import json
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable



class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库  将输出打印为列表
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # p: predict, t: GT
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()  # init a table for print
        table.field_names = ["", "Precision", "Recall", "Specificity","F1-Score"]
        for i in range(self.num_classes):  # for each class
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 4) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 4) if TN + FP != 0 else 0.
            F1_Score = round(2*(Recall*Precision)/(Recall+Precision),4)if Recall+Precision != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity,F1_Score])
        print(table)
#样本数混淆矩阵
    # def plot(self):  # plot confusion matrix
    #     matrix = self.matrix
    #     print(matrix)
    #     plt.imshow(matrix, cmap=plt.cm.Blues)  # color from white to blue
    #
    #     plt.xticks(range(self.num_classes), self.labels, rotation=45)
    #     plt.yticks(range(self.num_classes), self.labels)
    #
    #     # show colorbar
    #     plt.colorbar()
    #
    #     plt.xlabel('True Labels')
    #     plt.ylabel('Predicted Labels')
    #     plt.title('Confusion matrix')
    #
    #     # 在图中标注数量/概率信息
    #     thresh = matrix.max() / 2
    #     # Note:
    #     #       x: left -> right; y: top -> bottom
    #     for x in range(self.num_classes):
    #         for y in range(self.num_classes):
    #             # 注意这里的matrix[y, x]不是matrix[x, y]
    #             info = int(matrix[y, x])
    #             plt.text(x, y, info,
    #                      verticalalignment='center',
    #                      horizontalalignment='center',
    #                      color="white" if info > thresh else "black")
    #     plt.tight_layout()
    #     plt.show()
#小数混淆矩阵
    def plot(self):  # plot confusion matrix
        matrix = torch.FloatTensor(self.matrix)
        [row,col]=matrix.shape
        row, col = list(range(row)),list(range(col))

        # for i in row:
        #     matrix[i,col]=matrix[i,col]/matrix[i,col].sum()
        for i in col:
            matrix[row,i]=matrix[row,i]/matrix[row,i].sum()
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)  # color from white to blue

        plt.xticks(range(self.num_classes), self.labels, rotation=0)
        plt.yticks(range(self.num_classes), self.labels)

        # show colorbar
        plt.colorbar()

        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        # Note:
        #       x: left -> right; y: top -> bottom
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = round(float(matrix[y, x]),4)
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
