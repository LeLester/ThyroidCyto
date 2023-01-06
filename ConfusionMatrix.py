
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import torchvision
from prettytable import PrettyTable



matrix =  [[969 , 2 ,  0  , 1 ,  0 ,  0],
 [  1, 975,   9,  12,   10,   6],
 [  0,   11, 937,   13,  8,   3],
 [  0,   2,   3, 920,   9,   0],
 [  2,   7,   7,  13, 902,  20],
 [  0,   7,   9,   3,   16, 1002]]
labels = ["Ⅰ","Ⅱ","Ⅲ","Ⅳ","Ⅴ","Ⅵ"]



matrix =np.array(matrix)

map = []
x=0
for i in range(1000):
   if i==0:
       x+=0
   elif i<40:
       x+=0.0075
   elif i<900:
       x+=0.00059
   elif i<1000:
       x+=0.002
   if x<=1:
      map.append(x)
   else:
       print(x)
       map.append(1)
   if i==999:
       print("x={}".format(x))




cmap =plt.cm.coolwarm


newcolors = cmap(map)

newcmap = ListedColormap(newcolors[::1])
print(type(newcmap))
print(type(cmap))

plt.imshow(matrix, interpolation='nearest' ,cmap=newcmap,vmin=0)
plt.xticks(range(6), labels)
plt.yticks(range(6), labels)
clb = plt.colorbar(orientation = 'horizontal', ticks=[0, 100, 200,  300,  400,  500, 600, 700, 800, 900,1000 ])
clb.ax.set_title('title')
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title('Confusion matrix')



for x in range(6):
    for y in range(6):
        info = matrix[y, x]
        plt.text(x, y, info,
                 verticalalignment='center',
                 horizontalalignment='center',
                 color="black" if info > 70 else "white")
plt.tight_layout()
plt.show()





class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list,matrix):
        self.matrix = matrix
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]

        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            print("{}:".format(i+1), "TP:{}".format(TP), "FP:{}".format(FP), "FN:{}".format(FN), "TN:{}".format(TN), "样本数：{}".format(TP+FN))

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):

        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=45)

        plt.yticks(range(self.num_classes), self.labels)

        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')


        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):

                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
        plt.savefig("/confusion_matrix")


A =ConfusionMatrix(6,labels,matrix)
A.summary()
