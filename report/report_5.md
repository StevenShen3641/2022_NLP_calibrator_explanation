1. 用更加简洁直观的方式叙述
2. 用更通俗的方式表示实验数据
3. 语义更加贴近主题



## A Baseline for Detecting Misclassified and OOD Examples in Neural Networks

### Abstract

作者基于softmax函数的概率分布，通过探测样本是否是错误分类样本或是OOD的基线模型，得出了以下结论：

1. 正确进行分类的样本通常比错误地分类或错误评估OOD样本具有更高的maximum softmax值
2. 错误地分类或错误评估OOD样本同样具有较高的softmax值，因此该值不能作为辨别样本是否错误或OOD的指标

并且作者认为由于这个基线仍有被超越的可能，因此这个方向有很大的研究空间

### Introduction

当测试样本的分布和训练样本不同时，分类器很有可能分类失败，并且这些错误分类结果往往具有较高的置信度，很难被察觉。这些高置信预测通常是由softmax函数导致的，微小变化在softmax函数中往往回导致结果产生重大变化（指数模型），因此即使结果是错的，其softmax值也往往较大。

但是另一方面，由于正确分类或ID的样本softmax值相对更高，我们可以通过统计数据来区分正确性，只是不能纯粹将softmax值作为衡量置信度的标准。

### 问题形成和研究

问题类型：

**error and success prediction**

**in- and out-of-distribution detection**

评估方法：

前提：测试样本的分布要能够从图像上显示出来，因为如果样本过于集中在positive或是nagative的一类上，会产生误导：看似模型的准确率比较高，但实际上只是对少样本的类区分率不够。给定一百张水果的测试照片，（p = 苹果）有99张都是苹果的照片，那即使把最后一张香蕉也辨认成苹果，也具有很高的置信度，但显然是不合理的。

方法：

|      | t              | f              |
| ---- | -------------- | -------------- |
| p    | 预测为p的p样本 | 预测为p的n样本 |
| n    | 预测为n的n样本 | 预测为n的p样本 |

实际p样本：tp，fn

实际n样本：tn，fp

**AUROC**: 

横坐标：false positive rate (fpr = fp*/*(fp + tn))

纵坐标：true positive rate (tpr = tp*/*(tp + fn))

阈值可以取[0,1）之间的任意值，可以取无数个ROC曲线上的点，而把所有的点表示在同一个二维空间中的方法称为ROC曲线。当阈值为0时，坐标为（0，0），阈值接近1时，坐标为（1，1）。

AUROC表示ROC曲线下的面积大小，是一个标量。AUROC值越大的分类器，正确率越高，因为当FPR相同时，面积越大，TPR相对越高。

同时，研究表明AUROC值相当于一个positive class比一个negative class检测分数高的概率，及P(S>E)，等价于P(-E<-S)，即negative class的检测分数的负值小于positive class检测分数的负值的概率。

但是AUROC的一个问题是其不能够反映测试样本的分布，如下图所示，因此我们还需要引入AUPR。

**AUPR**

纵坐标：Precision: 查准率 = TP/(TP + FP)

很坐标：Recall: 查全率 = TP/(TP +FＮ)

当阈值越大时，预测positive为tp的概率越大，fn也越多，因此可以到达（0，1），但是和（1，0）的距离与测试样本的分布有关，negative样本越多，越能达到（1，0）。

AUPR同样表示曲线下面积大小，但其得到的信息比AUROC更多，因为当测试样本各个类的base rate不同时，影响的是与（1， 0）的接近程度，从而会影响面积的大小，如下图所示。所以其不仅能反映模型的好坏，还能反应测试样本不同类的分布。

![](E:\沈驰皓\教材\需要的教材和课程\NLP\report\files\5_1.gif)

###  采用softmax预测概率作为基线模型

我们通过softmax函数获取最大的类概率，即预测的类概率，并将该类作为该样本的label，从而检测一个例子是被错误地分类还是在分布外。

对于问题一：

1. 根据softmax值，将正确分类的样本作为positive class (SUCC)
2. 根据softmax的负值，将错误分类的样本作为positive class (ERR)

对于问题二：

1. 根据softmax值，将正确分类的ID样本作为positive class (In)
2. 根据softmax的负值，将OOD的样本作为positive class (Out)

将以上两类结果分别得出AUROC和AUPR，作为模型的置信度。由于P(S>E) = P(-E<-S)，因此两类的AUROC值是相同的。

由于样本分布会导致AUPR值产生重大变化，因此我们需要一个基准的比较参数进行比较，从而得出模型的好坏。我们取一个随机样本分类器作为base value，得出base value的AUPR和AUROC作为对照组进行比较。

Pred. Prob Wrong: mean softmax prob of error classified example.

Pred. Prob : mean softmax prob of OOD example.

#### CV

![](files\5_2.png)

理论上当一个样本预测错时，mean softmax prob应该显得更为平均，即都接近50%，但实际观察发现其mean softmax值都非常高，因此其不适合作为样本预测正确与否的置信率

#### NLP

**语义分类**

![](files\5_3.png)

**文本分类**（略）

**词性分类**

![](files\5_4.png)

![](files\5_5.png)

WSJ时华尔街日报的语料库，由于Weblog在词性风格上更接近WSJ，因此在探测ID和OOD任务上有更糟糕的表现。同样，其Pred. Prob几乎接近1，因此有着很差的预测置信率。