### Connecting Attributions and QA Model Behavior on Realistic Counterfactuals

#### Abstract

采用检测反事实实例与预测结果是否一致的方法，描述了不同归因方法对阅读理解（QA）解释的正确性和实用性。

结论：

1. 成对归因的方法比单标记归因更加合适阅读理解的归因
2. 给出了更适合归因的LATATTR模型。

#### Introduction

随着神经网络可解释性的掀起， 产生了许多事后归因方法，包括标记归因方法。但是我们很难评估这些方法产生的解释是否与模型一致。因此，错误的归因方法得出的结论很容易误导读者。同样，有些解释很难有社会效应，所以很难直接用到下游任务当中。

本文用反事实实例揭示模型的更高级的行为。这里的高级行为不是某一个词语更重要（标记级别，低级），而是通过比较两个词语的相关性从而给出结论。

方法：

![](E:\沈驰皓\教材\需要的教材和课程\NLP\report\files\4_1.png)

主要贡献：

1. 给出检测归因是否正确的新方法：检测反事实实例
2. 给出连接低级与高级行为的方法
3. 给出基于注意力的改进模型
4. 训练并测试QA任务，得出改进模型具有更好的表现等结论

#### Motivation

单标记方法INTGRAD和DIFFTASK在归因RoBERTa模型回答问题时都把重点放在了"documentary"这个词语上。

![](E:\沈驰皓\教材\需要的教材和课程\NLP\report\files\4_2.png)

在这里作者给出的假设是：一个好的解释必须建立在能正确解释现实的反事实实例的基础上。

然而实际上RoBERTa并没有在语义上理解并聚焦在正确的词语上，因此实际关注重点并不在"documentary"这个词语上。

当我们输入三个对归因特征词语"documentary"进行扰动后的反事实实例，发现RoBERTa得出了错误的答案，由此发现，这两个单标记归因模型的解释并不准确。

过去测量准确性的方法：删除片段，内插输入词语

​	弊端：影响了输入的分布，并且使句子失去了语义性

因此反事实实例在原有句子的基础上进行轻微扰动，很好的规避了这两个弊端，适用于阅读理解任务的归因。

#### 反事实实例的建立和表现

过程：

1. 建立对解释的假说H（该模型比较目标特征"documentary"）
2. 建立反事实组
3. 对基本实例（未经扰动）做出解释，并用这个解释去模拟反事实实例的结论

模拟方法：

1. 计算假说中目标特征集对于不同归因方法的重要因子f。对于单标记方法，仅需检测每个目标特征的重要因子si并对其求和得到f。对于特征交互的成对归因方法，我们两两成对提取目标特征集中的目标特征ij，计算重要因子sij并求和得到f。
2. 如果超过阈值，赋值z=1，表示反事实实例预测结果会发生改变。如果没有超过阈值，赋值z=0，表示预测结果不会发生改变。

#### 解释方法

##### 基于单标记归因

**LIME / SHAP**

采用线性模型计算样本及其周围的微小扰动的样本集所得出的归因值，即该线性模型在该特征上的对应权重。

**Integrated Gradient**

计算基线input和给定input按一定路径的梯度积分。基线input在nlp中通常为[0000]，尽管这种方法在其它领域很常用，但被证实在nlp领域并不高效。

**Differentiable Mask**

屏蔽一部分的标记，同时使分布尽可能接近原始分布，然后用单层神经网络判断哪个标记需要被舍弃。

##### 基于特征交互

**Archipelago**

测量非线性叠加的特征交互。用mask屏蔽大部分标记，只保留少量包含特征标记的部分，导致input变得无意义，因此对QA的归因解释不是十分准确。

**Attention Attribution**

使用注意力机制得出成对标记的解释。采用类似梯度积分的方式计算不同层对某两个标记的注意力分数的叠加得分作为归因分数。

**Layer-wise Attention Attribution**

在Attention Attribution的基础上做出改进，防止该模型的注意力数值随着层数的叠加而自发增加。因此该方法对每层进行单独计算，在计算当前层时屏蔽和干预当前层，其余层保持不变。


#### 实验和结果

##### Hotpot QA & Bridge Question

Hotpot QA: yes / no

Bridge Question: 两个单跳问题（简单问题）加上一个解释。在给反事实实例时，我们在文章最后加一个和第一个问题相关的干扰句。

![](E:\沈驰皓\教材\需要的教材和课程\NLP\report\files\4_4.png)

![](E:\沈驰皓\教材\需要的教材和课程\NLP\report\files\4_3.png)

 **SQuAD Adversarial**

采用addSent法对关键句的关键词进行改造：

Where was he born in 2001?

He was born in London in 2001.

=> He studied in China in 2010.

![](E:\沈驰皓\教材\需要的教材和课程\NLP\report\files\4_5.png)

结论：成对归因的方法比大多数的单标记方法都要优秀，且作者改进的LATATTR在大多数领域对模型的归因解释都十分优秀。