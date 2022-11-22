### Pre-trained models for NLP: A survey

### Introduction

substantial work has shown that pre-trained models (PTMs), on the large corpus can learn universal language representations, which are beneficial for downstream NLP tasks and can avoid training a new model from scratch.

本文重点讲述以下四个方面：

（1）全面描述

（1）重新分类

（3）资源共享

（4）未来方向

### Background

#### 2.1 Language Representation Learning

自然语言需要一个好的数学表示，因此我们需要词嵌入来完成。第一代PTM注重于上下文无关的词嵌入，第二代则注重于上下文语境相关的词嵌入。

上下文无关的词嵌入将每个单词用固定的嵌入方法表示，但是有很多弊端，比如遇到一词多义，超出词典范围的单词如何表示等，而上下文相关的词嵌入注重文本语义，语法结构等内容，很好地解决了上述弊端，词语的语义能够随着文本变化的更改而更改

#### 2.2 Neural Contextual Encoders

编码器主要分为序列模型和非序列模型

**序列模型**：按语句顺序获取内容

卷积模型：聚集临近内容并进行卷积操作

循环模型：短期内捕捉上下文文本联系，但很难捕捉单词之间的长期联系

**非序列模型**：建立预制的文字间的语法或语义结构，但这点往往较难

Fully-connected self-attention model：得出语句中每两个词之间的联系，并通过自连接机制计算对应的权重

#### 2.3 为什么要预训练

想要得到一个好的模型就需要增加网络层数以及参数个数，为了防止过拟合，我们也需要大量的样本进行训练。但是标记一个大的nlp数据样本的注释成本是巨大的。

相反，未标记的语料库极容易获取，通过预训练，我们可以先用一个好的数学表示来表示语料库，表示完后的内容对其他任务有极大的帮助：

1、利用学习好的通用语言的表示能够帮助下游任务

2、预训练能够在大规模数据集上学习尽可能好的通用表示，加快了目标任务的收敛

3、对于容量较小的数据，预训练能够被视为一种正则化，从而防止过拟合

### Overview of PTMs

major differences between PTMs are **the usage of contextual encoders, pre-training tasks, and purposes**.

#### 1.1 Pre-training Tasks

supervised learning: training input-output pairs

unsupervised learning: find clusters, densities

self-supervised learning: generate the label of the data automatically

#### 1.2 Language Modeling Task

##### 1.2.1 Language Model (LM: Auto-regressive LM)

表达式：$p(x_{1:T}) = \prod_{t = 1}^Tp(x_t|x_{0:t - 1}) = p(x_1)p(x_2|x_1)p(x3|x_2x_1) ... p(x_n|x_1 ... x_{n - 1})$.

适合处理相关性有关的问题（纠正错词）

**缺点**：按文本顺序拆解，从左至右或从右至左，无法同时获取双向上下文文本信息（解决方法：BiLM）

##### 1.2.2 Masked Language Modeling (MLM)

屏蔽给定句子中特定的词语（15%），训练模型（Bert等，一般为深度双向模型）基于句子中剩余单词推断出该词语

**优点**：获取双向文本信息

**缺点**：

1、mask在微调阶段可能并不出现，即预训练任务和最终任务（微调阶段）并不相同。方法：15%的选中比例中，80%转为mask，10%转为随机单词，10%不发生改变（但是预训练和fine-tuning阶段仍然会有问题）

2、训练速度慢，收敛速度慢，每次只训练15%，所需时间比单向模型长

3、没有考虑单词和单词之间的关联性

###### 1.2.2.1 Sequence-to-sequence MLM

基于encoder-decoder模型，用于处理seq2seq的任务，如：机器翻译，问题回答等

用于MASS，T5等具体模型中

###### 1.2.2.2 Enhanced-MLM (UniLM, XLM)

UniLM: 将mask预测任务扩展到三种方式中：单向预测、定向预测和seq2seq预测

XLM: 将MLM执行在一系列并行双语言的句子组上，实现多种语言的融合（Bert只用于英语）

SpanMLM: 由于Bert模型训练时随机选择，会割裂连续词组内各个单词的关联，SpanMLM随机选取连续多个屏蔽词语并汇总信息

StructureBERT：进一步整合语言结构特点

##### 1.2.3 Permuted Language Modeling (PLM)

PLM将句子各个单词全排列后抽取几种排列顺序，将某几个单词作为目标，根据其余的单词和目标的自然顺序来预测目标。

**优点**：PLM很好地补偿了LM模型单向问题和MLM中mask在预训练与微调阶段参数不一致的问题

**缺点**：实际情况是我们只将最后几个单词作为目标，因为收敛速度较为缓慢

##### 1.2.4 Denoising Autoencoder (DAE)

将输入作失真破坏处理，例如：单个或连续多个单词屏蔽，单词删除（在屏蔽基础上要决定删除位置），语句顺序打乱，“旋转”文档（改变前后顺序，目的是确定起始位置）

目标：恢复原始的未失真的输入

优点和缺点同MLM类似

##### 1.2.5 Contrastive Learning (CTL)

对比学习假设经过观测得到的样本对在语义上比随机抽样的样本对更相似，它需要我们建立一个函数s使得它分配较大的值给正样本对（相似的样本对），将较小的值分配给负样本对，从而达到分类的目的。在语言处理中，它需要给一个正确的句子更高的分，而给一个把其中心词替换为随机词的句子相对低的分

**优点**：复杂度相比LM大大降低，是一个较为理想的模型

###### 1.2.5.1 Deep InfoMax (DIM)

起始于图像处理，作用是最大化图像指代与图像局部区域之间的互信息。InfoWord将其运用到了语言处理上，从而最大化一个句子的指代和一个句子的部分指代的互信息

###### 1.2.5.2 Replaced Token Detection (RTD)

根据上下文语境来预测词语是否替换。（相关模型CBOW-NS，ELECTRA以及WKLM）

###### 1.2.5.3 Next Sentence Prediction (NSP)

选取一系列包含两个句子的句子组，50%第二个句子是第一个句子语义上的下一个句子，50%随机搭配，训练模型理解两个句子间的关系。但是该模型的可靠性受到了后续工作的质疑，没有NSP训练的模型反而优于有NSP训练的模型

用途：问题回答，自然语言推测

###### 1.2.5.4 Sentence Order Prediction (SOP)

SOP将两个连续的部分作为正样本，而调换顺序作为负样本对模型进行训练。相比较NSP而言，由于NSP融合了主题预测和句子间的相关性预测，而前者相较后者更加简单，因此NSP依赖于前者（主题预测），而SOP更加基于后者，因此效果更好

#### 1.3 Taxonomy

角度一：Representation Type

根据指代类型，分为non-contextual和contextual模型

角度二：Architectures

根据主干网络模型分类，主要有LSTM，Transformer Encoder，Transformer Decoder和完整的Transformer构造

角度三：Pre-training task types

根据1.2的task类型分类

角度四：Extensions

根据应用场景进行分类，包括 knowledge-enriched PTMs, multilingual or language-specific PTMs, multi-model PTMs, domain specific PTMs and compressed PTMs，会在第四部分讲到

