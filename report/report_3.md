# Pretrained Transformers Improve Out-of-Distribution Robustness



### Abstract

本文描述了各个模型在处理OOD数据的泛化能力的比较，并得出预训练Transformers（Bert）在处理OOD泛化和监测异常数据能力都要远优于其它模型的结论。检测了影响模型鲁棒性的具体因素。

### Train

我的测试集：

SST-2：包含精炼的专家影评，以及完整的非专业的影评。这项任务是给定句子的情感，类别分为两类正面情感（二分类任务）。我们先对测试集进行训练，再对其它集进行评估。

### Code

#### 1 argparse

命令行参数解析包

```python
import argparse


# 解释
def test2():
    # 解析器description为整个文件的解释（没用）
    parser = argparse.ArgumentParser(description="description")
    # 每个argument解释
    parser.add_argument('a', type=int, default=10, help="first para")
    parser.add_argument('b', type=int, default=5, help='second para')
    # usage: argparse_test.py [-h] a b
    #
    # description
    #
    # positional arguments:
    #   a           first para
    #   b           second para
    #
    # optional arguments:
    #   -h, --help  show this help message and exit
    args = parser.parse_args()  # 一定要有这条传参才会起作用，或者-h才会起作用
    s = args.a + args.b
    print(s)
    return


# 不能有两个parser，参数接收都会被传入第一个parser中
def test3():
    parser1 = argparse.ArgumentParser()
    parser2 = argparse.ArgumentParser()
    parser1.add_argument('a', type=int, default=10, help="first para")
    parser2.add_argument('b', type=int, default=5, help='second para')
    args1 = parser1.parse_args()
    args2 = parser2.parse_args()
    s = args1.a + args2.b
    print(s)
    # 输入a和b的值报错：parser1只有一个参数，不能传入两个
    return


# name or flags - 一个命名或者列表，如'a' '-a' '--a'
def test1():
    #  生成一个参数解析器
    parser = argparse.ArgumentParser()

    # 通过add_argument方法增加参数，这里增加a和b两个参数
    # '-a'和'--a'是同一个参数（这是一种缩写形式，可以直接用-a或--a传入参数）
    # 前面带-或--的就是optional argument，否则就是positional argument（在命令行中必须传入）default为默认参数，
    # 一定要有default，否则没有传参时报错，传参不报错
    # 没有-或--一定要传参，default没用
    parser.add_argument('-y', '--year', type=int, default=10)
    parser.add_argument('b', type=int, default=5)
    # 如果用a和b时不传参，只传一个[-h]，为自带help参数
    # python arg_parse.py -h
    #     usage: argparse_test.py [-h] a b
    #     positional arguments:
    #     a
    #     b
    #     optional arguments:
    #     -h, --help  show this help message and exit
    args = parser.parse_args()
    s = args.a + args.b
    print(s)
    return


# action - 参数在命令行中出现时使用的动作基本类型
def test4():
    # 默认值 ‘store’：存储参数值
    # ‘store_const’：不输入存储None，输入-参数存储const keyword argument指定的值（类似-h）
    # ‘store_true’(store_false)：（只针对-和--）用参数传递布尔值，此时参数变为判断参数，类型为bool，不能改为其它类型（类似-h）
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', action='store_true')
    parser.add_argument('-b', action='store_const', const=10)
    args = parser.parse_args()
    print(args.a)  # 不传参：False，传-a：True
    print(args.b)  # 不传参：None，传-b：10


# type和default略
# metavar和dest不需要考虑

# namespace问题：当参数是-和--同时出现，namespace取--的名字而非-的名字
# 即调取参数时应为args.--后的名字，否则报错，而终端输入时二者都能选
def test5():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--name', default='A')
    parser.add_argument('-y', '--year', default='10')
    # python argparse_test.py -y 20 -n Chihao
    args = parser.parse_args()
    print(args.name)  # Chihao
    print(args.year)  # 20
    print(args.n)  # 'Namespace' object has no attribute 'n'


if __name__ == "__main__":
    # test2()
    # test3()
    # test1()
    # test4()
    test5()
    
```

#### 2  set_seed() 函数

种子为一个random数列的固定顺序

固定好种子，保证结果的可复现性

```python
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # 为CPU设置随机数种子
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置随机数种子
```

#### 3 os.environ

os.environ是一个环境变量字典，可以获取可环境变量有关的所有的键和其对应的值

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 固定要使用的GPU序号
```

#### 4 logger

logging模块用于输出运行日志，可以设置输出日志的等级（可以设定输出重要信息的等级）、日志保存路径、日志文件回滚等

这里只考虑类对象logger的输出用法

```python
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 默认level为warning级别，因此不会输出info级别内容
    
    logger = logging.getLogger(__name__)  # 创建logger
    
    # 日志级别由高到低是：fatal, error, warn, info, debug 低级别的会输出高级别的信息，高级别的不会输出低级别的
    logger.info("Start print log")
    logger.debug("Do something")  # 不输出
    logger.warning("Something maybe fail.")
    logger.info("Finish")

    """
    输出：
    2022-10-16 19:36:02,144 - __main__ - INFO - Start print log
    2022-10-16 19:36:02,144 - __main__ - WARNING - Something maybe fail.
    2022-10-16 19:36:02,144 - __main__ - INFO - Finish
    """
```

#### 5 Transfer Text to Dataset

```python
# 导入并储存文件
def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in
        # distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file

    # 读取cache开头的文件
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    # 读取成功
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    # 创建文件
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)  # 得到一组guid label text的字典集

        # tokenizer是对应模型的取词器
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        # 保存
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # 转化成tensor并建立数据集
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    # 我们使用的是classification 0或1
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
```

