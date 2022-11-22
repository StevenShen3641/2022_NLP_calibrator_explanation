### Problems met during installing process and solutions

#### Xshell

1、登录时登录名和服务器名相同，都为yyl。

![](E:\沈驰皓\教材\需要的教材和课程\NLP\report\files\2_1.png)

2、配置环境时（如安装扩展包）使用conda可能导致pycharm及其它扩展必须升级或降级版本。如果希望其它内容版本不变，可以使用pip安装。直接pip install可能导致报错，可以使用国内镜像网站，如pip install opencv-python  -i https://pypi.tuna.tsinghua.edu.cn/simple，即可成功安装

3、由于root的位置是/home/yyl，因此collaborator在当前级

#### Pycharm

1、Pycharm要使用专业版，不能使用Community版本

**2、配置Pycharm添加虚拟环境时遇到的问题：**

定位Deployment Path时，2002.2.2版本每次打开Project就会重新在tmp文件夹中建立临时path，且会重新建立新的Deployment Connection

原因：没有设置Root Path导致每次打开Project，Deployment Path就会重置。

因此在Add New Interpreter on SSH后应立即打开Configuration

![](E:\沈驰皓\教材\需要的教材和课程\NLP\report\files\2_2.png)

Connection下，手动输入Root Path或者Autodetect

![](E:\沈驰皓\教材\需要的教材和课程\NLP\report\files\2_3.png)

在Mapping中，更改Deployment Path，删除该栏中的root path

如：/home/yyl/collaborator/Chihao/Demo1应改为/collaborator/Chihao/Demo1

![](E:\沈驰皓\教材\需要的教材和课程\NLP\report\files\2_4.png)

退出后重新打开Project，再打开Deployment时就已经不会建立临时路径了。