![](./imgset/img1.png)
# Data mining
>Homework 2: Clustering with sklearn 

>数据班 赵鑫鉴  201700181053
## Requirements
+ python==3.7.4
+ textblob==0.15.3
+ scikit-learn ==0.21.3（0.20.3以下版本可能会报错）
+ NumPy (>= 1.11.0)
+ SciPy (>= 0.17.0)
+ joblib (>= 0.11)
&emsp;运行此代码需要安装sklearn，可以参考以下几种下载或者更新方式：
```
使用清华镜像下载：
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn
更新：
pip install --upgrade scikit-learn
conda update scikit-learn
pip install -U sklearn
```
##Dataset：
本次作业中使用两个sklearn内置的数据集：

1.20newsgroups：

&emsp;该数据集是用于文本分类、文本挖据和信息检索研究的国际标准数据集之一。数据集收集了大约20,000左右的新闻组文档，均匀分为20个不同主题的新闻组集合。一些新闻组的主题特别相似(e.g. comp.sys.ibm.pc.hardware/ comp.sys.mac.hardware)，还有一些却完全不相关 (e.g misc.forsale /soc.religion.christian)。

2.digits：

这个数据集中并没有图片，而是经过提取得到的手写数字特征和标记，就免去了我们的提取数据的麻烦，但是在实际的应用中是需要我们对图片中的数据进行提取的。

digits导入使用示例：
```
from sklearn.datasets import load_digits
digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target
```


##实验要求：
在sklearn内置的数据集中评测不同聚类算法的聚类效果，使用以下三种评价指标和算法时间复杂度评测聚类效果：
![](./imgset/im12.png)

![](./imgset/img2.png)
![](./imgset/img3.png)
![](./imgset/img4.png)
![](./imgset/img5.png)
![](./imgset/img7.png)
![](./imgset/img6.png)
![](./imgset/img9.png)
![](./imgset/img8.png)
![](./imgset/img11.png)
![](./imgset/img10.png)










