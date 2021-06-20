# Products-Detection-and-Retrieval
自动售货机商品检测检索

* [模型权重下载(百度云)](https://pan.baidu.com/s/1JeicjkWVYDpyrTDGI7bKKA ) 提取码：1f7i

## 任务描述
对于自动售货机摄像头拍摄的静态数据，进行商品的检测，并按照图像检索的方式确定商品类别

<img src="https://github.com/ielym/Products-Detection-and-Retrieval/blob/main/imgs/1.png" height="500" />

## 阶段一

### 检测：
* Faster RCNN : resnext101_32x8d + ROIAlign
* objectness二分类，CIOU Loss
### 检索：
* CE Loss 预训练
* Triplet Loss, ArcFace 微调
* KNN, k=10, cosine distance
* 商品库图像数量平衡，提取特征平衡两种方案，防止KNN聚类的对于少量样本（商品库样本数量最少为2）的类别无法有效聚类。

<img src="https://github.com/ielym/Products-Detection-and-Retrieval/blob/main/imgs/7.jpg" width="100%" />

### 问题分析：
* 长尾分布

<img src="https://github.com/ielym/Products-Detection-and-Retrieval/blob/main/imgs/2.png" width="100%" />

* 遮挡问题

<img src="https://github.com/ielym/Products-Detection-and-Retrieval/blob/main/imgs/3.png" height="200" />

* 罐装，盒装，瓶装商品高宽差距较大。解决方案：KMean聚类

<p float="left">
  <img src="https://github.com/ielym/Products-Detection-and-Retrieval/blob/main/imgs/4.jpg" height="200" />
  <img src="https://github.com/ielym/Products-Detection-and-Retrieval/blob/main/imgs/5.jpg" height="200" />
</p>

<img src="https://github.com/ielym/Products-Detection-and-Retrieval/blob/main/imgs/6.png" height="300" />

* 光照影响明显

<p float="left">
  <img src="https://github.com/ielym/Products-Detection-and-Retrieval/blob/main/imgs/8.jpg" height="200" />
  <img src="https://github.com/ielym/Products-Detection-and-Retrieval/blob/main/imgs/9.jpg" height="200" />
</p>

