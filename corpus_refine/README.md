### simple_denoiser.py
简单的语料去噪文件，主要包含两个函数。
第一个是直接对于给定类别和概率进行降采样
第二个函数是基于关键词进行去噪

### biclassifier_denoiser.py
使用二分类器进行去噪。
指定正样本和和负样本，利用fasttext训练二分类器，
用二分类器从目标样本中去除正样本噪声

### cross_validation_denoiser.py
主要用于两个类别相互混淆的情况，例如health&baby
1.分别筛选出正样本和负样本。
2. 分别将正负样本等分成10份。
3. 每次选取一份样本作为测试集，剩下的9份样本作为训练集。
4. 利用训练好的模型对测试集进行去噪处理。例如测试集的标签是baby但是分类器
预测的是health且概率大于阈值，则认为该条样本是噪声，应该删除

### PR_analysis.py
用于precision和recall的分析
recall_analysis()用于分析召回的样本中，其他类别样本的数量
precision_analysis()，对于给定类别，分析该类样本都错分到哪些其他类样本上了


## 语料迭代记录
https://docs.qq.com/sheet/BqI21X2yZIht133T2f06wNPX3leh8C0oEhyy1adosE0DS2Dh1IQmKC2Cjyb92803mC441LBH2KA1Cy26ewVc1#BB08J2
