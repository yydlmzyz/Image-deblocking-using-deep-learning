# deblocking
使用神经网络对压缩图像进行优化，包括去除块效应等


References：
https://arxiv.org/abs/1609.04802v1
http://de.arxiv.org/pdf/1504.06993
https://arxiv.org/abs/1605.00366
https://arxiv.org/pdf/1611.07233.pdf

https://github.com/qobilidop/srcnn
https://github.com/shreka116/SRResNet
https://github.com/titu1994/Super-Resolution-using-Generative-Adversarial-Networks
https://github.com/tonitick/AR-CNN

DataSet:http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz

1st step：DataProcess.py
将作为Label和Data的image分别存放在两个文件夹中，顺序要对应
通过Data.py生成h5格式的训练数据data.h5


2nd step：Train_all.py/Train_batch.py 
Train_all.py需要一次调入全部数据到内存中，Train_batch每次只调入1个batch的数据，当内存小于数据量时，只能用Train_batch.py
根据需要可以使用ARCNN、L8、SRResNet_simple、SRResNet_complex中的model，SRResNet中的model没有测试过，还需要修改
训练过程中会生成weights.h5


3rd step：Show_result.py&Show_layers.py&Predict.py
Show_result.py需要输入一对label-input，进行预测，并比较结果
Show_layers.py 可以展示每一层每一个channel的输出
Predict.py可以将图片集成批处理,课用于视频帧的预测


Models：
在文件夹ARCNN、L8、SRResNet_complex、SRResNet_simple中分别为不同模型的代码、结构图、权重文件和效果展示
utilities：
包含一些用Matlab压缩图片，从视频中截取图像等功能的代码


