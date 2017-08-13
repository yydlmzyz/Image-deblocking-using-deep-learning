# deblock
使用CNN对压缩图像进行优化，包括去除块效应等


参考论文：
https://arxiv.org/abs/1609.04802v1
https://www.computer.org/csdl/proceedings/iccv/2015/8391/00/8391a576-abs.html
参考代码：
https://github.com/qobilidop/srcnn
https://github.com/shreka116/SRResNet
https://github.com/titu1994/Super-Resolution-using-Generative-Adversarial-Networks
https://github.com/tonitick/AR-CNN


使用方法：
第一步：Data.py
将作为Label和Data的image分别存放在两个文件夹中，顺序要对应
通过Data.py生成h5格式的训练数据data.h5

第二步：Train.py
运行Train.py，会调用MyModel.py中定义的模型对data.h5进行训练，
会生成weights的checkpoints文件、记录文件history.csv、最终输出训练好的模型Model.h5

第三步：Predict.py&Show.py
通过Predict.py，调用参数文件weights.h5,可以将图片集成批处理，输出优化后的图片集
通过Show.py可以将一对Labe.jpg&Data.jpg进行处理，并比较结果，进行展示

其它：
第一个模型结构及实验效果在文件夹MyModel_version1中，在MyModel_ARCNN中存放了一些其他模型作为比较补充



