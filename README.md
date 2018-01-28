## Image deblocking using Convolutional Neural Networks  

### Experiment Result：  

model | PSNR|SSIM|
---|---|---|
input(Q10) | 25.5151|0.7468|
DenseNet | 19.4077(-6.1067)|0.7222(-0.024)
DenseNet_shallow |20.6849(-4.8302)|0.7327(-0.014)
ARCNN|22.5322|0.7618
L8|22.8510|0.7674

### problem1:  
&emsp;&emsp;虽然输出图像的块效应有改善，但是却产生了噪声，尤其是在白色区域，更多更明显:  
RGB格式下：  
![image](https://github.com/yydlmzyz/Image-deblocking-using-CNN/blob/master/images/output3_L8_rgb/344010.jpg)  
YCbCr格式下：  
![image](https://github.com/yydlmzyz/Image-deblocking-using-CNN/blob/master/images/output4_L8_YCbCr/344010.jpg)  
&emsp;&emsp;有待解决。  
  
### problem2:  
&emsp;&emsp;DenseNet有问题，训练慢，而且效果差，准备放弃。经过对比实验，增加网络深度没效果，增加channels的数量有效果。  
  
### plan:  
&emsp;&emsp;尝试其他网络模型和多尺度

### References：  

1.[Compression Artifacts Removal Using Convolutional Neural Networks](https://arxiv.org/abs/1605.00366)  
2.[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802v1)  
3.[Compression Artifacts Reduction by a Deep Convolutional Network](https://arxiv.org/abs/1504.06993)  
4.[CAS-CNN: A Deep Convolutional Neural Network for Image Compression Artifact Suppression](https://arxiv.org/abs/1611.07233)  
5.[DataSet](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)


