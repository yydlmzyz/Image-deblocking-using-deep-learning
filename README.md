## Image deblocking using Convolutional Neural Networks  

### Experiment Result：  

model | PSNR|SSIM|
---|---|---|
input(Q10) | 25.6128|0.7394|
DenseNet | 25.7880(+0.1752)|0.7661(0.0267)
DenseNet_shallow |26.2750(+0.6597)|0.7650(+0.0256)
ARCNN|26.7920(+1.1792)|0.7774(+0.0380)
L8|26.8150(+1.2202)|0.7797(+0.0402)


### problem1:  
&emsp;&emsp;DenseNet有问题，训练慢，而且效果差。经过对比实验，增加网络深度没效果，增加channels的数量有效果。  
  

### References：  

1.[Compression Artifacts Removal Using Convolutional Neural Networks](https://arxiv.org/abs/1605.00366)  
2.[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802v1)  
3.[Compression Artifacts Reduction by a Deep Convolutional Network](https://arxiv.org/abs/1504.06993)  
4.[CAS-CNN: A Deep Convolutional Neural Network for Image Compression Artifact Suppression](https://arxiv.org/abs/1611.07233)  
5.[DataSet](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)


