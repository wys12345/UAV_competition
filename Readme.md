## 说明文档

1.按照官方程序处理数据集（testA testB均处理成5个模态）

2.修改feeder文件（比赛提供为npy数据集需要读取npy数据而不是npz数据）

3.修改config中的文件 通过config文件修改超参数 （运行命令 python main.py  config/xxxx.yaml）

总共用了1个Transformer模型（Mix_Former文件） 3个GCN(Mix_GCN文件)（CTR、BlockGCN、TDGCN）model文件夹下保存

4.output系列文件夹为模型训练保存日志及权重的路径  MixGCN中 output-c对应CTR 

5.通过MixGCN/E2.py 文件找最好的权重系数，并将权重系数保存

6.将feeder文件中的test路径修改为不同模态的test_B，然后在main.py  def eval的output = self.model(data)后面添加np.save('xxx.npy', output.cpu().numpy()) 保存不同模态的最好置信度

7.文件夹M C B T为4个模型在test_B上的置信度

8.通过MixGCN/megre.py 实现权重系数乘以不同模态的置信度并求和

在test_A的test的融合结果如下

Maximum accuracy: 77.7500%
Optimal weights: [1.2, 1.2, 1.2, 0.2, 1.2, 1.2, 0.8086936792522563, 0.2, 0.2, 0.2, 1.2, 1.2, 0.2, 0.2, 0.5231386815356827, 0.44442874390598386, 0.8679991412973587, 0.2, 0.2, 1.2]

Github只提供权重和训练日志，训练完整记录：https://drive.google.com/file/d/13u2xxLm3Kkr3jBfbP-L0Da1Noc60nRv1/view?usp=sharing