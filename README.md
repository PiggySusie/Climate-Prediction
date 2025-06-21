# 基于深度学习模型的气象预测与模型优化

### 项目概述

本项目的核心目标是优化天气预测模型，使用深度学习方法进行时序预测，尤其专注于不同模型在捕捉天气数据时序特征和周期性规律的表现。为了实现这一目标，我们使用了多种深度学习架构，如LSTM、ResNet和Transformer，并针对时序数据的预测任务进行了创新性优化，尤其是动态模型选择和模型融合策略。

### 文档材料

 [实验报告](基于深度学习模型的气象预测与模型优化_实验报告.md) 

 [技术文档](基于深度学习模型的气象预测与模型优化_技术文档.md) 



### 测试方法

#### 1. Meta-Learner测试

这部分内容采用Colab+GoogleDrive实现，在notebook最后提供了对最后一个混合模型的测试实现。

完整的Notebook、所需模型和精简版本已上传至 [Meta-Learner](https://github.com/PiggySusie/Climate-Prediction/tree/master/Meta-Learner) 

[模型](https://github.com/PiggySusie/Climate-Prediction/tree/master/Meta-Learner/model)/[精简数据](https://github.com/PiggySusie/Climate-Prediction/tree/master/Meta-Learner/data)需要下载到本地后上传colab进行测试，在Notebook的最后一部分，提供了相应的测试方式，直接按照指示运行即可，测试所用函数都是前文出现过的，为了方便测试统一复制到了最后一部分。



#### 2. ConvLSTM-PM2.5测试

完整的代码和数据已上传至 [ConvLSTM-Pm2.5](https://github.com/PiggySusie/Climate-Prediction/tree/master/ConvLSTM-Pm2.5) 

在[test/](https://github.com/PiggySusie/Climate-Prediction/tree/master/test)文件夹中提供了小样本的测试，在数据集中选取了10天的数据。建议在新环境安装`requirements.txt`中的依赖后，运行`predicttime.py`(与原测试只有路径区别)，可预测所选日期的pm2.5值，可选日期范围20130301-20130310。

另外，在可视化网站中也可以输入时间点进行预测，逻辑相同。



#### 3. pangu模型测试

完整的代码和部分数据已上传至 [Pangu-Weather](https://github.com/PiggySusie/Climate-Prediction/tree/master/Pangu-Weather) 

盘古模型在项目中只做了本地部署，具体参考了开源仓库[Pangu-Weather-ReadyToGo](https://github.com/HaxyMoly/Pangu-Weather-ReadyToGo), 没用专门做测试，可视化部分保留了部分文件，具体见第四条。



#### 4. 可视化成果展示

包含已经精简过的数据的可视化，运行`web/pm2.5/app.py`后打开`http://localhost:5000`，即可进行可视化验证，可能需要下载新的依赖，文件夹中有对应的`requirements.txt`。

