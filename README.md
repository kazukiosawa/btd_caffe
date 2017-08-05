# btd_caffe

## Block Term Decomposition (BTD) for CNNs
- [Accelerating Convolutional Neural Networks for Mobile Applications](http://dl.acm.org/citation.cfm?id=2967280)
- 2016, ACM Multimedia
- Peisong Wang and Jian Cheng / Chinese Academy of Sciences & University of Chinese Academy of Sciences, Beijing, China
- Parameters for '3.2 Whole-Model Acceleration for VGG-16’ in ‘3. EXPERIMENTS’
  >The S', T' and R for conv1_2 to conv5_3 are as follows:  
  >conv1_2: 11, 18, 1  
  >conv2_1: 10, 24, 1  
  >conv2_2: 28, 28, 2  
  >conv3_1: 36, 48, 4  
  >conv3_2: 60, 48, 4  
  >conv3_3: 64, 56, 4  
  >conv4_1: 64, 100, 4  
  >conv4_2: 116, 100, 4  
  >conv4_3: 132, 132, 4  
  >conv5_1: 224, 224, 4  
  >conv5_2: 224, 224, 4  
  >conv5_3: 224, 224, 4  
- 'group' parameter is used in 'convolution_param' in Caffe network definition (.prototxt)
  - description of 'group' in [Caffe Tutorial for Convolution Layer](http://caffe.berkeleyvision.org/tutorial/layers/convolution.html)
  >group (g) [default 1]: If g > 1, we restrict the connectivity of each filter to a subset of the input. Specifically, the input and output channels are separated into g groups, and the iith output group channels will be only connected to the iith input group channels.

## Usage
```sh
$ ./start.sh
```
## start.sh に記述するパラメータ
| 名前 | 説明 |
| :-- | :-- |
| ORIGINAL_DEPLOY | オリジナルモデル (deploy.prototxt)|
| ORIGINAL_MODEL | オリジナルモデル (.caffemodel) |
| TEMPLATE_DEPLOY | 低ランクモデルのテンプレート (deploy.prototxt) |
| TEMPLATE_TRAIN_TEST | 低ランクモデルのテンプレート (train_test.prototxt) |
| LOWRANK_DEPLOY | 低ランクモデル (deploy.prototxt) |
| LOWRANK_TRAIN_TEST | 低ランクモデル (train_test.prototxt) |
| LOWRANK_MODEL | 低ランクモデル (.caffemodel)|
| CONFIG | BTD用パラメータ設定ファイル (.csv)|

## BTD用パラメータ設定ファイル（.csv）
```
conv, S', T', R
```
- conv : prototxt内の"Convolution"layerの名前
- S' : 入力チャネル数
- T' : 出力チャネル数
- R  : ブロック数
