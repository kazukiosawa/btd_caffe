# btd_caffe

## BTD for CNN
- [Accelerating Convolutional Neural Networks for Mobile Applications](http://dl.acm.org/citation.cfm?id=2967280)
- 2016, ACM Multimedia
- Peisong Wang, 	Jian Cheng / 	Chinese Academy of Sciences & University of Chinese Academy of Sciences, Beijing, China
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

## Usage
```sh
$ ./start.sh
```
## start.sh に記述するパラメータ
| 名前 | 説明 |
| :-- | :-- |
|CAFFE_MODEL_ROOT | Caffeモデルを管理しているディレクトリ |
| MODEL_ROOT | オリジナルモデルのディレクトリ |
| ORIGINAL_DEPLOY | オリジナルモデル (deploy.prototxt)|
| ORIGINAL_MODEL | オリジナルモデル (caffemodel) |
| TEMPLATE_DEPLOY | BTD適用後のテンプレート (deploy.prototxt) |
| TEMPLATE_TRAIN_TEST | BTD適用後のテンプレート (train_test.prototxt) |
| LOWRANK_DEPLOY | 低ランクモデル (deploy.prototxt) |
| LOWRANK_TRAIN_TEST | 低ランクモデル (train_test.prototxt) |
| LOWRANK_MODEL | 低ランクモデル (caffemodel)|
| CONFIG | BTD用パラメータ設定ファイル|
