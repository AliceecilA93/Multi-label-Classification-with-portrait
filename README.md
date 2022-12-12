# Multi-label-Classification-with-portrait
 
## 진행기간 
- 2022.08.04 ~ 2022.08.19

## 목적
- **CNN-custom, MLP-custom, Resnet50, Alexnet 모델을 사용하여 인물사진으로 Multi-label Classification model 구축**  


## 코드 설명
   
코드     | 코드 링크   | 
:-------:|:-----------:|
이미지 전처리|[Image_preprocessing_portrait](https://github.com/AliceecilA93/Multi-label-Classification-with-portrait/blob/main/Source%20code/Image_preprocessing_portrait.ipynb)|
CNN-custom Model | [Convolutional Neural Networks](https://github.com/AliceecilA93/Multi-label-Classification-with-portrait/blob/main/Source%20code/Convolutional%20Neural%20Networks.ipynb)|
MLP-custom Model| [Multi-Layer Perceptron](https://github.com/AliceecilA93/Multi-label-Classification-with-portrait/blob/main/Source%20code/Multi-Layer%20Perceptron.ipynb)| 
Resnet50 Model| [Resnet50_hair](https://github.com/AliceecilA93/Multi-label-Classification-with-portrait/blob/main/Source%20code/Resnet50_hair.ipynb) |
Alextnet Model| [Alexnet_gender](https://github.com/AliceecilA93/Multi-label-Classification-with-portrait/blob/main/Source%20code/Alexnet_gender.ipynb)|          


## 사용된 데이터  

- AI HUB [페르소나 기반의 가상 인물 몽타주 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=618)


## 사용된 모델 

- CNN-custom
- MLP-custom
- Resnet50
- Alexnet


## 과정  

 1. 개발환경 : Python, Tensorflow, Colab
 
 2. 데이터 전처리
 
 * Image preprocessing
   
    - Image resize & padding
   
     ![image](https://user-images.githubusercontent.com/112064534/207093115-288d7ac3-84dc-4fea-94bf-2a0edebcfa62.png)

    - Image Augmentation 
   
   ![image](https://user-images.githubusercontent.com/112064534/207093296-7fbb5183-4b5a-4601-8887-b7593115a864.png)

    - Labeling
 ```c
def sex_label(sex):
    if sex == 'M':
        return 0
    else:
        return 1

def age_label(age):
  if 20 <= age < 30:
    return 0
  elif 30 <= age < 40:
    return 1
  elif 40 <= age < 50:
    return 2
  elif 50 <= age < 60:
    return 3
  else:
    return 4

def face_type_label(face_type):
  if face_type == '둥근형':
    return 0
  elif face_type == '긴형':
    return 1
  elif face_type == '사각형':
    return 2
  elif face_type == '계란형':
    return 3
  elif face_type == '역삼각형':
    return 4
  else:
    return 5
    
def hairstyle_label(hairstyle):
  if hairstyle == '직모(생머리)':
    return 0
  elif hairstyle == '스포츠형':
    return 1
  elif hairstyle == '곱슬머리':
    return 2
  elif hairstyle == '웨이브형':
    return 3
  else:
    return 4

```  
    
   - Zero-centering
   
   ![image](https://user-images.githubusercontent.com/112064534/207093226-72561924-cb73-4fea-894a-736d276f684b.png)

 
 3. 데이터셋
   
  ![image](https://user-images.githubusercontent.com/112064534/207093001-df342ff2-031a-4462-ba28-1f37b4870034.png)
    


 4. 데이터셋 분석
  
 * Age
  
   - 데이터 분포 
  
  ![image](https://user-images.githubusercontent.com/112064534/207094883-49e0f7bb-951c-40c4-8d8f-0dd4b7d43134.png)

   - Resnet50 시각화 
  
  ![image](https://user-images.githubusercontent.com/112064534/207095052-f755a9d5-180c-436c-a2a3-61cce2fff70f.png)

  
 * Gender
   
   - 데이터 분포 
   
   ![image](https://user-images.githubusercontent.com/112064534/207095259-c8b49596-d12a-445a-a725-22ff8f51e16d.png)

   - Alexnet 시각화 
   
   ![image](https://user-images.githubusercontent.com/112064534/207095364-e7ea5dd8-2036-4b62-bbd9-f192ffb7761d.png)

  
* Hairstyle
  
  - 데이터 분포 
 
  ![image](https://user-images.githubusercontent.com/112064534/207095467-555b551e-ac5c-4c4b-b3c4-175a65dd46d6.png)

  - Resnet50 시각화 
  
  ![image](https://user-images.githubusercontent.com/112064534/207095629-d0c71295-fc4f-4387-a480-59c033b6b59a.png)



 
 5. 성능개선 노력(Resnet50_hair) 
 
 * 데이터 핸들링 
 
   - 라벨 갯수 줄이기 ( 5 => 3) 
   - 증강 ( 5,446 => 15,000)
   - Zero-centering
  
 * 모델 핸들링 
 
   - Model 변경 ( CNN-custom, MLP-custom, Resnet50, Alexnet) 
   - Resnet50 모델의 개방 정도 조절 ( -10, -20) 
   - Resnet50 모델의 Dense 조절  ( Layers , Depth) 
   - Compile & Training ( Learning rate, Epoch, Batch size, callbacks) 
 

## 결과
- 인물 이미지로 Age, Gender, Hairstyle로 labeling해본 결과 최종적으로 Age는 Resnet50 Model, 
  Gender는 Alexnet Model, Hairstyle은 Resnet50 Model이 각 라벨당 Best Model로 선정되었다. 
  하지만 Age와 Gender는 성능이 좋은 반면 그에 비해 Hairstyle은 성능이 낮아서 Hairstyle Model의 성   능을 개선시키고자 데이터핸들링과 모델핸들링을 시도하였다. 그 결과 라벨 수를 줄이는 것보다 라벨 수   를 유지한 상태에서 데이터를 증강시키고 일부층 개방을 -10, Dense Layer는 2층보다는 한 층, Dense Layer 너비는 512보다 256으로 작게, Batch size는 50보다 125으로 넓게, Epoch수는 30보다는 15로 작게  하여 Val loss를 기준으로 callback함수를 썼을 때 성능을 0.02% 올릴 수 있었다. 모든 시도를 해봤음에도 불구하고 더 이상 수치가 올라가지 않아 Hairstyle을 구분해주기 위해서는 데이터셋을 변경하거나 다른 데이터를 넣어줘야할 것 같다. 
  


## 참조

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Communications of the ACM 60.6 (2017): 84-90.
