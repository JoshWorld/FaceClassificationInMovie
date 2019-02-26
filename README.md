## FaceClassificationInMovie
#### 영화에서 다중 객체 중 특정 객체 얼굴 인식 및 분류
#### Pre-Training Model을 이용하여 다중 객체 중 얼굴을 인식하여 자동으로 분류 한 후, 학습시킨 후 얼굴을 분류하는 프로젝트
아래 논문과 오픈소스코드를 참고하였습니다.
- Google Facenet 논문 (https://arxiv.org/abs/1503.03832)
- Tensorflow Face Detection Model (https://github.com/yeephycho/tensorflow-face-detection)
- Keras Openface (https://github.com/iwantooxxoox/Keras-OpenFace)


## 과정
1. Face Detection (Using Tensorflow face detection model)
2. Face Classification (Using facenet model and algorithm)
3. Training Face (Using SVM)

## 결과

## 모듈 구성
- backup : 작업하면서 만들었던 과거의 코드들
- keras_version : keras 로 구현한 CNN
- tensorflow_version : tensorflow로 구현한 CNN
- labels : object detection 에 필요한 label
- models : object detection 과 facenet 에 필요한 pre-training model
- test_data : test data 모음
- report : 발표 PPT
- face_detection.py : face detection model 을 이용한 face detection 모듈
- person_detection.py : object detection model 을 이용한 person detection 모듈
- beta_20190221.py : distance weight, algorithm 이 적용된 모듈

## 설치 방법
1. python 3.6 설치
2. Tensorflow, Keras 설치 (설치문서 참고)
3. pip install -r requerments.txt 로 필요 라이브러리 설치
4. Tensorflow Object Detection API 설치 (설치문서 참고)

## 사용 방법