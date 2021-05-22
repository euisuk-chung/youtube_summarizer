## 사용법

### (1) 데이터셋 생성
```bash
'''
Arguments
    1. dataset_basedir: 데이터셋을 모아놓는 베이스 디렉토리\
    (default: /path/to/your/repository/dataset)
    2. config_path: BertSum argument 파일\
    (config.yaml 파일 위치, default: ./config.yaml)
    3. window_size: 윈도우 크기 (default: 4)
    4. dataset_size: 데이터 크기 (default: 50000)
    5. random_point: 랜덤 여부 (default: False) \
    명시하지 않는 경우(False) \
    기사의 [첫번째(0) ~ window_size]를 샘플링 명시하는 경우(True) 기사의 위치 중 랜덤하게 선택하여 window_size 만큼 샘플링
    6. embed_type: 어떤 embedding값을 사용할 것인가 (default : bert)\
    bert : Kobert Embedding\
    word : FastText Embedding
'''
python make_data.py \
    --dataset_basedir {} \
    --config_path {} \
    --window_size {} \
    --dataset_size {} \
    --embed_type {} \
    --random_point
```

### (2) 학습
```bash
'''
Arguments
    1. data_path: 학습을 위한 텐서 리스트 위치 (default: path/to/your/repository/dataset/subtext_dataset)
    2. save_path: 모델 저장 위치 (default: ./ckpt/)
    3. window_size: 윈도우 크기 (default: 4)
    4. random_point: 이용하는 데이터가 랜덤 포인트로 생성이 되었는지 여부
                  명시하지 않을 경우 False
                  명시하지 할 경우 True
'''
python train.py \
    --data_path {} \
    --save_path {} \
    --window_size {} \
    --random_point
```