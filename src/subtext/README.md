## 사용법

### (1) 데이터셋 생성
```bash
'''
Arguments
    dataset_basedir: 데이터셋을 모아놓는 베이스 디렉토리 (default: path/to/your/repository/dataset)
    config_path: BertSum argument 파일 (config.yaml 파일 위치, default: ./config.yaml)
    window_size: 윈도우 크기 (default: 4)
    dataset_size: 데이터 크기 (default: 50000)
    random_point: 명시하지 않는 경우(False) 기사의 [첫번째(0) ~ window_size]를 샘플링
                  명시하는 경우(True) 기사의 위치 중 랜덤하게 선택하여 window_size 만큼 샘플링
                  (default: False)
'''
python make_data.py \
    --dataset_basedir {} \
    --config_path {} \
    --window_size {} \
    --dataset_size {} \
    --random_point
```

### (2) 학습
```bash
'''
Arguments
    data_path: 학습을 위한 텐서 리스트 위치 (default: path/to/your/repository/dataset/subtext_dataset)
    save_path: 모델 저장 위치 (default: ./ckpt/)
    window_size: 윈도우 크기 (default: 4)
    random_point: 이용하는 데이터가 랜덤 포인트로 생성이 되었는지 여부
                  명시하지 않을 경우 False
                  명시하지 할 경우 True
'''
python train.py \
    --data_path {} \
    --save_path {} \
    --window_size {} \
    --random_point
```