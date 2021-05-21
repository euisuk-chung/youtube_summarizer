# Dataset Directory

### Components
article_dataset: 기사 데이터 (BertSum 학습을 위한 데이터)<br>
subtext_dataset: subtext 알고리즘을 학습하기 위한 여타 데이터<br>
youtube_dataset: 다운로드한 유튜브 mp4 파일 및 Pororo를 사용한 스크립트 데이터<br>


```bash
└── dataset
    ├── README.md
    ├── article_dataset
    │   ├── dev.jsonl
    │   ├── test.jsonl
    │   └── train.jsonl
    ├── preprocessed
    │   └── processed_wiki_ko.txt
    ├── subtext_dataset
    │   ├── nn_dataset_w1_fixed.pkl
    │   ├── nn_dataset_w2_fixed.pkl
    │   ├── nn_dataset_w2_random.pkl
    │   ├── nn_dataset_w2v_w3_fixed.pkl
    │   ├── nn_dataset_w3_fixed.pkl
    │   ├── nn_dataset_w3_random.pkl
    │   ├── nn_dataset_w4_fixed.pkl
    │   └── nn_dataset_w4_random.pkl
    ├── tokenized_dataset
    │   ├── tokenized_news_data.pkl
    │   ├── tokenized_news_data_NEW.pkl
    │   ├── tokenized_wiki_data.pkl
    │   └── tokenized_wiki_data_NEW.pkl
    └── youtube_dataset


``` 
