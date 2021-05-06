import os
import json
from pororo import Pororo
import IPython

def make_summary(data, summary_type, video_name):
    summary = Pororo(task = 'summarization', model = summary_type, lang = 'ko')
    result = summary(data)
    text = open('output/[{}]{}.txt'.format(summary_type[:3],video_name), 'w')
    text.write(result)
    text.close()


label_path = 'data/label'

label_list = os.listdir('data/label')

for i in range(len(label_list)):
    with open(os.path.join(label_path, label_list[i]), 'r') as f:
        label_json = json.load(f)

    print('Making Summary:', label_list[i])
    make_summary(label_json['text'], 'extractive', label_list[i])
