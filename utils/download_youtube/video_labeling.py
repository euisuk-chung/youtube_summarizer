import os
import argparse
from download_video import video_downloader
from clovaspeech_script import ClovaSpeechClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_list_path', default = './video_list.txt')
    parser.add_argument('--video_path', default='/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/dataset/youtube_dataset/video')
    parser.add_argument('--label_path', default='/home/sks/korea_univ/21_1/TA/team_project/youtube_summarizer/dataset/youtube_dataset/label')
    
    args = parser.parse_args()
    
    video_downloader(args.video_list_path, args.video_path)
    print('Downloading videos done.')
    
    video_list = os.listdir(args.video_path)
    for i in video_list:
        video_path = os.path.join(args.video_path, i)

        if i.replace('mp4','txt') in os.listdir(args.label_path): 
            print('Audio Script has been already extracted')
            pass

        else:
            res = ClovaSpeechClient().req_upload(file=video_path, completion='sync')

            with open(os.path.join(args.label_path, i.replace('mp4','txt')), 'w') as script:
                script.write(res.text)
            script.close()
            print('[Audio Extraction Finished] {}'.format(i))


if __name__=='__main__':
    main()

