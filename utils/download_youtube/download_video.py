## python -m pip install pytube
from pytube import YouTube
import os

def video_downloader(video_list_path, video_path):

    # video list to extract
    video_list = open(video_list_path,'r')

    while True:
        try:
            line = video_list.readline()
            if not line: break

            # 비디오 그룹명 지정
            if not line.startswith('http'):
                group_name = line.replace('\n', '')
                
            else:
                ## youtube homepage link
                yt = YouTube(line)
                file_ext = 'mp4'
                
                m, s = divmod(yt.length, 60)
                file_id = yt.embed_url.split('/')[-1]
                filename = f'{group_name}_{file_id}_{m}m_{s}s'

                if '.'.join((filename, file_ext)) in os.listdir(video_path):
                    print(f"{filename} already in the downloaded video list.")
                    continue
                else:
                    yt.streams.filter(progressive=True, 
                        file_extension= 'mp4').order_by('resolution').desc().first().download(video_path,
                                                                                             filename=filename)
                    print('[Video Downloaded] {}'.format(line))
                
        except Exception as E:
            print(f"Error occured: {E}")
            continue

    video_list.close()
    
    return