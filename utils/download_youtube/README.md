## youtube-summarization


### [Step1] Download Youtube Video to Local

```python
python download_video.py --video_list_path {} --video_path {}
```

### [Step2] Extract video script via Clova Speech API

```python
python clovaspeech_script.py --video_path {} --label_path {}
```

* 위 방법으로 video 다운로드 후 script 추출할 경우, naver cloud platform 활용하지 않아도 됨.
* 모든 video, script는 사용자가 지정한 local 경로에 저장됨. 