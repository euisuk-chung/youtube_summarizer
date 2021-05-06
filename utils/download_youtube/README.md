## youtube-summarization


### 사용법

```bash
python video_labeling.py --video_list_path {} --video_path {} --label_path {}
```

* 위 방법으로 video 다운로드 후 script 추출할 경우, naver cloud platform 활용하지 않아도 됨.
* 모든 video, script는 사용자가 지정한 local 경로에 저장됨.


### 주의사항
다운로드 및 스크립트 추출을 하고자 하는 유튜브 영상 링크는 video_list.txt파일에 명시하면 되나 아래와 같은 형식을 따를 것.<br>


```bash
그룹명
유튜브 영상 링크
그룹명
유튜브 영상 링크
...

example:
  KBS뉴스
  https://youtube.com/xxxxxx
  https://youtube.com/xxxxxy
  SBS뉴스
  https://youtube.com/xxxxyy
  https://youtube.com/xxxxzz
```
