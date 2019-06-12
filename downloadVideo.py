from pytube import YouTube
yt=YouTube('https://www.youtube.com/watch?v=WEeRMh_L5NE')
yt.streams\
  .filter(progressive=True, file_extension='mp4')\
  .order_by('resolution')\
  .desc()\
  .first()\
  .download()

print('동영상 다운로드 완료')
