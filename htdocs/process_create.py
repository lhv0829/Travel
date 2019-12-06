#!//anaconda3/bin/python3
print("content-type:text/html;")
print()
import cgi
from scipy.spatial import distance as dist
import numpy as np
import cv2
import pafy
from skimage.measure import compare_ssim as ssim
import os
import scipy.stats
import numpy as np
import tensorflow.compat.v1 as tf
import multiprocessing
from multiprocessing import Process

form = cgi.FieldStorage()
url = form["link"].value

def mse(imageA, imageB):
    imageA=cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB=cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    err=np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    return err

def hist_bhattacharyya(imageA, imageB):
    hsv1=cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
    hsv2=cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
    
    hist1=cv2.calcHist([hsv1], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    hist2=cv2.calcHist([hsv2],[0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    
    value=cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    return value

def delete(path):
    for root, dirs, files in os.walk(path):
        for currentFile in files:
            exts = ('.jpg')
            if currentFile.lower().endswith(exts):
                os.remove(os.path.join(root, currentFile))

delete('/Applications/mampstack-7.3.11-0/apache2/htdocs')

vPafy = pafy.new(url)
play = vPafy.getbest(preftype="webm")
cap = cv2.VideoCapture(play.url)

count = 0
thresh_hist = 0.1
thresh_bha = 0.5
thresh_mse = 2000
thresh_ssim = 0.45

ret, prev_frame = cap.read()
cv2.imwrite('frame%d.jpg'%count,prev_frame);
count = 1

while ret:
    ret, curr_frame = cap.read()

    hist_prev = cv2.calcHist([prev_frame], [0], None, [256], [0,256])
    hist_prev = cv2.normalize(hist_prev, hist_prev).flatten()
    hist = cv2.calcHist([curr_frame], [0], None, [256], [0,256])
    hist = cv2.normalize(hist, hist).flatten()
    
    if ret:
        d = dist.chebyshev(hist_prev, hist)
        if d > thresh_hist:
            value_b=hist_bhattacharyya(prev_frame, curr_frame)
            if value_b>thresh_bha:
                error=mse(prev_frame,curr_frame)
                if error>thresh_mse:
                    grayP=cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    grayC=cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                    (score, diff)=ssim(grayP, grayC, full=True)
                    if score<thresh_ssim:
                        cv2.imwrite('frame%d.jpg'%count,curr_frame);
                        count += 1
                        prev_frame = curr_frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#test
modelFullPath = '/tmp/output_graph.pb'                                      # 읽어들일 graph 파일 경로
labelsFullPath = '/tmp/output_labels.txt'                                   # 읽어들일 labels 파일 경로
#attractions = []


def create_graph():
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        
create_graph()
f = open(labelsFullPath, 'rb')
lines = f.readlines()
labels = [str(w).replace("\n", "") for w in lines]

def run_inference_on_image(imagePath):
    answer = None

    #if not tf.gfile.Exists(imagePath):
    #    tf.logging.fatal('File does not exist %s', imagePath)
    #    return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    #create_graph()
    #lock.acquire()
    with tf.Session() as sess:
               softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
               predictions = sess.run(softmax_tensor,
                                      {'DecodeJpeg/contents:0': image_data})
               predictions = np.squeeze(predictions)
               top_k = predictions.argsort()[-1:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
               #f = open(labelsFullPath, 'rb')
               #lines = f.readlines()
               #labels = [str(w).replace("\n", "") for w in lines]
               if predictions[top_k[0]] > 0.2:
                   answer = labels[top_k[0]]
                   print('''<ul><h2><li>%s</li></h2></ul>''' % answer[2:-3])

#def printlist(attractions):
#    print('''<h1><strong>Tourist attractions in this video...</strong></h1>''')
#    if not attractions:
#        print('''<h2>Sorry, we can't find any tourist attractions in this video.</h2>''')
#    else:
#        for i in attractions:
#            print('''<ul><h2><li>%s</li></h2></ul>''' % i)

print('''<!doctype html>
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="author" content="colorlib.com">
    <link href="https://fonts.googleapis.com/css?family=Poppins:400,500,700" rel="stylesheet" />
    <link href="css/main2.css" rel="stylesheet" />
  </head>
  <body>
    <div class="s013">
      <div id="side_left">
        <iframe width="550" height="315" src="
''')
print(url[:24]+"embed/"+url[32:])
print('''" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
      </div>
      <div id="side_right">
      <h1><strong>Tourist attractions in this video…</strong></h1>
''')

if __name__ == '__main__':
    procs = []
    for i in range(count):
        proc = Process(target=run_inference_on_image, args=('frame'+str(i)+'.jpg',))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()

print('''
      </div>
    </div>
  </body>
</html>
''')

print()