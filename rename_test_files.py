import os
from PIL import Image

for flower in os.listdir('./test/'):
    count = 0
    for file in os.listdir('./test/' + flower):
        print('./test/'+flower+'/'+flower+str(count)+'.jpg')
        try:
            img = Image.open('./test/'+flower+'/'+file)
            img.save('./test/'+flower+'/'+flower+str(count)+'.jpg')
            os.remove('./test/'+flower+'/'+file)
            count += 1
        except:
            pass