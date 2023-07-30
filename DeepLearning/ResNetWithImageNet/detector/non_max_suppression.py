import numpy as np
import cv2

def non_maxi_suppression(boxes,probs= None,overlapThresh =0.3):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == 'i':  #eğer geleb kutuların dtype'ı integer ise 
        boxes = boxes.astype('float')  #kutularımızı floata çevirelim

    x1 = boxes[:,0]  #tüm kutuların 0. indexi bana x1'input
    y1 = boxes[:,1] #tüm kutuların 1. indexi bana y1'input
    x2 = boxes[:,2] #tüm kutuların 2. indexi bana x2'input
    y2 = boxes[:,3] #tüm kutuların 3. indexi bana y2'input

    #alan bulalım 
    area = (x2-x1+1)*(y2-y1+1)

    idxs = y2

    #olasılık değerleri
    if probs is not None:  #eğer ki olasılık değerimiz boş değilse 
        idxs = probs
    idxs = np.argsort(idxs)  #olasılık değerlerine göre sıralıyoruz(indexi)
    
    pick = [] #secilen kutular

    while len(idxs) > 0:
        last = len(idxs) - 1 
        i = idxs[last]
        pick.append(i)
        #en buyuk ve en kucuk x ve y degerleri
        xx1 = np.maximum(x1[i],x1[idxs[:last]])
        yy1 = np.maximum(y1[i],y1[idxs[:last]])
        xx2 = np.minimum(x2[i],x2[idxs[:last]])
        yy2 = np.minimum(y2[i],y2[idxs[:last]])

        #width ve height 
        w = np.maximum(0,xx2-xx1 +1)
        h = np.maximum(0,yy2-yy1 +1)

        #overlap yani IoU
        overlap = (w*h)/area[idxs[:last]]

        idxs =np.delete(idxs,np.concatenate(([last],np.where(overlap > overlapThresh)[0])))   #where ile şartı sağlayan indexi buluyoruz eskisiyle birlleştiriyoruz ve siliyoruz
    return boxes[pick].astype('int')
