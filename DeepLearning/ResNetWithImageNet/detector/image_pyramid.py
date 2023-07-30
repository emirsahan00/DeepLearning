import cv2
import matplotlib.pyplot as plt

def image_pyramid(image,scale = 1.5,minSize = (33,33)):  #algoritmamıza resim üretmek için kullanacağız
    yield image  #bellekte çok yer kaplamaması adına
    
    while True: #sonsuz bir döngü oluşturuyoruz
        w = int(image.shape[1]/scale) #resmin widthini scale değerine bölüyoruz
        image = cv2.resize(image,dsize=(w,w)) #ve w x w boyutlarında yeninden boyutlandırıyoruz 
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:  # yükseklik 33'dan küçükse ya da genişlik 33'dan küçükse döngüyü kır 
            break
        yield image 


#img = cv2.imread('src/ApeIron.png') #resmi okuyoruz
#im = image_pyramid(img)  #okuduğumuz resmi image_pyramid fonksiyonua yolluyoruz 
# for i,image in enumerate(im):
#     if i == 7:
#         plt.imshow(image)
#     #print(i)

