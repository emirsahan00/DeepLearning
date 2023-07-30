import cv2  #gerekli küpühaneleri tanımlıyoruz
import matplotlib.pyplot as plt

def sliding_window(image,step,ws): #step dikdörtgenin resim üzerinde kaç piksel atlayarak dolaşacağı / ws ise dikdörtgen boyutu
    for y in range(0,image.shape[0]-ws[1],step): 
        for x in range(0,image.shape[1]-ws[0],step):
            yield (x,y,image[y:y+ws[1],x:x+ws[0]])  #belleği yormamak adına yield ile generate ediyoruz

# img =cv2.imread('src/ApeIron.png')
# im = sliding_window(img,12,(200,150))

# for i,image in enumerate(im):
#     print(i)
#     if i == 1200:
#         plt.imshow(image[2])
        
