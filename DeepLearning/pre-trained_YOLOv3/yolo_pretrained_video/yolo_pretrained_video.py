import cv2
import numpy as np

cap = cv2.VideoCapture('D:/ImageProcessing/YOLOv4/yolo_pretrained_video/src/people.mp4')  #videomuzu okuyourz

while True: #frameleri okumak için sonsuz bir döngü başlatıyoruz
    ret,frame = cap.read() #frameleri okuyoruz
    
    frame_height = frame.shape[0] #her bir frame'in genişlik ve yükseklik bilgilerini alıyoruz
    frame_width = frame.shape[1] 

    frame_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB= True,crop=False) #frameleri modele aktarabilmemiz için blob formatına(4 boyut) çevirmemiz gerekiyor (frame,en optimal değer,model hangi boyutta resimlerle eğitildiyse,bgr -->rgb,frameleri kırpmasını istemiyoruz)
    labels = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"]

    colors = ["0,0,255","0,255,0","255,0,0","255,234,50","0,233,100"] #hangi renklerde bounding box olmasını istiyorsak o rengin bgr karşılığını giriyoruz
    colors = [np.array(color.split(',')).astype('int') for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(18,1)) #diziyi büyütüyoruz

    model = cv2.dnn.readNetFromDarknet("D:/ImageProcessing/YOLOv4/pretrained_model/yolov3.cfg","D:/ImageProcessing/YOLOv4/pretrained_model/yolov3.weights") #modelimizi içe aktardık
    layers = model.getLayerNames() #bütün layer(katmanları) çekiyoruz

    output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers()] #model.getUnconnectedOutLayers() bu bize çıktı katmanlarımızın indexini döndürüyor ama bir fazlasını
    model.setInput(frame_blob)

    detections_layers = model.forward(output_layer) #çıktı katmanının içindeki değerlere eriştik


    #### NON-MAXIMUM SUPPRESSION - OPERATION 1 ####
    ids_list = []
    boxes_list = []
    confidence_list = []
    #### END OF OPERATION 1 ####

    for detection_layer in detections_layers:
        for object_deteciton in detection_layer:
            scores = object_deteciton[5:]  #ilk 5 değer bounding box değerleri 5ten sonraki değerler score yani tahmin değerleri

            predict_id = np.argmax(scores) #scores içindeki max değerin indexini bulduk
            confidence = scores[predict_id] #güven skoru

            if confidence  > 0.30: #eğer tahmin değeri %30 un üstündeyse
                label = labels[predict_id] 
                bounding_box = object_deteciton[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])  #bize nesnenin merkez,width ve height koordinatlarını döndürür
                
                (box_center_x,box_center_y,box_width,box_height) = bounding_box.astype('int')

                start_x = int(box_center_x - (box_width/2))  #cv2.rectangle ile bounding box çizdirmek için merkezi verilen noktaların sol üst ve sağ alt köşesini buluyoruz
                start_y = int(box_center_y - (box_height/2))

                #### NON-MAXIMUM SUPPRESSION - OPERATION 2 ####
                ids_list.append(predict_id)
                confidence_list.append(float(confidence))
                boxes_list.append([start_x,start_y,int(box_width),int(box_height)])
                #### END OF OPERATION 2 ####

    max_ids = cv2.dnn.NMSBoxes(boxes_list,confidence_list,0.5,0.4) #en yüksek güvenirliğe sahip dikdörtgenlerin id lerini döndürüyor (array olarak)
    for max_id in max_ids:
        max_class_id = max_id
        box = boxes_list[max_class_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        #### NON-MAXIMUM SUPPRESSION - OPERATION 3 ####
        predict_id = ids_list[max_class_id]
        label = labels[predict_id]
        confidence = confidence_list[max_class_id]
        #### END OF OPERATION 2 ####

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predict_id]  #aynı nesneler için aynı renk olmasını sağlıyoruz
        box_color = [int(each) for each in box_color]
                
        label = '{}: {:.2f}%'.format(label,confidence*100)  #label ve güven değerini label değişkenine string olarak atıyoruz
        print('predicted object {}',format(label))
                
        cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),box_color,2) #nesnemizin çevreleyen bir dikdörtgen çiziyoruz(bounding box)
        cv2.putText(frame,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,2)


    cv2.imshow('Deteciton Window',frame) #ve son olarak tespit edilmiş videomuzu imshowlarız
    if cv2.waitKey(1) == ord('q'): # 'q' tuşuna basılırsa döngüyü kır 
        break
cap.release() #frameleri serbest bırak ve tüm pencereleri kapat
cv2.destroyAllWindows()