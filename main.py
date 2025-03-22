import os, cv2
import numpy as np

os.system('cls')
imgPath = './images'
labelPath = './labels'
imgList = os.listdir(imgPath)

output_layers,classes=[],[]
net = cv2.dnn.readNet("./models/yolov3.weights", "./models/yolov3.cfg")
layer_names = net.getLayerNames()
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i - 1])

with open("./models/classes.txt", "r") as f:
    for line in f.readlines():
        classes.append(line.strip().title())

for img_name in imgList:
    img = cv2.imread(os.path.join(imgPath, img_name))
    height, width, channels = img.shape
    cv2.imshow("Image", img)

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids,confidences,boxes = [],[],[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    dst_txt = img_name.split('.')[0] + '.txt'
    with open(os.path.join(labelPath, dst_txt), 'w') as f:
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                f.write(f'{label} : {x} {y} {x + w} {y + h}\n')

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "Person":
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()