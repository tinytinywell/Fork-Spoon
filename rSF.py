import cv2
import os
import numpy as np


def object_detection(outs, image):
    """
    Remove the bounding boxes with low confidence
    :param outs: forward output
    :param image: input image
    """
    # Confidence used to remove bbox with lower confidence
    score_confidence = 0.5
    # threshold used to perform nms
    nms_threshold = 0.3
    # list to store the location of bbox
    bbox = []
    # list to store confidence
    confidences = []
    # list to store class id
    class_ids = []
    # get the height and width of image
    h_img, w_img = image.shape[:2]
    # the outputs(outs) has 3 parts(out, from 3 scales), each part has multiple detections(detected bbox)
    # check all the bboxes and remove the weak predictions
    for out in outs:
        for detection in out:
            # the scores of 80 classes
            class_scores = detection[5:]
            # find the index with maximal score
            class_id = np.argmax(class_scores)
            # maximal score
            max_confidence = class_scores[class_id]
            # keep the bbox with higher score
            if max_confidence > score_confidence:
                # scale the bounding box coordinates back relative to the size of the image
                center_x_bbox, center_y_bbox = detection[0]*w_img, detection[1]*h_img
                w_bbox, h_bbox = int(detection[2]*w_img), int(detection[3]*h_img)
                # get the top and left corner x, y of the bbox
                x_bbox, y_bbox = int(center_x_bbox - w_bbox/2), int(center_y_bbox - h_bbox/2)
                # update bbox, confidences and class_ids
                bbox.append([x_bbox, y_bbox, w_bbox, h_bbox])
                confidences.append(float(max_confidence))
                class_ids.append(class_id)
    # nms to suppress weak, overlapping bboxes
    id_nms = cv2.dnn.NMSBoxes(bbox, confidences, score_confidence, nms_threshold)
    if len(id_nms) > 0:
        for index in id_nms:
            index = index[0]
            x_bbox, y_bbox, w_bbox, h_bbox = bbox[index][:4]
            # draw the bbox and put text
            cv2.rectangle(image, (x_bbox, y_bbox), (x_bbox + w_bbox, y_bbox + h_bbox), (0, 155, 255), 2)
            text = "{}: {:.4f}".format(labels[class_ids[index]], confidences[index])
            cv2.putText(image, text, (x_bbox, y_bbox - 10), cv2.QT_FONT_NORMAL, 0.7, (0, 155, 255), 2)


# the path of py file
file_path = os.path.abspath(os.path.dirname(__file__))
# the path of label file coco.names
label_path = file_path + '/coco.names'
# the path of weights file yolov3.weights
weights_path = file_path + '/yolov3.weights'
# the path of cfg file yolov3.cfg
cfg_path = file_path + '/yolov3.cfg'
# open label file and store all classes name in a list
with open(label_path) as f:
    labels = f.read().split('\n')
# load the net
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
# open the computer camera
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # read the camera frame
    ret, img = cam.read()
    #img = cv2.imread(file_path + '\\test.jpg')
    # transform img to blob format as the input of the net
    img_blob = cv2.dnn.blobFromImage(img, 1.0/400.0, (416, 416), None, True, False)
    net.setInput(img_blob)
    # get the names of 3 output layers and forward
    outLayersName = net.getUnconnectedOutLayersNames()
    # the shape of outputs:(numbers of detected bbox, 85),
    # 85=4(center x,center y,w,h of bbox)+1(confidence)+80(scores of 80 classes)
    outputs = net.forward(outLayersName)
    object_detection(outputs, img)

    cv2.imshow("detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()