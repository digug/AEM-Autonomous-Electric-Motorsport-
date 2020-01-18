import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# Initialize starting parameters for the program
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

parser = argparse.ArgumentParser(
    description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

classesFile = "C:\\Users\\dougl\\Desktop\\YOLOv3_with_OpenCV\\coco.names"
classes = None

# Open the classs files to load in possible detections
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Configuration file for YOLOV3
modelConf = 'C:\\Users\\dougl\\Desktop\\YOLOv3_with_OpenCV\\yolov3.cfg'
# Pre-trained weights for YOLOV3
modelWeights = 'C:\\Users\\dougl\\Desktop\\YOLOv3_with_OpenCV\\yolov3.weights'


# Loading the neural network
net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)  # The target processor
# ^Perferable target could be cv.dnn.DNN_TARGET_OPENCL - Intel GPU

# Get the names of the output layers


def getOutputsNames(net):
    # Get the names of all the layers in network
    layerNames = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima supression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores.
    # Assign the box's class label as the class with the highest score.
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                # x location of the left edge of the bounding box
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppession to eliminate redundant overlapping
    # boxes with lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        drawPred(classIds[i], confidences[i], left,
                 top, left + width, top + height)


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box by providing input frame and specifying coord.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(
        label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.putText(frame, label, (left, top),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


winName = 'Object Detection with OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 1000, 1000)

# The following identifies the type of file of input (video/webcam/image) and uses orresponding process
outputFile = "yolo_out_py.avi"
if(args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_yolo_out_py.jpg'
elif(args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

'''
# Get the video writer initialized to save the output video
if(not args.image):
    vid_writer = cv.Videowriter(outputFile, cv.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
'''

# Loop to keep the program running until a key is hit
while cv.waitKey(1) < 0:
    # getting from from video/input
    hasFrame, frame = cap.read()

    # Stop the programm if end of vid is reached
    if not hasFrame:
        print("Done processing!")
        print("Output file is stored as ", outputFile)
        cv.waitkey(3000)
        break

    # 4D blob creation from input frame
    blob = cv.dnn.blobFromImage(
        frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets blob as the input to the neural network
    net.setInput(blob)

    # Forward pass to get the output if output layer
    outs = net.forward(getOutputsNames(net))

    # Remove bounding boxes with "objects" with low confidence
    postprocess(frame, outs)

    # Show window with window name and input fram as inputs
    cv.imshow(winName, frame)

    # Put effiency imformation.
    # The function getPerfProfile returns overall time for inference(t) and the timings for each of the layers (in layerTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t*1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    '''          
    # Write the frame with the detection boxes
    if(args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.unit8))'''
