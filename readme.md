# EE 443: MOT Tutorial
1. Use the "EE 443 MOT video generator.ipynb" notebook
2. Make sure EE443_track1.zip is in your Google Drive before mounting it.
3. Run each cell line by line. Read the text cells and code comments for any specific instructions.
4. When downloading ByteTrack, please change requirements.txt to have onnx and onnxruntime==1.12.0 instead of 1.8.0/1.8.1
5. The first instance of the Bytetrack code will generate detections.txt for you to use with the main.py starter code. Ignore any other file produced, it won't be correct. Download detections and use it with the embeddings.npy file through main.py to get a result.txt file.
6. Then upload this result.txt file and modify the filepath to match it. Run the second ByteTrack cell to generate the video with the correct labels

# EE 443 Final Project Report - Multi Object Tracking

## ***Section i. Introduction and Overview***
Multiple Object Tracking (MOT) is a computer vision application of machine learning that applies detection, tracking, classification, and identification components in order to develop a model that can track multiple objects in a video stream. In particular, our implementation’s objective was to be able to track multiple people walking around in an enclosed area, using the data from multiple cameras for training and testing the model on another camera. Each person needs to be identified with an accurately sized bounding box and assigned a unique ID that does not change even if the person leaves and reenters the camera feed.

## ***Section ii. Implementation***
To implement the functionality described in the specification, we made use of many tools that performed different tasks for us. Each task required a certain amount of processing and tweaking in order to be able to use the selected tool for the task properly. The process that we underwent to use these tools will be expanded on in this section.

### Detection
To detect images, we used YOLOv8. This included passing each frame of the video into the model and having it identify instances of humans, then drawing accurately sized bounding boxes with a certain confidence score. We used a pre-trained yolov8n model on the COCO dataset, then trained that model further on the data provided to us. In order to achieve this, we first had to reformat all the training labels into COCO format so that the YOLOv8 model would be able to understand and use it for training. In addition, each frame needed a corresponding label text file with the same filename as the frame image. 

Since the dataset was so large, we only trained the model for 1 epoch over all of the provided training datasets. We were able to achieve good results after validating the trained model. As shown in the image below, accurate bounding boxes were able to be drawn for each detected instance of a person, which was the functionality that we were aiming to achieve.

### Tracking 
For tracking the detected instances of people across multiple frames in a video, we used ByteTrack. ByteTrack made use of a prediction from the trained YOLOv8 model for every frame in a video, and sequenced the bounding boxes as well as annotations like confidence score, tracking ID and class to create a .mp4 video containing the entire multiple object tracking result. However, ByteTrack’s tracking ID did not account for re-identification, which ended up implementing separately using the starter code provided by Hsiang-Wei Huang. 

ByteTrack’s main use was to be able to associate every detection box instead of only ones with high confidence scores, and use the low confidence score detections to filter out background detections and recover true objects. They make use of tracklets with low scoring detection boxes in order to distinguish and track objects that may be obstructed by other objects.

For our model, ByteTrack would generate tracks after it generated YOLOv8 predicted detection coordinates. It used these tracks to update the byte_track object to keep track of the trajectory of an object through tracklets. This allowed for smoother bounding box movement in the generated video, as the bounding box would traverse along with the detected person instead of getting redrawn at every single frame.

### Re-Identification 
Re-Identification was accomplished via KMeans clustering on a large set of custom embeddings, then feeding it back into ByteTrack and provided code to adjust and maintain consistent labels throughout the video as people enter and exit the frame. The total number of clusters was set to be the number of people in a given clip. We used scikit-learn’s built in clustering algorithm as opposed to a manual implementation. Outside of the number of detected people, and the initial random state, all other parameters were left as default. Better performance could perhaps be achieved through the implementation of a more complex or alternative algorithm, but KMean clustering appears to have produced good results.



Custom Embeddings: https://drive.google.com/file/d/1SsIGDiYEjQOCmZIJ9Gvzbg0BPTnq3Hfs/view
