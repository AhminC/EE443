import numpy as np
from IoU_Tracker import tracker
from Processing import postprocess

if __name__ == "__main__":

    # camera = 74 # validation set
    camera = 75 # test set

    # The number of people in the dataset is 5. Bonus points for method that does not require this line of hard coding 
    number_of_people = 5

    result_path = 'latest_result.txt'

    # Load the data
    detection = np.loadtxt('./detections4_mod.txt',delimiter=',',dtype=None)
    embedding = np.load('./embedding_latest.npy', allow_pickle=True)
    # inds = detection[:,0] == camera
    test_detection = detection
    test_embedding = embedding
    sort_inds = test_detection[:, 1].argsort()
    test_detection = test_detection[sort_inds]
    test_embedding = test_embedding[sort_inds]

    # TODO
    # Can we remove some low score detection here? For example, removing all the detections with condifence score lower than 0.3?
    # Detection format: <camera ID>, <Frame ID>, <class>, <x1>, <y1>, <x2>, <y2>, <confidence score>
    # Remember to change the embedding file too if you modify the detection!
    # inds = test_detection[:,7] >= 0.3
    # test_detection = test_detection[inds]
    # test_embedding = test_embedding[inds]

    mot = tracker()
    postprocessing = postprocess(number_of_people,'kmeans')

    # Run the IoU tracking
    tracklets = mot.run(test_detection,test_embedding)

    features = np.array([trk.final_features for trk in tracklets])

    # Run the Post Processing to merge the tracklets
    labels = postprocessing.run(features) # The label represents the final tracking ID, it starts from 0. We will make it start from 1 later.

    tracking_result = []

    print('Writing Result ... ')

    for i,trk in enumerate(tracklets):
        final_tracking_id = labels[i]+1 # make it starts with 1
        for idx in range(len(trk.boxes)):
            frame = trk.times[idx]
            x1,y1,x2,y2 = trk.boxes[idx]
            x,y,w,h = x1,y1,x2-x1,y2-y1
            
            result = '{},{},{},{},{},{},{},-1,-1 \n'.format(camera,final_tracking_id,frame,x,y,w,h)
    
            tracking_result.append(result)
    
    print('Save tracking results at {}'.format(result_path))

    with open(result_path,'w') as f:
        f.writelines(tracking_result)
