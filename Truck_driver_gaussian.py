from IPython.display import clear_output
from google.colab.patches import cv2_imshow
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import youtube_dl
import pafy
import pandas as pd
from datetime import datetime

def eye_aspect_ratio(eye):
        data = {}
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear,A,B,C


def ProcessVideoForEyes(video_link    = None,
                        model_path    = None,
                        collect_data  = False,
                        new_dataframe = pd.DataFrame()):
    """
    this function creates all the things required for the 
    Processig of Eyes in a Frame
    """ 

    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    # print("[INFO] loading facial landmark predictor...")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


    url = video_link
    vPafy = pafy.new(url)
    play = vPafy.getbest()

    cap = cv2.VideoCapture(play.url)
    result = cv2.VideoWriter( "processed_video.avi",  
                              cv2.VideoWriter_fourcc(*'MJPG'), 
                              10, (640,360))

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    print(" PROCESSING THE VIDEO NOW ...")
    print(" PLEASE WAIT.....")
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (640, 360))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR,ld15,ld24,ld03 = eye_aspect_ratio(leftEye)
                rightEAR,rd15,rd24,rd03 = eye_aspect_ratio(rightEye)

    
                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0

                # Gaussion maths
                value1 = -0.5 * np.power(((ld15 - 8.4)/1.55),2) 
                E1 = ((1/1.55) * (1/2.5066) * np.exp(value1))
                
                value2 = -0.5 * np.power(((ld15 - 6.85)/1.52),2)
                E2 = ((1/1.52) * (1/2.5066) * np.exp(value2)) + np.exp(-0.205)
                

                

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                    
                    if collect_data:
                        #########  Subject To change #######
                        data = {}

                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")

                        data['Time'] = [current_time]
                        data['ear_threshhold'] = [EYE_AR_THRESH]
                        data['l_ear'] = [leftEAR]
                        data['r_ear'] = [rightEAR]

                        data['l_d15'] = [ld15]
                        data['l_d24'] = [ld24]
                        data['l_d03'] = [ld03]

                        data['r_d15'] = [rd15]
                        data['r_d24'] = [rd24]
                        data['r_d03'] = [rd03]
                        
                        data['Average_ear'] = [ear]
                        data['blink']       = [1]
                        ####################################

                    COUNTER += 1
                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    if collect_data:
                        #########  Subject To change #######
                        data = {}

                        now = datetime.now()
                        current_time = now.strftime("%m/%d/%Y, %H:%M:%S")

                        data['Time'] = [current_time]
                        data['ear_threshhold'] = [EYE_AR_THRESH]
                        data['l_ear'] = [leftEAR]
                        data['r_ear'] = [rightEAR]

                        data['l_d15'] = [ld15]
                        data['l_d24'] = [ld24]
                        data['l_d03'] = [ld03]

                        data['r_d15'] = [rd15]
                        data['r_d24'] = [rd24]
                        data['r_d03'] = [rd03]
                        
                        data['Average_ear'] = [ear]
                        data['blink']       = [0]
                        new_dataframe = new_dataframe.append(pd.DataFrame(data))
                        ####################################
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                    # reset the eye frame counter
                    COUNTER = 0

                # draw the total number of blinks on the frame along with
                # the computed eye aspect ratio for the frame


                if ear > 0.15:
                    if E1 > E2:
                        cv2.putText(frame,"ACTIVE DRIVER",(10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 0, 255), 2)
                    if E2 >= E1:
                        cv2.putText(frame,f"DIZZY DRIVER : {E2}",(10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 0, 255), 2)    
                    # cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 0, 255), 2)
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 0, 255), 2)
            result.write(frame) 

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    result.release()
    print("video is available and named as \"processed_video.avi\" ")
    return new_dataframe
