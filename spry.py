import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
import os,glob
# from main import frames
from tqdm import tqdm
from matplotlib import image
from matplotlib import pyplot as plt



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing_styles = mp.solutions.drawing_styles

def keyparts(path, name):
    IMAGE_FILES = [name]
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
            )

            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # cv2.imwrite('/result/annotated_image' + str(idx) + '.jpg', annotated_image)
    return annotated_image

def keyparts_main(path):
    labels = ['NOSE_x','NOSE_y','NOSE_z',
              'LEFT_EAR_vis','RIGHT_EAR_vis',
              'LEFT_EAR_z',"LEFT_EAR_x",'LEFT_EAR_y',
              'RIGHT_EAR_x', 'RIGHT_EAR_y','RIGHT_EAR_z',
              'LEFT_EYE_x', "LEFT_EYE_y",
              'RIGHT_EYE_x', 'RIGHT_EYE_y',
              'RIGHT_SHOULDER_x', 'RIGHT_SHOULDER_y',
              'RIGHT_WRIST_x', 'RIGHT_WRIST_y',
              'RIGHT_ELBOW_x', 'RIGHT_ELBOW_y', 'LEFT_SHOULDER_x', 'LEFT_SHOULDER_y', 'LEFT_ELBOW_x', 'LEFT_ELBOW_y',
              'LEFT_WRIST_x', 'LEFT_WRIST_y',
              'LEFT_ANKLE_x', 'LEFT_ANKLE_y', 'RIGHT_ANKLE_x', 'RIGHT_ANKLE_y', 'RIGHT_HIP_x', "RIGHT_HIP_y",
              "LEFT_HIP_x", 'LEFT_HIP_y',
              "RIGHT_KNEE_x","RIGHT_KNEE_y",
              "LEFT_KNEE_x", "LEFT_KNEE_y",
              "RIGHT_INDEX_x", "RIGHT_INDEX_y",
              "LEFT_INDEX_x", "LEFT_INDEX_y",
              "RIGHT_PINKY_x","RIGHT_PINKY_y",
              "LEFT_PINKY_x", "LEFT_PINKY_y",
              "LEFT_EYE_z",'RIGHT_EYE_z',
              'RIGHT_SHOULDER_z','RIGHT_WRIST_z',
              'LEFT_SHOULDER_z', 'LEFT_WRIST_z',
              'LEFT_ELBOW_z', 'RIGHT_ELBOW_z',
              'RIGHT_HIP_z', 'LEFT_HIP_z',"LEFT_PINKY_z",
              "RIGHT_PINKY_z","RIGHT_KNEE_z","LEFT_KNEE_z","RIGHT_INDEX_z","LEFT_INDEX_z",
              "RIGHT_ANKLE_z","LEFT_ANKLE_z","LEFT_HEEL_x", "LEFT_HEEL_y", "LEFT_HEEL_z",
              "RIGHT_HEEL_x","RIGHT_HEEL_y","RIGHT_HEEL_z",
              "LEFT_FOOT_INDEX_x","LEFT_FOOT_INDEX_y","LEFT_FOOT_INDEX_z",
              "RIGHT_FOOT_INDEX_x","RIGHT_FOOT_INDEX_y","RIGHT_FOOT_INDEX_z",
              "RIGHT_MOUTH_x","RIGHT_MOUTH_y","RIGHT_MOUTH_z",
              "LEFT_MOUTH_x","LEFT_MOUTH_y","LEFT_MOUTH_z"
              ]




    final=[]
    # For static images:
    length = []
    for name in glob.glob(path+'/*.jpg'):
        length.append(name)

    IMAGE_FILES = []
    for i in range(len(length)):
        h = path+'/frame'+str(i)+'.jpg'
        IMAGE_FILES.append(h)

    print(IMAGE_FILES)

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5) as pose:
        for i in tqdm(range(0, 1),total = 1,desc="processing "):

            for idx, file in tqdm(enumerate(IMAGE_FILES), desc = 'Running model on  frames'):
                data = []
                image = cv2.imread(file)
                image_height, image_width, _ = image.shape
                # Convert the BGR image to RGB before processing.
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.pose_landmarks:
                    continue

                xlpr = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width
                data.append(xlpr)
                ylpr = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height
                data.append(ylpr)
                ylz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].z
                data.append(ylz)

                xlvis = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].visibility
                data.append(xlvis)
                ylvis = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].visibility
                data.append(ylvis)

                xlear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].z
                data.append(xlear)
                xll = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x * image_width
                data.append(xll)
                yll = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y * image_height
                data.append(yll)

                xllr = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x * image_width
                data.append(xllr)
                yllr = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y * image_height
                data.append(yllr)
                ylear = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].z
                data.append(ylear)

                xl = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x * image_width
                data.append(xl)
                yl = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y * image_height
                data.append(yl)

                xr = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x * image_width
                data.append(xr)
                yr = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y * image_height
                data.append(yr)

                x1 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width
                data.append(x1)
                y1 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height
                data.append(y1)

                x2 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x * image_width
                data.append(x2)
                y2 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * image_height
                data.append(y2)

                x3 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * image_width
                data.append(x3)
                y3 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * image_height
                data.append(y3)

                x4 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width
                data.append(x4)
                y4 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height
                data.append(y4)

                x5 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x * image_width
                data.append(x5)
                y5 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * image_height
                data.append(y5)

                x6 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x * image_width
                data.append(x6)
                y6 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * image_height
                data.append(y6)

                x7 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].x * image_width
                data.append(x7)
                y7 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].y * image_height
                data.append(y7)

                x8 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].x * image_width
                data.append(x8)
                y8 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].y * image_height
                data.append(y8)

                x9 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x * image_width
                data.append(x9)
                y9 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y * image_height
                data.append(y9)

                x99 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * image_width
                data.append(x99)
                y99 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y * image_height
                data.append(y99)

                rkneex = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].x * image_width
                data.append(rkneex)
                rkneey = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].y * image_height
                data.append(rkneey)

                lkneex = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x * image_width
                data.append(lkneex)
                lkneey = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y * image_height
                data.append(lkneey)

                rindex = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].x * image_width
                data.append(rindex)
                rindexy = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].y * image_height
                data.append(rindexy)

                lindex = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].x * image_width
                data.append(lindex)
                lindey = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].y * image_height
                data.append(lindey)

                pnkx = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].x * image_width
                data.append(pnkx)
                pnky = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].y * image_height
                data.append(pnky)

                lpnkx = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_PINKY].x * image_width
                data.append(lpnkx)
                rpnky = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_PINKY].y * image_height
                data.append(rpnky)

                leyz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].z
                data.append(leyz)
                reyz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].z
                data.append(reyz)

                rshz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].z
                data.append(rshz)
                rwriz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].z
                data.append(rwriz)

                lshz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z
                data.append(lshz)
                lwriz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z
                data.append(lwriz)

                lelwz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].z
                data.append(lelwz)
                relwz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].z
                data.append(relwz)

                lhipz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].z
                data.append(lhipz)
                rhipz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].z
                data.append(rhipz)

                lpinkz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_PINKY].z
                data.append(lpinkz)
                rpinkz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].z
                data.append(rpinkz)

                rkneezz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].z
                data.append(rkneezz)
                lkneezz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].z
                data.append(lkneezz)

                rindexzzz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].z
                data.append(rindexzzz)
                llindexzzz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].z
                data.append(llindexzzz)

                rankz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE].z
                data.append(rankz)
                lankz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].z
                data.append(lankz)

                lheelx = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HEEL].x * image_width
                data.append(lheelx)
                lheely = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HEEL].y * image_height
                data.append(lheely)
                lheelz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HEEL].z
                data.append(lheelz)

                rheelx = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HEEL].x * image_width
                data.append(rheelx)
                rheely = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HEEL].y * image_height
                data.append(rheely)
                rheelz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HEEL].z
                data.append(rheelz)

                lfootix = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].x * image_width
                data.append(lfootix)
                lfootiy = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].y * image_height
                data.append(lfootiy)
                lfootiz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX].z
                data.append(lfootiz)

                rfootix = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX].x * image_width
                data.append(rfootix)
                rfootiy = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX].y * image_height
                data.append(rfootiy)
                rfootiz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX].z
                data.append(rfootiz)

                rmoux = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].x * image_width
                data.append(rmoux)
                rmouy = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].y * image_height
                data.append(rmouy)
                rmouz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].z
                data.append(rmouz)

                lmoux = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].x * image_width
                data.append(lmoux)
                lmouy = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].y * image_height
                data.append(lmouy)
                lmouz = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].z
                data.append(lmouz)

                final.append(data)

    cc = np.array(final)
    df = pd.DataFrame(cc, columns=labels)

    df['CHEST_x'] = ((df['RIGHT_SHOULDER_x']+df['LEFT_SHOULDER_x'])/2)-5
    df['CHEST_y'] = ((df['RIGHT_SHOULDER_y'] + df['LEFT_SHOULDER_y']) / 2)-5
    df['CHEST_z'] = df['RIGHT_SHOULDER_z']

    df['NECK_x'] = ((df['RIGHT_MOUTH_x'] + df['LEFT_MOUTH_x']) / 2)
    df['NECK_y'] = ((df['RIGHT_MOUTH_y'] + df['RIGHT_MOUTH_y']) / 2) +10
    df['NECK_z'] = df['RIGHT_MOUTH_z']

    df['LUMBAR_x'] = ((df['RIGHT_HIP_x']+df['LEFT_HIP_x'])/2)+5
    df['LUMBAR_y'] = ((df['RIGHT_HIP_y'] + df['RIGHT_HIP_y']) / 2)+5
    df['LUMBAR_z'] = df['RIGHT_HIP_z']

    df['ABDOMEN_x'] = ((df['RIGHT_HIP_x'] + df['LEFT_HIP_x']) / 2) + 5
    df['ABDOMEN_y'] = ((df['RIGHT_HIP_y'] + df['RIGHT_HIP_y']) / 2) + 5
    df['ABDOMEN_z'] = df['RIGHT_HIP_z']
    return df


menu = ["Angle Plot", "General Plot"]
choice = st.sidebar.selectbox("Menu", menu)

video_url = st.text_input("Video URL")
button = st.button("RUN")
if button:
    r = requests.get(video_url, allow_redirects=True)
    filename = "test.webm"
    open('test.webm', 'wb').write(r.content)
    cam = cv2.VideoCapture("test.webm")
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except:
        print("")
    
    # frame
    currentframe = 0
    
    while(True):
        
        # reading from frame
        ret,frame = cam.read()
    
        if ret:
            # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

def layer(folder_name, path, parts=[],frame=int):
    df= keyparts_main(folder_name)
    z1 = parts[0] + "_x"
    z2 = parts[0] + "_y"
    z3 = parts[1] + "_x"
    z4 = parts[1] + "_y"
    z5 = parts[2] + "_x"
    z6 = parts[2] + "_y"
    
    data = image.imread(path)
    k1 = df[z1][frame]
    k2 = df[z2][frame]
    k3 = df[z3][frame]
    k4 = df[z4][frame]
    k5 = df[z5][frame]
    k6 = df[z6][frame]
    
    x = [k1,k3,k5]
    y = [k2,k4,k6]
    plt.plot(x, y, color="white", linewidth=3)
    plt.imshow(data)
    # plt.savefig("output/"+str(frame)+".jpg")
    return plt

if choice == "Angle Plot":

    parts = st.multiselect("Parts", ['NOSE',
                'LEFT_EAR',
                'RIGHT_EAR',
                'LEFT_EYE',
                'RIGHT_EYE',
                'RIGHT_SHOULDER',
                'RIGHT_WRIST',
                'RIGHT_ELBOW', 'LEFT_SHOULDER', 'LEFT_ELBOW',
                'LEFT_WRIST',
                'LEFT_ANKLE', 'RIGHT_ANKLE', 'RIGHT_HIP',
                'LEFT_HIP',
                "RIGHT_KNEE",
                "LEFT_KNEE",
                "RIGHT_INDEX",
                "LEFT_INDEX",
                "RIGHT_PINKY",
                "LEFT_PINKY",
                "LEFT_HEEL",
                "RIGHT_HEEL",
                "LEFT_FOOT_INDEX",
                "RIGHT_FOOT_INDEX",
                "RIGHT_MOUTH",
                "LEFT_MOUTH",
                "LUMBAR",
                "CHEST",
                "NECK"
                ])
    show = st.checkbox("show")
    if show:
        length = len([f for f in os.listdir('data') 
        if f.endswith('.jpg') and os.path.isfile(os.path.join('data', f))])
        length = length - 1
        st.write("Video has ", str(length), " frames")
        slider = st.slider("Frame", 1, length, length)
        name = "data/"+"frame"+str(slider)+".jpg"
        answer = layer('data', name, parts, slider)
        st.pyplot(answer)
elif choice == "General Plot":
    show = st.checkbox("show")
    if show:
        length = len([f for f in os.listdir('data') 
        if f.endswith('.jpg') and os.path.isfile(os.path.join('data', f))])
        length = length - 1
        st.write("Video has ", str(length), " frames")
        slider = st.slider("Frame", 1, length, length)
        name = "data/"+"frame"+str(slider)+".jpg"
        answer = keyparts('data', name)
        st.image(answer)