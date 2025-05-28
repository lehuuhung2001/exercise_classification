import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
from collections import Counter


label = "Warmup...."
predictions = np.array([])
# print(predictions)
event_label_updated = threading.Event()
event_predictions_updated = threading.Event()


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("model.h5")

cap = cv2.VideoCapture("video20.mp4")

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_world_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        if lm.visibility < 0.55 and id > 0:
            prev_lm = results.pose_landmarks.landmark[id - 1]
            next_lm = results.pose_landmarks.landmark[id + 1] if id + 1 < len(results.pose_landmarks.landmark) else prev_lm

            cx = int((prev_lm.x + next_lm.x) / 2 * w)
            cy = int((prev_lm.y + next_lm.y) / 2 * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global predictions
    event_predictions_updated.clear()
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    predictions = model.predict(lm_list)
    np.set_printoptions(suppress=True, precision=6)
    event_predictions_updated.set()
    detect_label(predictions)

def detect_label(predictions):
    global label
    event_label_updated.clear()
    if predictions[0][0] > 0.5:
        label = "BARBELL BICEPS CURL"
    elif predictions[0][1] > 0.5:
        label = "BENCH PRESS"
    elif predictions[0][2]> 0.5:
        label = "CHEST FLY MACHINE"
    elif predictions[0][3] > 0.5:
        label = "DEADLIFT"
    elif predictions[0][4]> 0.5:
        label = "DECLINE BENCH PRESS"
    elif predictions[0][5] > 0.5:
        label = "HAMMER CURL"
    elif predictions[0][6]> 0.5:
        label = "HIP THRUST"
    elif predictions[0][7] > 0.5:
        label = "INCLINE BENCH PRESS"
    elif predictions[0][8]> 0.5:
        label = "LAT PULLDOWN"
    elif predictions[0][9] > 0.5:
        label = "LATERAL RAISE"
    elif predictions[0][10]> 0.5:
        label = "LEG EXTENSION"
    elif predictions[0][11] > 0.5:
        label = "LEG RAISES"
    elif predictions[0][12]> 0.5:
        label = "PLANK"
    elif predictions[0][13] > 0.5:
        label = "PULL UP"
    elif predictions[0][14]> 0.5:
        label = "PUSH UP"
    elif predictions[0][15] > 0.5:
        label = "ROMANIAN DEADLIFT"
    elif predictions[0][16]> 0.5:
        label = "RUSSIAN TWIST"
    elif predictions[0][17] > 0.5:
        label = "SHOULDER PRESS"
    elif predictions[0][18]> 0.5:
        label = "SQUAT"
    elif predictions[0][19]> 0.5:
        label = "T BAR ROW"
    elif predictions[0][20] > 0.5:
        label = "TRICEP DIPS"
    elif predictions[0][21]> 0.5:
        label = "TRICEP PUSHDOWN"
    event_label_updated.set()
    return label

def most_common_element(lst):
    count = Counter(lst)  # Đếm số lần xuất hiện của mỗi phần tử
    return count.most_common(1)[0][0]  # Trả về phần tử xuất hiện nhiều nhất

def main(): 
    # average_accuray = None
    global label
    global predictions
    sum_predictions =0
    i = 0
    warmup_frames = 15
    n_time_steps = 35
    lm_list = []
    time_predictions = 0
    list_label = []
    while True:

        ret, img = cap.read()
        if not ret:
            break
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        i = i + 1
    
        if i > warmup_frames:
            # print("Start detect....")

            if results.pose_world_landmarks:
                c_lm = make_landmark_timestep(results)
                lm_list.append(c_lm)

                if len(lm_list) == n_time_steps:
                    # predict
                    t1 = threading.Thread(target=detect, args=(model, lm_list,))
                    t1.start()
                    list_label.append(label)
                    if predictions.size > 0:
                        sum_predictions += predictions[0][:]
                        time_predictions+=1
                    lm_list = []

                img = draw_landmark_on_image(mpDraw, results, img)
            else: 
                label = "Waiting..."
        img = draw_class_on_image(label, img)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break
        if i == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            average_predictions = sum_predictions/time_predictions
            average_predictions = np.expand_dims(average_predictions, axis=0)
            main_label = detect_label(average_predictions)
            print(main_label)
    

    label_most = most_common_element(list_label)
    print(label_most)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()