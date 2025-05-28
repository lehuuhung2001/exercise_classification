import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf

label = "Warmup...."
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# model = tf.keras.models.load_model("../model_bidirectional_world_3.h5")

# EMA smoothing
prev_c_lm = None  # Global hoặc đặt trong real_time_detection nếu muốn giới hạn scope
alpha = 0.3       # Độ mượt, từ 0.1–0.5 

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
        # print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    if (label == "NOT DETECTED"):
        fontColor = (0, 0, 255)
    elif (label == "Warmup...."):
        fontColor = (0, 165, 255)
    else:
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
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    # print(lm_list.shape)
    results = model.predict(lm_list)
    print(type(results))
    np.set_printoptions(suppress=True, precision=6)
    print(results)
    if results[0][0] > 0.9:
        label = "BARBELL BICEPS CURL"
    elif results[0][1] > 0.9:
        label = "BENCH PRESS"
    elif results[0][2]> 0.9:
        label = "CHEST FLY MACHINE"
    elif results[0][3] > 0.9:
        label = "DEADLIFT"
    elif results[0][4]> 0.9:
        label = "DECLINE BENCH PRESS"
    elif results[0][5] > 0.9:
        label = "HAMMER CURL"
    elif results[0][6]> 0.9:
        label = "HIP THRUST"
    elif results[0][7] > 0.9:
        label = "INCLINE BENCH PRESS"
    elif results[0][8]> 0.9:
        label = "LAT PULLDOWN"
    elif results[0][9] > 0.9:
        label = "LATERAL RAISE"
    elif results[0][10]> 0.9:
        label = "LEG EXTENSION"
    elif results[0][11] > 0.9:
        label = "LEG RAISES"
    elif results[0][12]> 0.9:
        label = "PLANK"
    elif results[0][13] > 0.9:
        label = "PULL UP"
    elif results[0][14]> 0.9:
        label = "PUSH UP"
    elif results[0][15] > 0.9:
        label = "ROMANIAN DEADLIFT"
    elif results[0][16]> 0.9:
        label = "RUSSIAN TWIST"
    elif results[0][17] > 0.9:
        label = "SHOULDER PRESS"
    elif results[0][18]> 0.9:
        label = "SQUAT"
    elif results[0][19]> 0.9:
        label = "T BAR ROW"
    elif results[0][20] > 0.9:
        label = "TRICEP DIPS"
    elif results[0][21]> 0.9:
        label = "TRICEP PUSHDOWN"
    else:
        label = "NOT DETECTED"
    return label


def real_time_detection(model):
    global label
    global lm_list
    global prev_c_lm
    n_time_steps = 35
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    i = 0
    warmup_frames = 120

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

                # EMA smoothing
                if prev_c_lm is None:
                    prev_c_lm = c_lm
                else:
                    c_lm = [alpha * curr + (1 - alpha) * prev for curr, prev in zip(c_lm, prev_c_lm)]
                    prev_c_lm = c_lm

                lm_list.append(c_lm)
                if len(lm_list) == n_time_steps:
                    # predict
                    t1 = threading.Thread(target=detect, args=(model, lm_list,))
                    t1.start()
                    lm_list = []

                img = draw_landmark_on_image(mpDraw, results, img)
            else: 
                label = "Waiting..."
        img = draw_class_on_image(label, img)
        _, jpeg = cv2.imencode('.jpg', img)
        img = jpeg.tobytes()
        # Truyền frame qua HTTP dưới dạng MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')
        # cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     real_time_detection(model)
