from flask import Flask, request, render_template, send_from_directory, session, Response
import os
import tensorflow as tf
import cv2
import numpy as np
import inference_lstm
from inference_lstm import make_landmark_timestep, draw_landmark_on_image, draw_class_on_image, detect,detect_label, pose, mpDraw, event_label_updated, event_predictions_updated, most_common_element
from process.live_detection import real_time_detection
import threading
from flask_socketio import SocketIO
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
POSE_FOLDER = "static/poses"

app.secret_key = "supersecretkey"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["POSE_FOLDER"] = POSE_FOLDER

socketio = SocketIO(app)

label_most = None

model = tf.keras.models.load_model("model.h5")
# print(model)
# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(POSE_FOLDER):
    os.makedirs(POSE_FOLDER)


# Lưu video xử lí POSE 
def process_video(video_path, output_path):
    global label_most
    i = 0
    sum_predictions =0
    time_predictions = 0
    warmup_frames = 15
    n_time_steps = 35
    lm_list = []
    list_label = []
    # global label
    cap = cv2.VideoCapture(video_path)

    # Lấy thông tin video
    fourcc = cv2.VideoWriter_fourcc(*"AVC1")  # Codec để ghi file mp4
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while cap.isOpened():
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
                    event_label_updated.clear()  # Reset sự kiện trước khi dự đoán
                    event_predictions_updated.clear()
                    t1 = threading.Thread(target=detect, args=(model, lm_list))
                    t1.start()
                    event_predictions_updated.wait()
                    event_label_updated.wait()  # Chờ đến khi `detect()` cập nhật xong
                    list_label.append(inference_lstm.label)
                    if inference_lstm.predictions.size > 0:
                        sum_predictions += inference_lstm.predictions[0][:]
                        time_predictions+=1
                    lm_list = []

                img = draw_landmark_on_image(mpDraw, results, img)
            else: 
                label = "Waiting..."
        img = draw_class_on_image(inference_lstm.label, img)

        if i == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            average_predictions = sum_predictions/time_predictions
            average_predictions = np.expand_dims(average_predictions, axis=0)
            main_label = detect_label(average_predictions)
            # print(main_label)

        out.write(img)  # Ghi frame vào video output
    label_most = most_common_element(list_label)
    cap.release()
    out.release()


@app.route("/", methods=["GET", "POST"])
def live_detection():
    # Trả về trang HTML với video stream được nhúng vào trong một khung nhỏ
    return render_template("live_detection.html")

@app.route('/video_feed')
def video_feed():
    # Trả về video stream theo định dạng MJPEG
    return Response(real_time_detection(model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/videoclassification", methods=["GET", "POST"])
def video_classification():
    input_path = session.get("input_path")
    file_name = session.get("file_name")
    if request.method == "POST":
        file = request.files["file"]
        # print(file, flush=True)
        if file and file.filename.endswith(".mp4"):
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            input_path = input_path.replace("\\", "/")
            
            session["input_path"] = input_path
            session["file_name"] = file.filename
            file.save(input_path)
    return render_template("video_classification.html", video_url=input_path)

@app.route("/identify", methods=["GET", "POST"])
def identify():
    input_path = session.get("input_path")  # Lấy video_url đã lưu
    file_name = session.get("file_name")
    output_path = os.path.join(app.config["POSE_FOLDER"], file_name)
    output_path = output_path.replace("\\", "/")
    process_video(input_path, output_path)
    print(label_most)
    # print(input_path, flush=True)
    # print(output_path, flush=True)
    # return f"Identifying actions in video: {output_path}"
    return render_template("identify.html", video_url=output_path, label_most = label_most)
# Route để Flask phục vụ video từ thư mục `static/uploads/`
# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# @app.route('/video_feed')
# def video_feed():
#     # Trả về video stream theo định dạng MJPEG
#     return Response(real_time_detection(model),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
