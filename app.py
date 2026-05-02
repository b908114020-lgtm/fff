import streamlit as st
import cv2
import math
import time
import threading
import numpy as np
import av

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ==============================
# 姿勢判斷函式
# ==============================
def calc_distance(p1, p2):
    return math.sqrt(
        (float(p1[0]) - float(p2[0])) ** 2 +
        (float(p1[1]) - float(p2[1])) ** 2
    )


def judge_posture(results):
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return "No Person Detected"

    keypoints = results[0].keypoints.xy[0]
    confs = results[0].keypoints.conf[0]

    if len(keypoints) < 13:
        return "Keypoints Not Enough"

    nose = keypoints[0]

    left_ear = keypoints[3]
    right_ear = keypoints[4]

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    left_hip = keypoints[11]
    right_hip = keypoints[12]

    left_ear_conf = float(confs[3])
    right_ear_conf = float(confs[4])
    left_shoulder_conf = float(confs[5])
    right_shoulder_conf = float(confs[6])

    shoulder_width = calc_distance(left_shoulder, right_shoulder)

    torso_length = (
        calc_distance(left_shoulder, left_hip) +
        calc_distance(right_shoulder, right_hip)
    ) / 2

    is_side_lying = False

    if left_shoulder_conf < 0.4 or right_shoulder_conf < 0.4:
        is_side_lying = True

    elif torso_length > 5 and shoulder_width / torso_length < 0.5:
        is_side_lying = True

    if is_side_lying:
        left_visibility = left_ear_conf + left_shoulder_conf
        right_visibility = right_ear_conf + right_shoulder_conf

        if right_visibility > left_visibility + 0.2:
            return "Left-Side Lying"

        elif left_visibility > right_visibility + 0.2:
            return "Right-Side Lying"

        else:
            dist_nose_to_left_ear = calc_distance(nose, left_ear)
            dist_nose_to_right_ear = calc_distance(nose, right_ear)

            if dist_nose_to_left_ear < dist_nose_to_right_ear:
                return "Right-Side Lying"
            else:
                return "Left-Side Lying"

    return "Supine"


# ==============================
# 載入 YOLO 模型
# ==============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")


model = load_model()


# ==============================
# 即時影像處理器
# ==============================
class VideoProcessor:
    def __init__(self, model, capture_interval, alarm_threshold):
        self.model = model
        self.capture_interval = capture_interval
        self.alarm_threshold = alarm_threshold

        self.last_capture_time = 0
        self.last_posture = None
        self.current_posture = "Waiting..."
        self.consecutive_count = 0
        self.is_alarm = False

        self.last_annotated_frame = None
        self.last_capture_frame = None

        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        # 複製原始畫面
        display_img = img.copy()

        # 每隔 N 秒截一張圖做偵測
        if now - self.last_capture_time >= self.capture_interval:
            self.last_capture_time = now

            results = self.model(img, verbose=False)
            posture = judge_posture(results)

            if posture == self.last_posture:
                self.consecutive_count += 1
            else:
                self.consecutive_count = 1
                self.last_posture = posture

            self.current_posture = posture
            self.is_alarm = self.consecutive_count >= self.alarm_threshold

            annotated_img = results[0].plot()

            # 加上文字資訊
            info_text = f"Posture: {self.current_posture} | Count: {self.consecutive_count}"

            cv2.putText(
                annotated_img,
                info_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if self.is_alarm:
                h, w = annotated_img.shape[:2]

                cv2.rectangle(
                    annotated_img,
                    (0, 0),
                    (w, h),
                    (0, 0, 255),
                    15
                )

                cv2.putText(
                    annotated_img,
                    "ALARM: STAYED TOO LONG",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA
                )

            with self.lock:
                self.last_annotated_frame = annotated_img.copy()
                self.last_capture_frame = img.copy()

            display_img = annotated_img

        else:
            # 非截圖偵測時間，仍顯示目前狀態
            info_text = f"Posture: {self.current_posture} | Count: {self.consecutive_count}"

            cv2.putText(
                display_img,
                info_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if self.is_alarm:
                h, w = display_img.shape[:2]

                cv2.rectangle(
                    display_img,
                    (0, 0),
                    (w, h),
                    (0, 0, 255),
                    15
                )

                cv2.putText(
                    display_img,
                    "ALARM: STAYED TOO LONG",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA
                )

        return av.VideoFrame.from_ndarray(display_img, format="bgr24")


# ==============================
# Streamlit 頁面
# ==============================
st.set_page_config(
    page_title="即時病房姿勢監測系統",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 即時病房姿勢監測系統")

st.write(
    "系統會開啟鏡頭進行即時錄影，並每隔一段時間自動截圖，"
    "使用 YOLOv8 Pose 偵測病人姿勢。"
)

st.sidebar.header("⚙️ 系統設定")

capture_interval = st.sidebar.number_input(
    "每隔幾秒截圖偵測一次",
    min_value=1,
    max_value=60,
    value=5,
    step=1
)

alarm_threshold = st.sidebar.number_input(
    "連續幾次相同姿勢後觸發警報",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

st.sidebar.markdown("### 姿勢判斷類別")
st.sidebar.write("- Left-Side Lying：左側躺")
st.sidebar.write("- Right-Side Lying：右側躺")
st.sidebar.write("- Supine：平躺 / 趴睡")
st.sidebar.write("- No Person Detected：沒有偵測到人")


st.info(
    "請按下 START 開啟鏡頭。若瀏覽器詢問相機權限，請選擇允許。"
)

ctx = webrtc_streamer(
    key="realtime-posture-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    video_processor_factory=lambda: VideoProcessor(
        model=model,
        capture_interval=capture_interval,
        alarm_threshold=alarm_threshold
    ),
    async_processing=True
)

st.markdown("---")

st.subheader("📌 使用方式")

st.write(
    """
    1. 按下 START 開啟鏡頭  
    2. 系統會即時顯示病房畫面  
    3. 每隔設定秒數自動截圖並進行姿勢偵測  
    4. 若連續多次偵測為相同姿勢，系統會顯示紅色警報框  
    """
)

st.warning(
    "提醒：如果你修改左側的截圖秒數或警報門檻，建議先 STOP 再 START，讓設定重新生效。"
)
