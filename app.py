import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import math
import tempfile
import os


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

    left_ear_conf = confs[3]
    right_ear_conf = confs[4]
    left_shoulder_conf = confs[5]
    right_shoulder_conf = confs[6]

    shoulder_width = calc_distance(left_shoulder, right_shoulder)

    torso_length = (
        calc_distance(left_shoulder, left_hip)
        + calc_distance(right_shoulder, right_hip)
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


def draw_status_on_image(image, posture, count, is_alarm):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    text = f"Posture: {posture} | Count: {count}"

    draw.rectangle((10, 10, 620, 70), fill=(0, 0, 0))
    draw.text((25, 30), text, fill=(255, 255, 255))

    if is_alarm:
        w, h = image.size
        border = 12

        for i in range(border):
            draw.rectangle(
                (i, i, w - i - 1, h - i - 1),
                outline=(255, 0, 0)
            )

        draw.rectangle((10, 85, 700, 145), fill=(255, 0, 0))
        draw.text((25, 105), "ALARM: STAYED TOO LONG", fill=(255, 255, 255))

    return image


def analyze_image(image, model, alarm_threshold):
    image = image.convert("RGB")
    image_np = np.array(image)

    results = model(image_np, verbose=False)
    current_posture = judge_posture(results)

    if current_posture == st.session_state.last_posture:
        st.session_state.consecutive_count += 1
    else:
        st.session_state.consecutive_count = 1
        st.session_state.last_posture = current_posture

    is_alarm = st.session_state.consecutive_count >= alarm_threshold

    annotated_np = results[0].plot()
    annotated_image = Image.fromarray(annotated_np)

    annotated_image = draw_status_on_image(
        annotated_image,
        current_posture,
        st.session_state.consecutive_count,
        is_alarm
    )

    return current_posture, annotated_image, is_alarm


def show_result(current_posture, annotated_image, is_alarm):
    if current_posture == "Left-Side Lying":
        st.warning("目前姿勢：左側躺 Left-Side Lying")
    elif current_posture == "Right-Side Lying":
        st.warning("目前姿勢：右側躺 Right-Side Lying")
    elif current_posture == "Supine":
        st.success("目前姿勢：平躺 / 趴睡 Supine")
    elif current_posture == "No Person Detected":
        st.error("偵測結果：沒有偵測到人")
    elif current_posture == "Keypoints Not Enough":
        st.error("偵測結果：人體關鍵點不足")
    else:
        st.error(f"偵測結果：{current_posture}")

    st.write(f"連續相同姿勢次數：{st.session_state.consecutive_count}")

    if is_alarm:
        st.error("⚠️ 警報：病人已連續多張圖片維持相同姿勢，建議協助翻身或確認狀況。")
    else:
        st.info("目前尚未達到警報條件。")

    st.image(
        annotated_image,
        caption="YOLOv8 Pose 姿勢偵測結果",
        use_container_width=True
    )


def extract_frames_from_video(video_path, interval_seconds):
    """
    這個無 OpenCV 版本暫時不支援影片截圖。
    若要使用影片截圖，還是需要 OpenCV 或 imageio。
    """
    return [], 0, 0


st.set_page_config(
    page_title="病房姿勢監測系統",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 病房姿勢監測系統")

st.write(
    "本系統使用 YOLOv8 Pose 偵測病人姿勢，"
    "目前提供上傳圖片與鏡頭拍照分析。"
)

st.sidebar.header("⚙️ 系統設定")

alarm_threshold = st.sidebar.number_input(
    "連續幾張相同姿勢後觸發警報",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

st.sidebar.markdown("### 姿勢判斷類別")
st.sidebar.write("- Left-Side Lying：左側躺")
st.sidebar.write("- Right-Side Lying：右側躺")
st.sidebar.write("- Supine：平躺 / 趴睡")


@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")


model = load_model()

if "last_posture" not in st.session_state:
    st.session_state.last_posture = None

if "consecutive_count" not in st.session_state:
    st.session_state.consecutive_count = 0


mode = st.radio(
    "請選擇影像輸入方式",
    ["上傳圖片", "使用鏡頭拍攝"],
    horizontal=True
)


if mode == "上傳圖片":
    uploaded_files = st.file_uploader(
        "請上傳病房圖片，可以一次上傳多張",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader("📊 上傳圖片分析結果")

        for uploaded_file in uploaded_files:
            st.markdown("---")
            st.write(f"### 📄 檔案名稱：{uploaded_file.name}")

            image = Image.open(uploaded_file)

            current_posture, annotated_image, is_alarm = analyze_image(
                image=image,
                model=model,
                alarm_threshold=alarm_threshold
            )

            show_result(current_posture, annotated_image, is_alarm)

    else:
        st.info("請先上傳圖片，系統會開始進行姿勢分析。")


elif mode == "使用鏡頭拍攝":
    st.subheader("📷 使用鏡頭拍攝")

    camera_image = st.camera_input("請用鏡頭拍攝病房畫面")

    if camera_image is not None:
        st.subheader("📊 鏡頭拍攝分析結果")

        image = Image.open(camera_image)

        current_posture, annotated_image, is_alarm = analyze_image(
            image=image,
            model=model,
            alarm_threshold=alarm_threshold
        )

        show_result(current_posture, annotated_image, is_alarm)

    else:
        st.info("請先開啟鏡頭並拍攝一張照片。")


st.markdown("---")

if st.button("🔄 重置連續姿勢計數"):
    st.session_state.last_posture = None
    st.session_state.consecutive_count = 0
    st.success("已重置連續姿勢計數。")
