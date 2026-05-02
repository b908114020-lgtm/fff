import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import math


# ==============================
# 姿勢判斷函式
# ==============================
def calc_distance(p1, p2):
    """
    計算兩個關鍵點之間的距離
    """
    return math.sqrt(
        (float(p1[0]) - float(p2[0])) ** 2 +
        (float(p1[1]) - float(p2[1])) ** 2
    )


def judge_posture(results):
    """
    根據 YOLOv8 Pose 的人體關鍵點判斷姿勢。

    回傳：
    - Left-Side Lying
    - Right-Side Lying
    - Supine
    - No Person Detected
    - Keypoints Not Enough
    """

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
        calc_distance(left_shoulder, left_hip) +
        calc_distance(right_shoulder, right_hip)
    ) / 2

    is_side_lying = False

    # 條件 1：肩膀信心度太低，可能被身體遮擋
    if left_shoulder_conf < 0.4 or right_shoulder_conf < 0.4:
        is_side_lying = True

    # 條件 2：肩寬 / 軀幹長度比例太小，代表雙肩可能重疊
    elif torso_length > 5 and shoulder_width / torso_length < 0.5:
        is_side_lying = True

    if is_side_lying:
        left_visibility = left_ear_conf + left_shoulder_conf
        right_visibility = right_ear_conf + right_shoulder_conf

        # 右側特徵比較清楚，推測左側躺
        if right_visibility > left_visibility + 0.2:
            return "Left-Side Lying"

        # 左側特徵比較清楚，推測右側躺
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
# 單張圖片分析函式
# ==============================
def analyze_image(image, model, alarm_threshold):
    """
    輸入 PIL Image，輸出分析結果、標註圖片、是否警報
    """

    image = image.convert("RGB")
    image_np = np.array(image)

    # YOLO 推論
    results = model(image_np, verbose=False)

    # 判斷姿勢
    current_posture = judge_posture(results)

    # 連續姿勢計算
    if current_posture == st.session_state.last_posture:
        st.session_state.consecutive_count += 1
    else:
        st.session_state.consecutive_count = 1
        st.session_state.last_posture = current_posture

    # 產生骨架標註圖片
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # 在圖片上加文字
    info_text = (
        f"Posture: {current_posture} | "
        f"Count: {st.session_state.consecutive_count}"
    )

    cv2.putText(
        annotated_image,
        info_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # 判斷是否觸發警報
    is_alarm = st.session_state.consecutive_count >= alarm_threshold

    if is_alarm:
        h, w = annotated_image.shape[:2]

        cv2.rectangle(
            annotated_image,
            (0, 0),
            (w, h),
            (255, 0, 0),
            15
        )

        cv2.putText(
            annotated_image,
            "ALARM: STAYED TOO LONG",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            4,
            cv2.LINE_AA
        )

    return current_posture, annotated_image, is_alarm


# ==============================
# 顯示分析結果函式
# ==============================
def show_result(current_posture, annotated_image, is_alarm):
    """
    在 Streamlit 頁面顯示分析結果
    """

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
        st.error(
            "⚠️ 警報：病人已連續多張圖片維持相同姿勢，"
            "建議協助翻身或確認狀況。"
        )
    else:
        st.info("目前尚未達到警報條件。")

    st.image(
        annotated_image,
        caption="YOLOv8 Pose 姿勢偵測結果",
        use_container_width=True
    )


# ==============================
# Streamlit 頁面設定
# ==============================
st.set_page_config(
    page_title="病房姿勢監測系統",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 病房姿勢監測系統")

st.write(
    "本系統使用 YOLOv8 Pose 偵測病人姿勢，"
    "可透過上傳圖片或使用鏡頭拍攝，判斷病人目前為左側躺、右側躺或平躺，"
    "協助觀察是否長時間維持同一姿勢。"
)


# ==============================
# 側邊欄設定
# ==============================
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

st.sidebar.markdown("### 使用提醒")
st.sidebar.write("鏡頭模式會使用瀏覽器相機權限。")
st.sidebar.write("若部署在 Streamlit Cloud，請確認瀏覽器允許相機權限。")


# ==============================
# 載入 YOLO 模型
# ==============================
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")


model = load_model()


# ==============================
# 初始化 Session State
# ==============================
if "last_posture" not in st.session_state:
    st.session_state.last_posture = None

if "consecutive_count" not in st.session_state:
    st.session_state.consecutive_count = 0


# ==============================
# 選擇輸入方式
# ==============================
mode = st.radio(
    "請選擇影像輸入方式",
    ["上傳圖片", "使用鏡頭拍攝"],
    horizontal=True
)


# ==============================
# 上傳圖片模式
# ==============================
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

            show_result(
                current_posture=current_posture,
                annotated_image=annotated_image,
                is_alarm=is_alarm
            )

    else:
        st.info("請先上傳圖片，系統會開始進行姿勢分析。")


# ==============================
# 鏡頭拍攝模式
# ==============================
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

        show_result(
            current_posture=current_posture,
            annotated_image=annotated_image,
            is_alarm=is_alarm
        )

    else:
        st.info("請先開啟鏡頭並拍攝一張照片。")


# ==============================
# 重置按鈕
# ==============================
st.markdown("---")

if st.button("🔄 重置連續姿勢計數"):
    st.session_state.last_posture = None
    st.session_state.consecutive_count = 0
    st.success("已重置連續姿勢計數。")
