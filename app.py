import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from posture_detector import judge_posture


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
    "可判斷病人目前為左側躺、右側躺或平躺，"
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

st.sidebar.write("姿勢判斷類別：")
st.sidebar.write("- Left-Side Lying：左側躺")
st.sidebar.write("- Right-Side Lying：右側躺")
st.sidebar.write("- Supine：平躺 / 趴睡")


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
# 上傳圖片
# ==============================
uploaded_files = st.file_uploader(
    "請上傳病房圖片，可以一次上傳多張",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# ==============================
# 分析圖片
# ==============================
if uploaded_files:
    st.subheader("📊 分析結果")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # YOLO 推論
        results = model(image_np, verbose=False)

        # 姿勢判斷
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
        info_text = f"Posture: {current_posture} | Count: {st.session_state.consecutive_count}"

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

        # 顯示單張圖片結果
        st.markdown("---")
        st.write(f"### 📄 檔案名稱：{uploaded_file.name}")

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

else:
    st.info("請先上傳圖片，系統會開始進行姿勢分析。")


# ==============================
# 重置按鈕
# ==============================
if st.button("🔄 重置連續姿勢計數"):
    st.session_state.last_posture = None
    st.session_state.consecutive_count = 0
    st.success("已重置連續姿勢計數。")
