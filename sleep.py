import cv2
import math
import os
import glob
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
from IPython.display import clear_output

# ==========================================
# 1. 解壓縮圖片檔案
# ==========================================
zip_path = "/content/躺姿-20260502T030636Z-3-001.zip"
extract_path = "/content/pose_images"

if os.path.exists(zip_path):
    print(f"📦 正在解壓縮檔案: {zip_path}...")
    # -q 為安靜模式，-o 為強制覆蓋，-d 指定目標資料夾
    !unzip -q -o "{zip_path}" -d "{extract_path}"
    print("✅ 解壓縮完成！")
else:
    print(f"❌ 錯誤：找不到檔案 {zip_path}，請確認檔案已上傳。")

# ==========================================
# 2. 初始化 YOLOv8 模型與設定路徑
# ==========================================
model = YOLO("yolov8n-pose.pt")

# 自動尋找解壓縮後的資料夾（處理可能存在的子資料夾）
image_files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    # 使用 recursive=True 來搜尋可能藏在子目錄裡的圖片
    image_files.extend(glob.glob(os.path.join(extract_path, "**", ext), recursive=True))

if not image_files:
    print(f"❌ 錯誤：在解壓後的路徑中找不到任何圖片！")
else:
    print(f"📂 找到 {len(image_files)} 張圖片，開始進行側躺判定...\n")

    # 統計數據
    side_lying_count = 0
    supine_count = 0

    # ==========================================
    # 3. 開始遍歷判讀
    # ==========================================
    for img_path in sorted(image_files):
        image = cv2.imread(img_path)
        if image is None:
            continue

        # 執行 YOLO 辨識
        results = model(image, verbose=False)
        status = "偵測失敗"
        color = (255, 255, 255) # 白色

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0]
            confs = results[0].keypoints.conf[0]

            # 確保關鍵點數量足夠（至少到髖部）
            if len(keypoints) >= 13:
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                left_hip = keypoints[11]
                right_hip = keypoints[12]

                ls_conf = confs[5]
                rs_conf = confs[6]

                def calc_distance(p1, p2):
                    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                # 肩膀寬度 vs 軀幹長度
                shoulder_width = calc_distance(left_shoulder, right_shoulder)
                torso_length = (calc_distance(left_shoulder, left_hip) + calc_distance(right_shoulder, right_hip)) / 2

                # 判斷邏輯
                if ls_conf < 0.4 or rs_conf < 0.4:
                    status = "Side-Lying (Shoulder Blocked)"
                    side_lying_count += 1
                    color = (0, 255, 255) # 黃色
                elif torso_length > 5:
                    ratio = shoulder_width / torso_length
                    if ratio < 0.5:
                        status = f"Side-Lying (Ratio: {ratio:.2f})"
                        side_lying_count += 1
                        color = (0, 255, 0) # 綠色
                    else:
                        status = f"Supine/Prone (Ratio: {ratio:.2f})"
                        supine_count += 1
                        color = (255, 0, 0) # 藍色

        # 4. 繪製結果文字並顯示
        annotated_image = results[0].plot()
        filename = os.path.basename(img_path)

        # 在圖上標註結果
        cv2.putText(annotated_image, f"{status}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        # 縮小圖片方便預覽
        h, w = annotated_image.shape[:2]
        display_w = 600
        display_h = int(h * (display_w / w))
        small_img = cv2.resize(annotated_image, (display_w, display_h))

        print(f"📄 處理中: {filename}")
        cv2_imshow(small_img)
        print("-" * 60)

    # ==========================================
    # 5. 輸出最終統計
    # ==========================================
    print("\n" + "★"*30)
    print(f"📊 分析報告")
    print(f"▶ 總處理圖片數: {len(image_files)}")
    print(f"▶ 側躺 (Side-Lying): {side_lying_count} 張")
    print(f"▶ 平躺/趴睡 (Supine): {supine_count} 張")
    print("★"*30)
  import cv2
import math
import os
import glob
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
from IPython.display import clear_output

# ==========================================
# 1. 檔案解壓縮 (若已解壓過會自動覆蓋)
# ==========================================
zip_path = "/content/躺姿-20260502T030636Z-3-001.zip"
extract_path = "/content/pose_images"

if os.path.exists(zip_path):
    !unzip -q -o "{zip_path}" -d "{extract_path}"
else:
    print(f"⚠️ 找不到壓縮檔 {zip_path}，將直接嘗試讀取 {extract_path}")

# ==========================================
# 2. 初始化 YOLOv8 模型與準備圖片
# ==========================================
model = YOLO("yolov8n-pose.pt")

image_files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    image_files.extend(glob.glob(os.path.join(extract_path, "**", ext), recursive=True))

if not image_files:
    print(f"❌ 錯誤：在路徑中找不到圖片！")
else:
    print(f"📂 找到 {len(image_files)} 張圖片，開始進行左右側躺判定...\n")

    # 統計數據
    left_side_count = 0
    right_side_count = 0
    supine_count = 0

    # ==========================================
    # 3. 開始遍歷判讀
    # ==========================================
    for img_path in sorted(image_files):
        image = cv2.imread(img_path)
        if image is None:
            continue

        results = model(image, verbose=False)
        status = "偵測失敗"
        color = (255, 255, 255)

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0]
            confs = results[0].keypoints.conf[0]

            if len(keypoints) >= 13:
                # 取得關鍵點座標
                nose = keypoints[0]
                left_ear = keypoints[3]
                right_ear = keypoints[4]
                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]
                left_hip = keypoints[11]
                right_hip = keypoints[12]

                # 取得信心度 (Confidence)
                le_conf = confs[3] # 左耳
                re_conf = confs[4] # 右耳
                ls_conf = confs[5] # 左肩
                rs_conf = confs[6] # 右肩

                def calc_distance(p1, p2):
                    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

                # 計算基礎特徵
                shoulder_width = calc_distance(left_shoulder, right_shoulder)
                torso_length = (calc_distance(left_shoulder, left_hip) + calc_distance(right_shoulder, right_hip)) / 2

                is_side_lying = False

                # 條件 1：如果有任一側肩膀被嚴重遮擋
                if ls_conf < 0.4 or rs_conf < 0.4:
                    is_side_lying = True
                # 條件 2：肩寬與軀幹長度比例過小 (雙肩在畫面上重疊)
                elif torso_length > 5 and (shoulder_width / torso_length) < 0.5:
                    is_side_lying = True

                # 🔥 判斷左右側躺邏輯
                # 🔥 判斷左右側躺邏輯
                if is_side_lying:
                    left_visibility = le_conf + ls_conf
                    right_visibility = re_conf + rs_conf

                    if right_visibility > left_visibility + 0.2:
                        status = "Left-Side Lying"  # 把中文拿掉
                        left_side_count += 1
                        color = (0, 165, 255)

                    elif left_visibility > right_visibility + 0.2:
                        status = "Right-Side Lying" # 把中文拿掉
                        right_side_count += 1
                        color = (0, 255, 0)

                    else:
                        dist_nose_to_le = calc_distance(nose, left_ear)
                        dist_nose_to_re = calc_distance(nose, right_ear)

                        if dist_nose_to_le < dist_nose_to_re:
                            status = "Right-Side Lying" # 把中文拿掉
                            right_side_count += 1
                            color = (0, 255, 0)
                        else:
                            status = "Left-Side Lying"  # 把中文拿掉
                            left_side_count += 1
                            color = (0, 165, 255)
                else:
                    ratio = shoulder_width / torso_length if torso_length > 0 else 0
                    status = f"Supine (Ratio: {ratio:.2f})" # 把中文拿掉
                    supine_count += 1
                    color = (255, 0, 0)

        # 4. 繪製結果文字並顯示
        annotated_image = results[0].plot()
        filename = os.path.basename(img_path)

        cv2.putText(annotated_image, f"{status}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

        h, w = annotated_image.shape[:2]
        display_w = 600
        display_h = int(h * (display_w / w))
        small_img = cv2.resize(annotated_image, (display_w, display_h))

        print(f"📄 檔案: {filename}")
        cv2_imshow(small_img)
        print("-" * 60)

    # ==========================================
    # 5. 輸出最終統計
    # ==========================================
    print("\n" + "★"*30)
    print(f"📊 分析報告")
    print(f"▶ 總處理圖片數: {len(image_files)}")
    print(f"▶ 左側躺 (Left-Side): {left_side_count} 張")
    print(f"▶ 右側躺 (Right-Side): {right_side_count} 張")
    print(f"▶ 平躺/趴睡 (Supine): {supine_count} 張")
    print("★"*30)
  import cv2
import math
import os
import glob
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# 1. 初始化模型與路徑
model = YOLO("yolov8n-pose.pt")
extract_path = "/content/pose_images"

image_files = sorted(glob.glob(os.path.join(extract_path, "**", "*.jpg"), recursive=True))

# ==========================================
# 2. 建立追蹤變數
# ==========================================
last_posture = None      # 儲存上一次的姿勢名稱
consecutive_count = 0    # 連續計數
ALARM_THRESHOLD = 5      # 設定連續 5 張報警

if not image_files:
    print("❌ 找不到圖片！")
else:
    print(f"🔔 開始監測，連續 {ALARM_THRESHOLD} 張相同姿勢將觸發警報...\n")

    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None: continue

        results = model(image, verbose=False)
        current_posture = "Unknown" # 預設姿勢

        # 骨架分析邏輯 (精簡版)
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kps = results[0].keypoints.xy[0]
            conf = results[0].keypoints.conf[0]

            if len(kps) >= 13:
                # 取得關鍵點 (0:鼻, 3:左耳, 4:右耳, 5:左肩, 6:右肩, 11:左髖, 12:右髖)
                ls_conf, rs_conf = conf[5], conf[6]
                le_conf, re_conf = conf[3], conf[4]

                def dist(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

                s_width = dist(kps[5], kps[6])
                t_len = (dist(kps[5], kps[11]) + dist(kps[6], kps[12])) / 2

                # 判定基礎姿勢
                is_side = (ls_conf < 0.4 or rs_conf < 0.4) or (t_len > 0 and (s_width / t_len) < 0.5)

                if is_side:
                    # 判定左右側 (加上臉部可見度判定)
                    if (re_conf + rs_conf) > (le_conf + ls_conf) + 0.2:
                        current_posture = "Left-Side"
                    elif (le_conf + ls_conf) > (re_conf + rs_conf) + 0.2:
                        current_posture = "Right-Side"
                    else:
                        current_posture = "Right-Side" if dist(kps[0], kps[3]) < dist(kps[0], kps[4]) else "Left-Side"
                else:
                    current_posture = "Supine"

        # ==========================================
        # 3. 連續姿勢判定邏輯
        # ==========================================
        if current_posture == last_posture:
            consecutive_count += 1
        else:
            consecutive_count = 1
            last_posture = current_posture

        # ==========================================
        # 4. 繪製結果與報警
        # ==========================================
        annotated_img = results[0].plot()
        display_color = (255, 0, 0) # 預設藍色 (OpenCV BGR)

        # 顯示當前資訊
        info_text = f"Pos: {current_posture} | Count: {consecutive_count}"
        cv2.putText(annotated_img, info_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 觸發警報
        if consecutive_count >= ALARM_THRESHOLD:
            # 畫上顯眼的紅色大框框與警報字樣
            cv2.rectangle(annotated_img, (0,0), (annotated_img.shape[1], annotated_img.shape[0]), (0, 0, 255), 15)
            cv2.putText(annotated_img, "!!! ALARM: STAYED TOO LONG !!!", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            print(f"⚠️ 警報觸發！檔案 {os.path.basename(img_path)} 已連續 {consecutive_count} 張維持 {current_posture}")

        # 顯示圖片
        h, w = annotated_img.shape[:2]
        small_img = cv2.resize(annotated_img, (600, int(h * 600 / w)))
        cv2_imshow(small_img)
        print("-" * 50)
      import cv2
import math
import os
import glob
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# 1. 初始化模型與路徑
model = YOLO("yolov8n-pose.pt")
extract_path = "/content/pose_images"

image_files = sorted(glob.glob(os.path.join(extract_path, "**", "*.jpg"), recursive=True))

# ==========================================
# 2. 建立追蹤變數
# ==========================================
last_posture = None      # 儲存上一次的姿勢名稱
consecutive_count = 0    # 連續計數
ALARM_THRESHOLD = 5      # 設定連續 5 張報警

if not image_files:
    print("❌ 找不到圖片！")
else:
    print(f"🔔 開始監測，連續 {ALARM_THRESHOLD} 張相同姿勢將觸發警報...\n")

    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None: continue

        results = model(image, verbose=False)
        current_posture = "Unknown" # 預設姿勢

        # 骨架分析邏輯 (精簡版)
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kps = results[0].keypoints.xy[0]
            conf = results[0].keypoints.conf[0]

            if len(kps) >= 13:
                # 取得關鍵點 (0:鼻, 3:左耳, 4:右耳, 5:左肩, 6:右肩, 11:左髖, 12:右髖)
                ls_conf, rs_conf = conf[5], conf[6]
                le_conf, re_conf = conf[3], conf[4]

                def dist(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

                s_width = dist(kps[5], kps[6])
                t_len = (dist(kps[5], kps[11]) + dist(kps[6], kps[12])) / 2

                # 判定基礎姿勢
                is_side = (ls_conf < 0.4 or rs_conf < 0.4) or (t_len > 0 and (s_width / t_len) < 0.5)

                if is_side:
                    # 判定左右側 (加上臉部可見度判定)
                    if (re_conf + rs_conf) > (le_conf + ls_conf) + 0.2:
                        current_posture = "Left-Side"
                    elif (le_conf + ls_conf) > (re_conf + rs_conf) + 0.2:
                        current_posture = "Right-Side"
                    else:
                        current_posture = "Right-Side" if dist(kps[0], kps[3]) < dist(kps[0], kps[4]) else "Left-Side"
                else:
                    current_posture = "Supine"

        # ==========================================
        # 3. 連續姿勢判定邏輯
        # ==========================================
        if current_posture == last_posture:
            consecutive_count += 1
        else:
            consecutive_count = 1
            last_posture = current_posture

        # ==========================================
        # 4. 繪製結果與報警
        # ==========================================
        annotated_img = results[0].plot()
        display_color = (255, 0, 0) # 預設藍色 (OpenCV BGR)

        # 顯示當前資訊
        info_text = f"Pos: {current_posture} | Count: {consecutive_count}"
        cv2.putText(annotated_img, info_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 觸發警報
        if consecutive_count >= ALARM_THRESHOLD:
            # 畫上顯眼的紅色大框框與警報字樣
            cv2.rectangle(annotated_img, (0,0), (annotated_img.shape[1], annotated_img.shape[0]), (0, 0, 255), 15)
            cv2.putText(annotated_img, "!!! ALARM: STAYED TOO LONG !!!", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            print(f"⚠️ 警報觸發！檔案 {os.path.basename(img_path)} 已連續 {consecutive_count} 張維持 {current_posture}")

        # 顯示圖片
        h, w = annotated_img.shape[:2]
        small_img = cv2.resize(annotated_img, (600, int(h * 600 / w)))
        cv2_imshow(small_img)
        print("-" * 50)
