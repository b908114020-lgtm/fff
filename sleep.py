import math


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

    # 沒有偵測到人體關鍵點
    if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return "No Person Detected"

    keypoints = results[0].keypoints.xy[0]
    confs = results[0].keypoints.conf[0]

    # YOLO Pose 關鍵點至少要有到肩膀與髖部
    if len(keypoints) < 13:
        return "Keypoints Not Enough"

    # ==============================
    # 取得主要關鍵點
    # ==============================
    nose = keypoints[0]

    left_ear = keypoints[3]
    right_ear = keypoints[4]

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    left_hip = keypoints[11]
    right_hip = keypoints[12]

    # ==============================
    # 取得信心度
    # ==============================
    left_ear_conf = confs[3]
    right_ear_conf = confs[4]

    left_shoulder_conf = confs[5]
    right_shoulder_conf = confs[6]

    # ==============================
    # 計算身體比例
    # ==============================
    shoulder_width = calc_distance(left_shoulder, right_shoulder)

    torso_length = (
        calc_distance(left_shoulder, left_hip) +
        calc_distance(right_shoulder, right_hip)
    ) / 2

    # ==============================
    # 判斷是否為側躺
    # ==============================
    is_side_lying = False

    # 條件 1：肩膀信心度太低，可能被身體遮擋
    if left_shoulder_conf < 0.4 or right_shoulder_conf < 0.4:
        is_side_lying = True

    # 條件 2：肩膀寬度相對軀幹長度太小，代表雙肩可能重疊
    elif torso_length > 5 and shoulder_width / torso_length < 0.5:
        is_side_lying = True

    # ==============================
    # 如果是側躺，進一步判斷左側躺或右側躺
    # ==============================
    if is_side_lying:
        left_visibility = left_ear_conf + left_shoulder_conf
        right_visibility = right_ear_conf + right_shoulder_conf

        # 如果右耳、右肩比較清楚，代表病人可能是左側躺
        if right_visibility > left_visibility + 0.2:
            return "Left-Side Lying"

        # 如果左耳、左肩比較清楚，代表病人可能是右側躺
        elif left_visibility > right_visibility + 0.2:
            return "Right-Side Lying"

        # 如果左右信心度差不多，用鼻子到耳朵距離輔助判斷
        else:
            dist_nose_to_left_ear = calc_distance(nose, left_ear)
            dist_nose_to_right_ear = calc_distance(nose, right_ear)

            if dist_nose_to_left_ear < dist_nose_to_right_ear:
                return "Right-Side Lying"
            else:
                return "Left-Side Lying"

    # ==============================
    # 不是側躺，先歸類為平躺 / 趴睡
    # ==============================
    return "Supine"
