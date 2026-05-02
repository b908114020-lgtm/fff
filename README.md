# fff
# 病房姿勢監測系統

本專案使用 YOLOv8 Pose 偵測病房影像中的人體關鍵點，並根據肩膀、髖部、耳朵等關鍵點，判斷病人目前姿勢是否為左側躺、右側躺或平躺。

此系統可作為智慧病房、壓瘡風險提醒、病人姿勢監測等應用的初步原型。

## 功能特色

- 上傳病房圖片
- 使用 YOLOv8 Pose 偵測人體骨架
- 判斷姿勢：
  - Left-Side Lying：左側躺
  - Right-Side Lying：右側躺
  - Supine：平躺 / 趴睡
- 計算連續相同姿勢次數
- 達到設定門檻時顯示警報
- 顯示骨架偵測後的圖片

## 專案結構

```text
pressure_injury_posture_monitor/
├── app.py
├── posture_detector.py
├── requirements.txt
└── README.md
