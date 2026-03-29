"""
[실행 방법]
1. VS Code 터미널을 엽니다.
2. 아래 명령어를 입력하여 실행합니다.
   python webcam_inference.py
   
[기능 요약]
- 노트북 웹캠을 켜서 소(Cow) 객체를 실시간으로 탐지합니다.
- 만약 카메라 렌즈 앞이 가려지거나 너무 어두워지면 화면 전체에 붉은 경고가 뜹니다.
- 카메라가 상하좌우로 휙 움직이면 해당 이동 방향을 감지하여 노란색 경고가 뜨게 됩니다.
- 실행 중 창을 클릭하고 키보드 'q'를 누르면 안전하게 종료됩니다.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def run_webcam():
    print("[*] 모델 로딩 및 카메라 연결 중...")
    
    # 모델 로드 
    weights_path = "runs/detect/baseline_test/weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = "runs/detect/runs/detect/baseline_test/weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = "yolo26n.pt"
        
    try:
        model = YOLO(weights_path)
    except Exception:
        model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] 웹캠을 열 수 없습니다.")
        return

    print("[*] 모니터링 시작! (종료: 'q' 키 입력)")

    # [임계값 세팅]
    DARKNESS_THRESHOLD = 40  
    BLUR_THRESHOLD = 100     
    TURN_THRESHOLD = 0.3     # 큰 충격(비율 변화) 감지
    MOVE_THRESHOLD = 5.0     # 픽셀 단위로 상하좌우를 구별할 때 감도 (낮을수록 더 작은 움직임 감지)

    prev_gray = None
    turn_alert_counter = 0   
    last_move_dir = "UNKNOWN"

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # [A] 카메라 가려짐 감지
        is_covered = (np.mean(gray) < DARKNESS_THRESHOLD) or (cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD)

        # [B] 상하좌우 방향 이동 감지 (위상 상관 기법: Phase Correlation 적용)
        is_turned = False
        if prev_gray is not None and not is_covered:
            
            # 1. 픽셀 변화 비율 계산 (전체적으로 급격히 변했을 때 대비)
            diff = cv2.absdiff(prev_gray, gray)
            change_ratio = np.count_nonzero(diff > 30) / (gray.shape[0] * gray.shape[1])
            
            # 2. 광학 흐름(Optical Flow)과 RANSAC 알고리즘을 이용한 강건한 카메라 이동 감지
            # 화면 내 추적하기 좋은 특징점(배경 코너 등) 추출
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=150, qualityLevel=0.01, minDistance=15)
            
            is_camera_moved = False
            
            if prev_pts is not None and len(prev_pts) > 10:
                # 다음 프레임에서 점들이 어디로 이동했는지 추적
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                
                # 정상적으로 추적된 점들만 필터링
                idx = np.where(status == 1)[0]
                good_prev = prev_pts[idx]
                good_curr = curr_pts[idx]
                
                # 살아남은 특징점이 충분히 있을 때만 변환 행렬 계산
                if len(good_prev) > 10:
                    # RANSAC 적용: 전경에서 움직이는 큰 물체의 점들은 Outlier로 무시하고, 
                    # 다수를 차지하는 정지 배경의 점들(Inlier)로만 이동량(dx, dy)을 구함
                    transform, inliers = cv2.estimateAffinePartial2D(
                        good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    
                    if transform is not None and inliers is not None:
                        # RANSAC 결과, 실제로 같은 방향으로 같이 움직인 점(Inlier)의 비율
                        inlier_ratio = np.sum(inliers) / len(good_prev)
                        
                        dx = transform[0, 2]
                        dy = transform[1, 2]
                        move_distance = np.sqrt(dx**2 + dy**2)
                        
                        # 카메라 이동이라고 확신하기 위한 조건: 움직임이 기준치 이상 & 일관성 있는 점이 70% 이상
                        if move_distance > MOVE_THRESHOLD:
                            if inlier_ratio >= 0.7:
                                is_camera_moved = True
                                turn_alert_counter = 20 # 약 1초 동안 알림 유지
                                if abs(dx) > abs(dy):
                                    last_move_dir = "LEFT" if dx > 0 else "RIGHT"
                                else:
                                    last_move_dir = "UP" if dy > 0 else "DOWN"
                            else:
                                # 점들이 크게 이동했지만 일관성이 낮음 == 화면 대부분을 가리는 물체 등장
                                is_covered = True
            
            # 3. 만약 큰 화면 변화(40% 이상)가 발생했다면 물체가 화면을 덮은 것으로 간주
            if not is_camera_moved and change_ratio > 0.4:
                is_covered = True
                
        # 20프레임 동안 이동 경고 카운트 다운 (가려진 상태라면 이동 경고는 무시)
        if is_covered:
            turn_alert_counter = 0
            is_turned = False
        elif turn_alert_counter > 0:
            is_turned = True
            turn_alert_counter -= 1
            
        prev_gray = gray.copy()

        # [C] 화면에 경고 출력
        if is_covered:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "ALERT: Camera Blocked!", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            overlay = annotated_frame.copy()
            overlay[:] = (0, 0, 255)
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            print("🚨 렌즈 가려짐 감지! 🚨")
            
        elif is_turned:
            # 방향에 맞춘 노란색 경고 텍스트
            results = model(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
            
            alert_msg = f"WARNING! CAMERA MOVED: [{last_move_dir}]"
            cv2.putText(annotated_frame, alert_msg, (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            overlay = annotated_frame.copy()
            overlay[:] = (0, 255, 255)
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            print(f"⚠️ 카메라가 [{last_move_dir}] 방향으로 움직임! ⚠️")
            
        else:
            # 정상적인 소 객체 탐지
            results = model(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()

        cv2.imshow("Vision-Doctor Monitor", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_webcam()
