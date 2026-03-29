import cv2
import numpy as np
import threading
import time
import os
import customtkinter as ctk
from PIL import Image, ImageTk
from ultralytics import YOLO

# UI 테마 설정
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class WebcamApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 메인 윈도우 설정
        self.title("Vision-Doctor: Live Webcam Monitor")
        self.geometry("1000x700")
        
        # 상태 변수
        self.running = False
        self.cap = None
        self.model = None
        self.camera_thread = None

        self._setup_ui()
        self._load_model_thread()

    def _setup_ui(self):
        # 레이아웃 분할 (좌측 7 : 우측 3)
        self.grid_columnconfigure(0, weight=7)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # -------------------------------------
        # 1. 좌측 화면: 비디오 출력 뷰
        # -------------------------------------
        self.video_frame = ctk.CTkFrame(self, corner_radius=15, border_width=2, border_color="#333333")
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.header_label = ctk.CTkLabel(self.video_frame, text="Live Camera Feed", font=("Pretendard", 22, "bold"))
        self.header_label.pack(pady=(15, 0))

        self.video_label = ctk.CTkLabel(self.video_frame, text="SYSTEM IDLE\n\n버튼을 눌러 모니터링을 시작하세요", 
                                        text_color="gray", font=("Pretendard", 20))
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # -------------------------------------
        # 2. 우측 화면: 컨트롤 보드 및 로그
        # -------------------------------------
        self.right_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        # 컨트롤 패널 (버튼부)
        self.control_box = ctk.CTkFrame(self.right_frame, corner_radius=15)
        self.control_box.pack(fill="x", pady=(0, 20), ipady=10)
        
        # 시작 버튼
        self.start_btn = ctk.CTkButton(self.control_box, text="▶ 시스템 시작 (Start)", font=("Pretendard", 16, "bold"),
                                       height=45, fg_color="#228B22", hover_color="#006400", command=self.start_camera)
        self.start_btn.pack(fill="x", padx=20, pady=(20, 10))
        
        # 정지 버튼
        self.stop_btn = ctk.CTkButton(self.control_box, text="⏹ 시스템 정지 (Stop)", font=("Pretendard", 16, "bold"),
                                      height=45, fg_color="#B22222", hover_color="#8B0000", state="disabled", command=self.stop_camera)
        self.stop_btn.pack(fill="x", padx=20, pady=10)

        # 프로그램 종료 버튼
        self.exit_btn = ctk.CTkButton(self.control_box, text="❌ 프로그램 종료 (Exit)", font=("Pretendard", 16, "bold"),
                                      height=45, fg_color="#333333", hover_color="#111111", command=self.exit_app)
        self.exit_btn.pack(fill="x", padx=20, pady=(10, 20))

        # 이벤트 로그 패널
        self.log_frame = ctk.CTkFrame(self.right_frame, corner_radius=15)
        self.log_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(self.log_frame, text="📋 모니터링 경고 로그", font=("Pretendard", 16, "bold")).pack(pady=10)
        
        self.log_text = ctk.CTkTextbox(self.log_frame, font=("Pretendard", 13), activate_scrollbars=True)
        self.log_text.pack(padx=15, pady=(0, 15), fill="both", expand=True)
        
        self.log_message("[System] UI 렌더링 완료. 모델 로딩을 기다리고 있습니다...")

    def _load_model_thread(self):
        # UI가 얼어붙는 현상을 막기 위해 AI 모델은 백그라운드에서 로딩
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        self.log_message("[System] AI 모델(YOLO)을 불러오고 있습니다...")
        weights_path = "runs/detect/baseline_test/weights/best.pt"
        if not os.path.exists(weights_path):
            weights_path = "runs/detect/runs/detect/baseline_test/weights/best.pt"
            
        try:
            self.model = YOLO(weights_path)
            self.log_message(f"[System] AI 모델 로드 완료: {os.path.basename(weights_path)}")
        except Exception:
            # 커스텀 모델이 없거나 손상되었을 경우 기본 모델 임시 지원
            self.model = YOLO("yolov8n.pt")
            self.log_message(f"[System] 👉 원본 모델 오류로 기본 모델(yolov8n.pt)을 로드했습니다.")
            self.log_message("[System] 모니터링을 시작할 준비가 되었습니다.")

    def log_message(self, msg):
        """ 스레드 안전하게 로그 텍스트를 화면에 삽입합니다 """
        timestamp = time.strftime("[%H:%M:%S]")
        full_msg = f"{timestamp} {msg}\n"
        self.after(0, self._insert_log, full_msg)

    def _insert_log(self, msg):
        self.log_text.insert("end", msg)
        self.log_text.yview_moveto(1.0) # 로그가 쌓이면 스크롤을 맨 밑으로 유지
        
    def start_camera(self):
        """ 카메라 추론 시작 (시작 버튼 동작) """
        if self.running:
            return
            
        if self.model is None:
            self.log_message("[Error] AI 모델이 아직 로드되지 않았습니다. 잠시만 기다려주세요.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log_message("[Error] 웹캠 장치와 연결할 수 없습니다!")
            return

        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.video_frame.configure(border_color="#00FF00")
        
        self.log_message("[Camera] ✅ 라이브 피드 전송 시작! 화면 추적을 구동합니다.")
        
        # 무거운 객체 추적 루프를 쓰레드로 격리하여 UI 렉 방지
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def stop_camera(self):
        """ 카메라 추론 정지 (정지 버튼 동작) """
        if not self.running:
            return
            
        self.running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.video_frame.configure(border_color="#333333")
        self.log_message("[Camera] ⏹ 사용자에 의해 피드 전송이 중지되었습니다.")
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # 비디오 화면 초기화
        self.video_label.configure(image="", text="SYSTEM IDLE\n\n대기 중입니다.", text_color="gray")

    def exit_app(self):
        """ 메모리를 안전하게 닫으며 프로그램 종료 (종료 버튼 동작) """
        self.running = False
        if self.cap:
            self.cap.release()
        self.quit()
        self.destroy()

    def camera_loop(self):
        """
        [핵심 로직] OpenCV 광학 흐름(RANSAC) & YOLO 방어 추적 루프 
        """
        DARKNESS_THRESHOLD = 40  
        BLUR_THRESHOLD = 100     
        MOVE_THRESHOLD = 5.0     
        
        prev_gray = None
        turn_alert_counter = 0   
        last_move_dir = "UNKNOWN"
        alert_state_logged = False # 로그 중복 도배 방지용

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: 
                break

            # 1. 흑백 변환 후 가려짐/어두워짐 판단
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_covered = (np.mean(gray) < DARKNESS_THRESHOLD) or (cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD)
            is_turned = False

            # 2. 강건한 광학 흐름 & RANSAC을 이용한 움직임/가려짐 판단
            if prev_gray is not None and not is_covered:
                diff = cv2.absdiff(prev_gray, gray)
                change_ratio = np.count_nonzero(diff > 30) / (gray.shape[0] * gray.shape[1])
                
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=150, qualityLevel=0.01, minDistance=15)
                is_camera_moved = False
                
                if prev_pts is not None and len(prev_pts) > 10:
                    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                    idx = np.where(status == 1)[0]
                    good_prev = prev_pts[idx]
                    good_curr = curr_pts[idx]
                    
                    if len(good_prev) > 10:
                        transform, inliers = cv2.estimateAffinePartial2D(
                            good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0
                        )
                        
                        if transform is not None and inliers is not None:
                            inlier_ratio = np.sum(inliers) / len(good_prev)
                            dx = transform[0, 2]
                            dy = transform[1, 2]
                            move_distance = np.sqrt(dx**2 + dy**2)
                            
                            # 거리가 임계값 이상 움직였을 경우 방향 계산
                            if move_distance > MOVE_THRESHOLD:
                                if inlier_ratio >= 0.7:
                                    is_camera_moved = True
                                    turn_alert_counter = 20
                                    if abs(dx) > abs(dy):
                                        last_move_dir = "LEFT" if dx > 0 else "RIGHT"
                                    else:
                                        last_move_dir = "UP" if dy > 0 else "DOWN"
                                else:
                                    # 큰 화면이 일관성 없이 크게 뒤틀림 = 전체를 덮는 무언가 등장
                                    is_covered = True
                
                # 움직임은 없지만 한 번에 화면 픽셀이 40% 이상 변했어도 가려짐 처리
                if not is_camera_moved and change_ratio > 0.4:
                    is_covered = True
                    
            if is_covered:
                turn_alert_counter = 0
                is_turned = False
            elif turn_alert_counter > 0:
                is_turned = True
                turn_alert_counter -= 1
                
            prev_gray = gray.copy()

            # 3. OpenCV 시각적 경고 렌더링 및 UI 연동
            if is_covered:
                if not alert_state_logged:
                    self.log_message("🚨 [위험] 장애물에 의해 렌즈가 심하게 가려지거나 어두워졌습니다!")
                    alert_state_logged = True
                    # 메인 프레임 테두리를 빨간색으로 점등
                    self.after(0, lambda: self.video_frame.configure(border_color="#FF0000"))
                    
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, "ALERT: Camera Blocked!", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                overlay = annotated_frame.copy()
                overlay[:] = (0, 0, 255)
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            
            elif is_turned:
                if not alert_state_logged:
                    self.log_message(f"⚠️ [주의] 카메라 자체 물리적 이동 감지: [{last_move_dir}]")
                    alert_state_logged = True
                    # 메인 프레임 테두리를 노란색으로 점등
                    self.after(0, lambda: self.video_frame.configure(border_color="#FFFF00"))

                results = self.model(frame, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()
                
                alert_msg = f"WARNING! CAMERA MOVED: [{last_move_dir}]"
                cv2.putText(annotated_frame, alert_msg, (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
                overlay = annotated_frame.copy()
                overlay[:] = (0, 255, 255)
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            
            else:
                if alert_state_logged:
                    self.log_message("✅ [정상] 시야가 회복되어 정상 모니터링을 재개합니다.")
                    alert_state_logged = False
                    # 메인 프레임 테두리를 초록색으로 복구
                    self.after(0, lambda: self.video_frame.configure(border_color="#00FF00"))

                results = self.model(frame, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()

            # 4. CustomTkinter UI 뷰포트로 이미지 밀어넣기 처리
            color_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(color_frame)
            
            # 해상도를 일반적인 UI창 표준 사이즈에 맞게 조절
            ctk_img = ctk.CTkImage(pil_img, size=(640, 480))
            self.after(0, self._update_image, ctk_img)
            
            time.sleep(0.01) # CPU 연산 부하를 약간 덜어주기 위함

    def _update_image(self, img):
        if self.running:
            self.video_label.configure(image=img, text="")

if __name__ == '__main__':
    app = WebcamApp()
    app.mainloop()
