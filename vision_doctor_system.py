import os
import random
import time
import cv2
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
from ultralytics import YOLO

# --- 설정 (경로 확인 필수!) ---
MODEL_PATH = r'01_Final_Demo/best.pt'
DATA_VAULT = r'02_Dataset_Vault'
DIRTY_CLASS_ID = 80 # 팀장님 확인 사항: 오염 클래스 인덱스

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class VisionDoctorUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Vision Doctor - Intelligent Monitoring System [SIMULATION Mode]")
        self.geometry("1400x850")

        # 모델 로드 (에러 방지용 가짜 모델 설정)
        try:
            self.model = YOLO(MODEL_PATH)
            print(f"✅ 모델 로드 성공: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ 모델 로드 실패. 경로를 확인하세요: {e}")
            self.model = None

        # UI 레이아웃 설정
        self.grid_columnconfigure(0, weight=7) # 좌측: 이미지 및 분석
        self.grid_columnconfigure(1, weight=3) # 우측: 컨트롤 및 리포트
        self.grid_rowconfigure(0, weight=1)

        # --- [좌측] 메인 분석 영역 ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, border_width=2, border_color="#333333")
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # 이미지 출력 라벨
        self.img_label = ctk.CTkLabel(self.main_frame, text="READY TO SCAN", font=("Pretendard", 24, "bold"), text_color="#555555")
        self.img_label.pack(expand=True, fill="both", padx=15, pady=15)

        # 하단 정보 라벨
        self.info_label = ctk.CTkLabel(self.main_frame, text="Conf: - | Time: - ms", font=("Pretendard", 14), text_color="gray")
        self.info_label.place(relx=0.03, rely=0.94)

        # --- [우측] 제어 및 리포트 영역 ---
        self.side_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.side_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        # 1. 제어 섹션
        self.control_box = ctk.CTkFrame(self.side_frame, corner_radius=15)
        self.control_box.pack(fill="x", pady=(0, 15), ipady=10)
        
        self.scan_btn = ctk.CTkButton(self.control_box, text="RANDOM SCAN", font=("Pretendard", 18, "bold"), 
                                      height=50, command=self.start_scan_thread)
        self.scan_btn.pack(padx=20, pady=15, fill="x")
        
        self.progress = ctk.CTkProgressBar(self.control_box)
        self.progress.pack(padx=20, pady=(0, 10), fill="x")
        self.progress.set(0)

        # 2. 상태 보드
        self.status_board = ctk.CTkLabel(self.side_frame, text="SYSTEM STANDBY", font=("Pretendard", 28, "bold"),
                                         height=100, corner_radius=10, fg_color="#2B2B2B", text_color="white")
        self.status_board.pack(fill="x", pady=15)

        # 3. LLM AI 진단 리포트 (중요!)
        self.report_frame = ctk.CTkFrame(self.side_frame, corner_radius=15)
        self.report_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(self.report_frame, text="📋 AI Diagnosis & Recovery Report [Simulated]", 
                     font=("Pretendard", 15, "bold")).pack(pady=10)
        
        self.report_text = ctk.CTkTextbox(self.report_frame, font=("Pretendard", 13), activate_scrollbars=True)
        self.report_text.pack(padx=15, pady=(0, 15), fill="both", expand=True)
        self.report_text.insert("0.0", "시스템이 가동되었습니다.\n분석 결과에 따라 가상 AI 리포트가 여기에 생성됩니다.")

    def start_scan_thread(self):
        # UI 프리징 방지를 위해 스레드로 실행
        threading.Thread(target=self.run_scan_process, daemon=True).start()

    def run_scan_process(self):
        self.scan_btn.configure(state="disabled")
        self.report_text.delete("0.0", "end")
        
        # 0. 하드웨어 상태 초기화
        self.update_status_board("CHECKING HW...", "#2B2B2B", "#333333")
        self.report_text.insert("0.0", "> [SYSTEM] Checking Hardware status (Simulated Mode)...\n> Camera: VIRTUAL\n> Wiper: STANDBY\n> Jetson: EMULATED")
        
        # 1. 프로그레스 바 애니메이션 (하드웨어 체크 및 이미지 로딩 시뮬레이션)
        for i in range(1, 101):
            time.sleep(0.01)
            self.progress.set(i/100)
        
        # 2. 랜덤 이미지 선택 (Vault 폴더 재귀 탐색)
        all_imgs = []
        if os.path.exists(DATA_VAULT):
            for root, dirs, files in os.walk(DATA_VAULT):
                for file in files:
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        all_imgs.append(os.path.join(root, file))
        
        if not all_imgs or self.model is None:
            self.report_text.insert("end", "\n\n[ERROR] No images found or model not loaded.")
            self.img_label.configure(text="TEST FAILED")
            self.scan_btn.configure(state="normal")
            return

        target_img = random.choice(all_imgs)
        
        # 3. YOLO 추론
        start_time = time.time()
        results = self.model.predict(target_img, conf=0.15, verbose=False)
        end_time = time.time()
        
        inference_time = int((end_time - start_time) * 1000)
        
        # 결과 렌더링
        res_img = results[0].plot(line_width=2, font_size=12)
        res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(res_img_rgb)
        
        # UI에 맞게 리사이즈 (Aspect Ratio 유지)
        img_w, img_h = pil_img.size
        ratio = min(900/img_w, 600/img_h)
        new_size = (int(img_w * ratio), int(img_h * ratio))
        ctk_img = ctk.CTkImage(pil_img, size=new_size)

        # 4. 결과 분석
        is_dirty = any(int(box.cls) == DIRTY_CLASS_ID for box in results[0].boxes)
        max_conf = max([box.conf for box in results[0].boxes]) if len(results[0].boxes) > 0 else 0
        
        # 5. UI 업데이트 및 하드웨어 시뮬레이션
        self.update_ui_result(ctk_img, is_dirty, max_conf, inference_time)
        
        # 6. LLM 리포트 생성 (가상 연동)
        self.generate_llm_report(is_dirty, target_img)
        
        self.scan_btn.configure(state="normal")

    def update_ui_result(self, img, is_dirty, conf, ms):
        # 이 메서드는 백그라운드 스레드에서 안전하게 UI를 업데이트합니다.
        self.after(0, lambda: self.img_label.configure(image=img, text=""))
        self.after(0, lambda: self.info_label.configure(text=f"Conf: {conf:.2f} | Time: {ms} ms"))
        
        if is_dirty:
            self.after(0, lambda: self.update_status_board("● DIRTY DETECTED", "#8B0000", "#FF0000"))
        else:
            self.after(0, lambda: self.update_status_board("● CLEAN / NORMAL", "#006400", "#00FF00"))

    def update_status_board(self, text, fg_color, border_color):
        self.status_board.configure(text=text, fg_color=fg_color)
        self.main_frame.configure(border_color=border_color)

    def generate_llm_report(self, is_dirty, img_path):
        # 나중에 여기에 Gemini API 등을 연결하면 됩니다.
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        img_name = os.path.basename(img_path)
        
        report_header = f"\n\n--- [Vision Doctor AI Report] ---\nTime: {timestamp}\nTarget: {img_name}\n\n"
        
        # 텍스트를 한 글자씩 출력하는 효과 (typing animation)
        def typing_effect(full_text):
            current_text = report_header
            for char in full_text:
                current_text += char
                # 스크롤바가 자동으로 맨 아래로 가도록 설정
                self.after(0, lambda t=current_text: self.report_text.delete("0.0", "end"))
                self.after(0, lambda t=current_text: self.report_text.insert("0.0", t))
                self.after(0, lambda: self.report_text.yview_moveto(1.0))
                time.sleep(0.005) # 타이핑 속도 조정

        if is_dirty:
            hw_sim_log = "\n> [HARDWARE] DIRTY Signal detected!\n> [HARDWARE] Sending signal to Jetson...\n> [HARDWARE] Wiper Motor START (Moving...)\n\n"
            content = hw_sim_log + (f"[진단 결과]\n렌즈 표면에 비정상적인 굴절 및 미세한 주름이 감지되었습니다. "
                       f"축사 환경 내 외부 오염물질(분뇨, 수분)일 가능성이 92%입니다.\n\n"
                       f"[복구 조치]\n1. 가상 Jetson 기반 카메라 세척 시스템(Wiper) 가동 명령 전송 완료.\n"
                       f"2. 물리적 서보 모터 구동을 통한 카메라 위치 초기화(Recalibration) 수행.\n"
                       f"3. 세척 후 재검사 대기 중.")
        else:
            hw_sim_log = "\n> [HARDWARE] Normal signal. Wiper STANDBY.\n\n"
            content = hw_sim_log + (f"[진단 결과]\n현재 렌즈 상태가 매우 양호합니다. 객체 인식 방해 요소가 발견되지 않았습니다.\n\n"
                       f"[복구 조치]\n물리적 조치가 불필요합니다. 지속적인 모니터링 체제를 유지합니다.")
            
        threading.Thread(target=typing_effect, args=(content,), daemon=True).start()

if __name__ == "__main__":
    # 데이터 폴더 없으면 경고
    if not os.path.exists(DATA_VAULT):
        print(f"❌ 폴더가 없습니다: {DATA_VAULT}\n빈 폴더를 생성하고 이미지를 넣어주세요.")
        os.makedirs(DATA_VAULT, exist_ok=True)
    
    app = VisionDoctorUI()
    app.mainloop()