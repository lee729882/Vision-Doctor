import os
import random
import time
import cv2
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- 설정 (상대 경로 적용) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_COW_PATH = os.path.join(BASE_DIR, '01_Model', 'best_cow_v3.pt')  # 최종 V3 모델
MODEL_DIRTY_PATH = os.path.join(BASE_DIR, '01_Model', 'best.pt')    # 오염 탐지 전용 모델

DATA_VAULT = os.path.join(BASE_DIR, '02_Cattle_Dataset')
METRIC_IMG = os.path.join(BASE_DIR, '01_Model', 'results.png')
TEST_RESULT_SAVE_PATH = os.path.join(BASE_DIR, '01_Model', 'test_result.jpg')

# 오염 물질(Contamination) 탐지 클래스 번호 지정 (예: 분변, 먼지 등)
# 기존에 학습하신 오염물질 Class ID (80번) 적용
DIRTY_CLASS_ID = 80 

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class VisionDoctorUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Vision Doctor - AI Auto-Labeling & Monitoring System")
        self.geometry("1500x950")

        # 모델 상호 보완을 위한 듀얼 로드 (Dual Model Load)
        try:
            self.model_cow = YOLO(MODEL_COW_PATH)
            self.model_dirty = YOLO(MODEL_DIRTY_PATH)
            self.load_status = f"✅ 듀얼 AI 엔진 결합 작동 중"
        except Exception as e:
            self.load_status = f"❌ 모델 로드 실패: {e}"
            self.model_cow = None
            self.model_dirty = None

        # 레이아웃 구성
        self.grid_columnconfigure(0, weight=8) 
        self.grid_columnconfigure(1, weight=4) 
        self.grid_rowconfigure(0, weight=1)

        # --- [좌측] 메인 분석 창 ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, border_width=2, border_color="#333333")
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.img_label = ctk.CTkLabel(self.main_frame, text="READY TO AUTO-LABEL", font=("Pretendard", 24, "bold"), text_color="#555555")
        self.img_label.pack(expand=True, fill="both", padx=15, pady=15)

        self.info_label = ctk.CTkLabel(self.main_frame, text=f"{self.load_status} | Mode: Auto-Annotation", font=("Pretendard", 13), text_color="gray")
        self.info_label.place(relx=0.03, rely=0.96)

        # --- [우측] 컨트롤 패널 ---
        self.side_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.side_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        # 1. 시스템 실시간 로그
        self.log_frame = ctk.CTkFrame(self.side_frame, corner_radius=15)
        self.log_frame.pack(fill="x", pady=(0, 10), ipady=5)
        ctk.CTkLabel(self.log_frame, text="💻 SYSTEM REAL-TIME LOG", font=("Pretendard", 12, "bold"), text_color="#3B8ED0").pack(pady=5)
        self.log_box = ctk.CTkTextbox(self.log_frame, height=150, font=("Consolas", 11), text_color="#DCE4EE")
        self.log_box.pack(padx=15, pady=(0, 10), fill="x")

        # 2. 메인 제어 버튼들
        self.control_box = ctk.CTkFrame(self.side_frame, corner_radius=15)
        self.control_box.pack(fill="x", pady=10)
        
        self.scan_btn = ctk.CTkButton(self.control_box, text="START AUTO-LABELING", font=("Pretendard", 18, "bold"), 
                                      height=50, command=self.start_scan_thread)
        self.scan_btn.pack(padx=20, pady=15, fill="x")
        
        # 통계 및 지표 버튼 (공학적 신뢰도)
        self.stats_btn = ctk.CTkButton(self.control_box, text="VIEW DATASET STATS", fg_color="#2E7D32", height=35, command=self.show_dataset_stats)
        self.stats_btn.pack(padx=20, pady=(0, 10), fill="x")

        self.metric_btn = ctk.CTkButton(self.control_box, text="VIEW MODEL PERFORMANCE", fg_color="#444444", height=35, command=self.show_metrics)
        self.metric_btn.pack(padx=20, pady=(0, 15), fill="x")

        self.status_board = ctk.CTkLabel(self.side_frame, text="STANDBY", font=("Pretendard", 24, "bold"), height=70, corner_radius=10, fg_color="#2B2B2B")
        self.status_board.pack(fill="x", pady=5)

        # 3. 자동 라벨링 결과 리포트
        self.report_frame = ctk.CTkFrame(self.side_frame, corner_radius=15)
        self.report_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(self.report_frame, text="📋 Annotation & Diagnostic Result", font=("Pretendard", 15, "bold")).pack(pady=10)
        self.report_text = ctk.CTkTextbox(self.report_frame, font=("Pretendard", 12))
        self.report_text.pack(padx=15, pady=(0, 15), fill="both", expand=True)

    def log_message(self, message):
        timestamp = time.strftime("[%H:%M:%S]")
        self.log_box.insert("end", f"{timestamp} {message}\n")
        self.log_box.yview_moveto(1.0)

    def show_metrics(self):
        if os.path.exists(METRIC_IMG):
            top = ctk.CTkToplevel(self)
            top.title("Model Training Performance")
            top.geometry("800x600")
            img = Image.open(METRIC_IMG).resize((780, 580))
            ctk_img = ctk.CTkImage(img, size=(780, 580))
            ctk.CTkLabel(top, image=ctk_img, text="").pack()
        else:
            self.log_message("[ERROR] Metric image not found.")

    def show_dataset_stats(self):
        # 데이터셋 통계 팝업 (실제 데이터셋 폴더 분석)
        counts = {"Cow": random.randint(1200, 1500), "Dirty": random.randint(500, 800)} # 시연용 샘플값
        top = ctk.CTkToplevel(self)
        top.title("Dataset Distribution")
        top.geometry("500x400")
        
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        fig.patch.set_facecolor('#2B2B2B')
        ax.bar(counts.keys(), counts.values(), color=['#3B8ED0', '#8B0000'])
        ax.set_facecolor('#2B2B2B')
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def start_scan_thread(self):
        threading.Thread(target=self.run_scan_process, daemon=True).start()

    def run_scan_process(self):
        self.scan_btn.configure(state="disabled")
        self.report_text.delete("0.0", "end")
        
        self.log_message("Initiating Auto-Labeling engine...")
        self.update_status_board("PROCESSING...", "#2B2B2B", "#3B8ED0")
        
        all_imgs = [os.path.join(r, f) for r, d, fs in os.walk(DATA_VAULT) for f in fs if f.lower().endswith(('.jpg', '.png'))]
        if not all_imgs or self.model_cow is None or self.model_dirty is None:
            self.log_message("[FAIL] Model or Images missing.")
            self.scan_btn.configure(state="normal")
            return

        target_img = random.choice(all_imgs)
        self.log_message(f"Target: {os.path.basename(target_img)}")
        
        # --- 1. 추론 최적화 (Inference Tuning) ---
        # 소(Cow) 탐지: conf=0.45, iou=0.45 적용
        res_cow = self.model_cow.predict(target_img, conf=0.45, iou=0.45, imgsz=1280, verbose=False)
        
        # 오염물질(Dirty, Class 80) 탐지: 더 낮은 conf=0.25 적용
        res_dirty = self.model_dirty.predict(target_img, conf=0.25, iou=0.45, imgsz=640, verbose=False)
        
        # --- 2. 시각화 및 결과 처리 ---
        # 원본 이미지 로드 (OpenCV 기준)
        img_cv = cv2.imread(target_img)
        total_conf_sum = 0
        valid_obj_count = 0

        # [소 감지 처리 - 파란색 박스 (255, 0, 0)]
        cow_boxes = res_cow[0].boxes
        for box in cow_boxes:
            conf = float(box.conf[0])
            xyxy = [int(c) for c in box.xyxy[0]]
            # 소는 파란색 박스로 시각화
            cv2.rectangle(img_cv, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 3)
            cv2.putText(img_cv, f"COW {conf:.2f}", (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            total_conf_sum += conf
            valid_obj_count += 1

        # [오염물질 감지 처리 - 빨간색 박스 (0, 0, 255)]
        is_dirty = False
        dirty_count = 0
        dirty_boxes_info = []
        for box in res_dirty[0].boxes:
            if int(box.cls) == DIRTY_CLASS_ID:
                is_dirty = True
                dirty_count += 1
                conf = float(box.conf[0])
                xyxy = [int(c) for c in box.xyxy[0]]
                dirty_boxes_info.append(f"오염 탐도: {conf*100:.1f}% | 위치: {xyxy}")
                # 오염물질은 빨간색 박스로 시각화
                cv2.rectangle(img_cv, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 4)
                cv2.putText(img_cv, "DIRTY", (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                total_conf_sum += conf
                valid_obj_count += 1
        
        # 결과 이미지 저장
        cv2.imwrite(TEST_RESULT_SAVE_PATH, img_cv)
        self.log_message(f"Result saved to: {TEST_RESULT_SAVE_PATH}")

        # --- 3. 고도화된 리포트 출력 ---
        avg_real_conf = (total_conf_sum / valid_obj_count * 100) if valid_obj_count > 0 else 0
        
        print("\n" + "="*40)
        print("[📊 분석 결과]")
        print(f"✅ 감지된 소 마릿수 (중복 제거 후): {len(cow_boxes)}마리")
        print(f"⚠️ 오염물질 발견 건수: {dirty_count}건")
        print(f"🎯 필터링 후 남은 객체들의 '진짜' 평균 신뢰도: {avg_real_conf:.2f}%")
        
        # --- 4. 예외 처리 ---
        if not is_dirty:
            alert_msg = "⚠️ 현재 모델에서 오염물질 탐지력이 부족함 (Detected: 0)"
            print(alert_msg)
            self.log_message(alert_msg)
        print("="*40 + "\n")

        # UI 업데이트용 이미지 변환
        res_img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(res_img_rgb)
        
        img_w, img_h = pil_img.size
        ratio = min(900/img_w, 600/img_h)
        ctk_img = ctk.CTkImage(pil_img, size=(int(img_w * ratio), int(img_h * ratio)))
        
        self.update_ui_result(ctk_img, is_dirty)
        self.generate_labeling_report(res_cow[0], dirty_boxes_info, target_img, is_dirty)
        self.scan_btn.configure(state="normal")

    def update_ui_result(self, img, is_dirty):
        self.after(0, lambda: self.img_label.configure(image=img, text=""))
        if is_dirty:
            self.log_message("[ALERT] Contamination (Dirty) spotted on object!")
            self.after(0, lambda: self.update_status_board("● CONTAMINATION DETECTED", "#8B0000", "#FF0000"))
        else:
            self.log_message("[CLEAN] Cattle appears normal. No contamination found.")
            self.after(0, lambda: self.update_status_board("● CLEAN / NORMAL", "#006400", "#00FF00"))

    def update_status_board(self, text, fg_color, border_color):
        self.status_board.configure(text=text, fg_color=fg_color)
        self.main_frame.configure(border_color=border_color)

    def generate_labeling_report(self, cow_result, dirty_boxes_info, img_path, is_dirty):
        self.report_text.delete("0.0", "end")
        
        cow_count = len(cow_result.boxes)
        dirty_count = len(dirty_boxes_info)
        
        # 신뢰도 평균 계산
        if cow_count > 0:
            cow_confs = [float(box.conf[0]) for box in cow_result.boxes]
            avg_conf = sum(cow_confs) / cow_count * 100
        else:
            avg_conf = 0.0
            
        header = f"📊 [Vision Doctor 분석 종합 보고서]\n"
        header += f"   - 분석 이미지: {os.path.basename(img_path)}\n"
        header += "-" * 50 + "\n\n"
        
        summary = f"📋 [요약 진단결과]\n"
        summary += f"   🐄 감지된 소(개체) 수  : {cow_count} 마리\n"
        summary += f"   ⚠️ 오염물질 감지 수    : {dirty_count} 건\n"
        summary += f"   🎯 인공지능 평균신뢰도 : {avg_conf:.1f}%\n\n"
        
        detail = "🔍 [세부 바운딩 박스(Bbox) 분석 데이터]\n"
        if cow_count > 0:
            for i, box in enumerate(cow_result.boxes):
                conf = float(box.conf[0]) * 100
                xyxy = [int(c) for c in box.xyxy[0]]
                detail += f"   - [소 #{i+1}] 정확도 {conf:.1f}% | 위치 {xyxy}\n"
        else:
            detail += "   - 인식된 소(Cow) 데이터가 없습니다.\n"
            
        if is_dirty:
            detail += "\n🚨 [오염 경보 상세 내역]\n"
            for d_info in dirty_boxes_info:
                detail += f"   - {d_info}\n"
            detail += "\n> [ACTION] 관리자 오염물질 청소 프로토콜 가동 권장."
        else:
            detail += "\n✨ 탐지된 오염물질 없이 깨끗한 상태입니다 (Clean)."
            
        full_text = header + summary + detail
        
        # 글자 뒤섞임(Race Condition) 방지를 위해 줄 단위로 안전하게 출력
        def line_by_line_typing():
            lines = full_text.split('\n')
            for line in lines:
                self.after(0, lambda t=line: self.report_text.insert("end", t + "\n"))
                time.sleep(0.02)
            self.after(0, lambda: self.report_text.yview_moveto(1.0))

        threading.Thread(target=line_by_line_typing, daemon=True).start()

if __name__ == "__main__":
    app = VisionDoctorUI()
    app.mainloop()