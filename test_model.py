import cv2
from ultralytics import YOLO
import time

def run_detection():
    # --- KONFIGURASI ---
    model_path = 'best.onnx'   # Ganti dengan path model Anda
    video_source = 0         # 0 untuk Webcam, atau ganti 'video.mp4'
    confidence_threshold = 0.5
    # -------------------

    # 1. Load Model
    print(f"Sedang memuat model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error: Model tidak ditemukan atau rusak. {e}")
        return

    # 2. Buka Sumber Video (Webcam/File)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera atau file video.")
        return

    # Atur resolusi kamera (opsional, jika pakai webcam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Tekan 'q' pada keyboard untuk keluar.")

    while True:
        start_time = time.time()
        
        # Baca frame dari kamera
        success, frame = cap.read()
        
        if not success:
            print("Video selesai atau stream terputus.")
            break

        # 3. Lakukan Deteksi (Inference)
        # verbose=False agar terminal tidak penuh teks
        results = model(frame, conf=confidence_threshold, verbose=False)

        # 4. Visualisasikan Hasil
        # results[0].plot() menggambar kotak bounding box ke frame
        annotated_frame = results[0].plot()

        # Hitung FPS (Frame Per Second)
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 5. Tampilkan di Window Desktop
        cv2.imshow("YOLOv8 Detection - Tekan 'q' untuk keluar", annotated_frame)

        # Cek input keyboard (tekan 'q' untuk quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersihkan resources
    cap.release()
    cv2.destroyAllWindows()
    print("Program selesai.")

if __name__ == "__main__":
    run_detection()