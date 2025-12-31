import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np
import av
import threading
from PIL import Image
from ultralytics import YOLO
import time  # Library tambahan untuk mengatur waktu per detik

st.set_page_config(
    page_title="Deteksi Katarak Mata",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- LOAD MODEL YOLOv8 ---
@st.cache_resource
def load_model():
    # Pastikan file best.onnx ada di folder yang sama
    return YOLO('best.onnx')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error memuat model: {e}. Pastikan file ada di direktori yang sama.")
    st.stop()

# --- STYLING CSS (TIDAK DIUBAH) ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    div.block-container {
        background-color: rgba(0, 0, 0, 0.4);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 2rem;
        max-width: 800px;
    }

    @media (max-width: 600px) {
        div.block-container {
            padding: 1rem;
            margin-top: 0.5rem;
            width: 95%;
        }
        h2 {
            font-size: 1.5rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 0 0.5rem;
            height: 40px;
        }
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
        height: 0px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        margin-bottom: 1rem;
        padding-left: 0;
        padding-right: 0;
        display: flex;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        color: white;
        flex: 1 1 auto;
        border: none;
        font-weight: 500;
        min-width: 120px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.2);
        color: #f0f0f0;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #bd34fe, #ff34b3);
        color: white;
        box-shadow: 0 4px 15px rgba(189, 52, 254, 0.4);
    }

    iframe[title="streamlit_webrtc.component.webrtc_streamer"] {
        background-color: #000;
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-sizing: border-box;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        display: block;
        width: 100%;
    }

    div.stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 12px;
        border: none;
        height: 3.5rem;
        width: 100%;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:active,
    div.stButton > button:focus {
        background-color: #ff4b4b;
        color: white !important;
        box-shadow: none;
        border: none;
    }
    
    div.stButton > button:hover {
        background-color: #ff6b6b;
        color: white;
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    
    [data-testid="stFileUploader"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Aplikasi Deteksi Katarak (YOLOv8)</h2>", unsafe_allow_html=True)
st.write("")

tab1, tab2 = st.tabs(["📷 Kamera Real-time", "📂 Upload File Mata"])

# --- TAB 1: REAL-TIME WEBRTC (ANTI-LAG VERSION) ---
with tab1:
    st.write("Arahkan mata Anda ke kamera untuk dianalisis, pastikan pencahayaan bagus untuk hasil yang lebih akurat.")

    class VideoProcessor:
        def __init__(self):
            self.frame = None
            
            # Variabel untuk komunikasi antar Thread
            self.lock = threading.Lock()
            self.source_img = None  # Gambar mentah untuk diproses AI
            self.detection_results = [] # Menyimpan hasil deteksi (koordinat kotak)
            
            # Status thread
            self.stopped = False
            
            # Jalankan Thread AI terpisah
            self.t = threading.Thread(target=self.run_inference)
            self.t.daemon = True # Thread mati otomatis saat program berhenti
            self.t.start()

        def run_inference(self):
            """Fungsi ini berjalan sendiri di background tanpa mengganggu video"""
            while not self.stopped:
                # Ambil gambar terbaru dari main thread
                img_to_process = None
                with self.lock:
                    if self.source_img is not None:
                        img_to_process = self.source_img.copy()
                
                if img_to_process is not None:
                    # 1. Resize kecil agar AI ngebut (320px)
                    h, w = img_to_process.shape[:2]
                    target_width = 640
                    ratio = target_width / float(w)
                    img_small = cv2.resize(img_to_process, (target_width, int(h * ratio)))
                    
                    # 2. Proses YOLO (Ini bagian yang berat)
                    results = model(img_small, verbose=False)
                    
                    # 3. Simpan koordinat kotak hasil deteksi
                    # Kita harus menormalkan koordinat agar bisa dipakai di resolusi asli
                    new_detections = []
                    for result in results:
                        for box in result.boxes:
                            # Koordinat di gambar kecil
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            name = model.names[cls]
                            
                            # Kembalikan koordinat ke ukuran asli
                            scale_back = 1 / ratio
                            x1 = int(x1 * scale_back)
                            y1 = int(y1 * scale_back)
                            x2 = int(x2 * scale_back)
                            y2 = int(y2 * scale_back)
                            
                            new_detections.append((x1, y1, x2, y2, name, conf))
                    
                    # Update hasil deteksi ke variabel shared
                    with self.lock:
                        self.detection_results = new_detections
                        self.source_img = None # Reset agar tidak memproses gambar yang sama berulang kali
                
                # Istirahat dikit biar CPU tidak 100%
                time.sleep(0.05)

        def recv(self, frame):
            # Ini adalah Main Thread (Jalur Video)
            # Harus secepat mungkin, jangan ada proses berat di sini!
            
            img = frame.to_ndarray(format="bgr24")
            
            # 1. Kirim gambar ke Thread AI (hanya copy, jangan tunggu)
            with self.lock:
                # Kita hanya update source jika source sebelumnya sudah diambil AI
                if self.source_img is None:
                    self.source_img = img
            
            # 2. Gambar kotak manual berdasarkan hasil terakhir dari Thread AI
            # Kita gambar manual pakai OpenCV karena result.plot() itu berat dan bikin lag
            with self.lock:
                detections = self.detection_results
            
            # Loop untuk menggambar kotak
            for (x1, y1, x2, y2, label, conf) in detections:
                # Warna Kotak (Hijau)
                color = (0, 255, 0) 
                
                # Gambar Persegi
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Gambar Label (Background label)
                label_text = f"{label} {conf:.2f}"
                (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                
                # Tulis Teks
                cv2.putText(img, label_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Simpan frame untuk tombol capture
            self.frame = img
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
        key="cataract-detection-async", # Ganti key biar refresh bersih
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing:
        if st.button("📸 Capture & Simpan Hasil"):
            if ctx.video_processor:
                frame_captured = ctx.video_processor.frame
                if frame_captured is not None:
                    frame_rgb = cv2.cvtColor(frame_captured, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Hasil Capture", use_container_width=True)
                    st.success("Gambar berhasil diambil!")
                else:
                    st.warning("Belum ada frame.")
            else:
                st.warning("Kamera sedang loading...")

# --- TAB 2: UPLOAD IMAGE ---
with tab2:
    st.write("Unggah foto mata untuk dianalisis.")
    
    uploaded_file = st.file_uploader("Pilih file gambar", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption='Foto Asli', use_container_width=True)
        
        if st.button("✅ Konfirmasi & Analisis"):
            with st.spinner('Sedang menganalisis katarak...'):
                # Konversi ke numpy array
                image_np = np.array(image_pil)
                
                # Jalankan Inferensi
                results = model(image_np)
                
                # Ambil gambar hasil plot (Format BGR dari YOLO)
                res_plotted = results[0].plot()
                
                # Tampilkan hasil
                st.image(res_plotted, caption='Hasil Deteksi AI', use_container_width=True)
                
                # Tampilkan probabilitas
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.success(f"Terdeteksi {len(boxes)} objek.")
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = model.names[cls]
                        st.info(f"Deteksi: **{name}** (Kepercayaan: {conf:.2%})")
                else:
                    st.warning("Tidak ada objek katarak/mata yang terdeteksi.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey; font-size: 0.8rem;'>Project Deteksi Katarak © 2025</p>", unsafe_allow_html=True)