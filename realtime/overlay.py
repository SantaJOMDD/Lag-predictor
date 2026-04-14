import sys
import os
import time
import pandas as pd
import joblib
import psutil
from ping3 import ping
import GPUtil
import keyboard

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QGridLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QFont

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "lag_model.pkl")

# Load model
features = ['bandwidth', 'throughput', 'congestion', 'packet_loss', 'latency', 'jitter']
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Failed to load model from {MODEL_PATH}: {e}")

class TelemetryThread(QThread):
    data_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.last_io = psutil.net_io_counters()
        self.last_time = time.time()
        self.last_latency = 0

    def run(self):
        while self.running:
            start_time = time.time()
            
            # --- HARDWARE METRICS ---
            # CPU
            cpu_percent = psutil.cpu_percent()
            
            # RAM
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            ram_used_gb = ram.used / (1024 ** 3)
            
            # GPU
            try:
                gpus = GPUtil.getGPUs()
                gpu_percent = gpus[0].load * 100 if gpus else 0.0
                gpu_vram = gpus[0].memoryUsed if gpus else 0.0
            except Exception:
                gpu_percent = 0.0
                gpu_vram = 0.0
            
            # --- NETWORK METRICS ---
            try:
                latency_sec = ping("8.8.8.8", timeout=1)
                if latency_sec is None:
                    latency = 1000.0
                    packet_loss = 1.0
                else:
                    latency = latency_sec * 1000.0
                    packet_loss = 0.0
            except Exception:
                latency = 1000.0
                packet_loss = 1.0

            jitter = abs(latency - self.last_latency) if self.last_latency else latency * 0.1
            self.last_latency = latency

            current_io = psutil.net_io_counters()
            current_time = time.time()
            dt = current_time - self.last_time
            dt = dt if dt > 0 else 0.1
            
            bytes_recv = current_io.bytes_recv - self.last_io.bytes_recv
            bytes_sent = current_io.bytes_sent - self.last_io.bytes_sent
            throughput_mbps = ((bytes_recv + bytes_sent) * 8) / (1024 * 1024 * dt)
            
            self.last_io = current_io
            self.last_time = current_time

            bandwidth = 5.0
            congestion = 0.2
            
            values = [bandwidth, throughput_mbps, congestion, packet_loss, latency, jitter]

            prediction_status = "Unknown"
            if model:
                data = pd.DataFrame([values], columns=features)
                pred = model.predict(data)[0]
                # Prediction corresponds to lag hitting ~3 seconds in the future
                prediction_status = "⚠️ LAG IN 3s" if pred == 1 else "🟢 STABLE"

            result = {
                "cpu": cpu_percent,
                "ram_pct": ram_percent,
                "ram_gb": ram_used_gb,
                "gpu_pct": gpu_percent,
                "gpu_vram": gpu_vram,
                "latency": latency,
                "jitter": jitter,
                "throughput": throughput_mbps,
                "status": prediction_status
            }
            
            self.data_updated.emit(result)

            elapsed = time.time() - start_time
            time.sleep(max(0, 1.0 - elapsed))

    def stop(self):
        self.running = False
        self.wait()

class OverlayWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Transparent, keep on top. Remove click-through so it can be dragged
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.setGeometry(50, 50, 360, 220)
        self.oldPos = self.pos()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)
        
        # Container styling
        self.central_widget.setStyleSheet("background-color: rgba(20, 20, 25, 230); border-radius: 10px; border: 1px solid #444;")
        
        font_main = QFont("Consolas", 10, QFont.Bold)
        font_large = QFont("Consolas", 14, QFont.Bold)
        
        def create_label(text, font, col="white"):
            lbl = QLabel(text)
            lbl.setFont(font)
            lbl.setStyleSheet(f"color: {col}; background: transparent; border: none; padding: 2px;")
            return lbl

        # Headers
        lbl_hw = create_label("💻 HARDWARE", font_main, "#00ccff")
        lbl_net = create_label("🌐 NETWORK", font_main, "#00ccff")
        
        self.layout.addWidget(lbl_hw, 0, 0, 1, 2)
        self.layout.addWidget(lbl_net, 0, 2, 1, 2)
        
        # Hardware Stats
        self.lbl_cpu = create_label("CPU: --%", font_main)
        self.lbl_ram = create_label("RAM: --%", font_main)
        self.lbl_gpu = create_label("GPU: --%", font_main)
        
        self.layout.addWidget(self.lbl_cpu, 1, 0, 1, 2)
        self.layout.addWidget(self.lbl_gpu, 2, 0, 1, 2)
        self.layout.addWidget(self.lbl_ram, 3, 0, 1, 2)

        # Network Stats
        self.lbl_ping = create_label("Ping: --ms", font_main)
        self.lbl_jitter = create_label("Jit: --ms", font_main)
        self.lbl_tp = create_label("Net: --Mb", font_main)
        
        self.layout.addWidget(self.lbl_ping, 1, 2, 1, 2)
        self.layout.addWidget(self.lbl_jitter, 2, 2, 1, 2)
        self.layout.addWidget(self.lbl_tp, 3, 2, 1, 2)
        
        # Divider
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #555;")
        self.layout.addWidget(line, 4, 0, 1, 4)

        # AI Prediction
        self.lbl_ai = create_label("🔮 PREDICTION:", font_main, "#bb88ff")
        self.layout.addWidget(self.lbl_ai, 5, 0, 1, 2)
        
        self.lbl_status = create_label("WAITING...", font_large, "white")
        self.layout.addWidget(self.lbl_status, 5, 2, 1, 2)
        
        # Drag Instruction
        lbl_info = create_label("[Drag to move] | [Ctrl+Shift+O] toggle", QFont("Consolas", 8), "#888")
        self.layout.addWidget(lbl_info, 6, 0, 1, 4, Qt.AlignCenter)

        self.layout.setContentsMargins(15, 15, 15, 10)

        # Worker Thread
        self.thread = TelemetryThread()
        self.thread.data_updated.connect(self.update_ui)
        self.thread.start()

        # Keyboard toggle hotkey
        keyboard.add_hotkey('ctrl+shift+o', self.toggle_visibility)

    def toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint (event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()

    def update_ui(self, data):
        self.lbl_cpu.setText(f"CPU: {data['cpu']:.0f}%")
        self.lbl_gpu.setText(f"GPU: {data['gpu_pct']:.0f}% | {data['gpu_vram']:.0f}MB")
        self.lbl_ram.setText(f"RAM: {data['ram_pct']:.0f}% | {data['ram_gb']:.1f}GB")
        
        self.lbl_ping.setText(f"Ping: {data['latency']:.0f}ms")
        self.lbl_jitter.setText(f"Jit: {data['jitter']:.0f}ms")
        self.lbl_tp.setText(f"Net: {data['throughput']:.1f}Mb")
        
        status = data['status']
        self.lbl_status.setText(status)
        if "⚠️" in status:
            self.lbl_status.setStyleSheet("color: #ff4c4c; background: transparent; border: none;")
        else:
            self.lbl_status.setStyleSheet("color: #4ce64c; background: transparent; border: none;")

    def closeEvent(self, event):
        self.thread.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OverlayWindow()
    window.show()
    sys.exit(app.exec_())
