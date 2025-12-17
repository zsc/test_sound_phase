import eventlet
eventlet.monkey_patch()  # 必须在最前面，用于WebSocket并发

import sys
import time
import threading
import numpy as np
import pyaudio
import librosa
from scipy import signal
from PIL import Image
import io
import base64
from flask import Flask, render_template_string
from flask_socketio import SocketIO

# ================= 配置参数 =================
SAMPLE_RATE = 16000
CHUNK_SIZE = 512      # 每次读取的音频帧数
N_FFT = 1024          # STFT 窗口大小
HOP_LENGTH = 256      # STFT 步长
N_MELS = 128          # Mel 频段数量
HISTORY_LEN = 128     # 频谱图时间轴显示的帧数 (宽度)
DEFAULT_THRESH = -60  # 默认 dB 阈值

# ================= DSP 处理类 =================
class AudioProcessor:
    def __init__(self):
        self.buffer = np.zeros(CHUNK_SIZE * HISTORY_LEN)
        self.mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
        self.threshold_db = DEFAULT_THRESH
        self.lock = threading.Lock()
        
    def add_data(self, data_bytes):
        # 将字节流转为 float32 (-1.0 到 1.0)
        audio_data = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        with self.lock:
            # 环形缓冲：移除旧数据，追加新数据
            self.buffer = np.roll(self.buffer, -len(audio_data))
            self.buffer[-len(audio_data):] = audio_data

    def set_threshold(self, val):
        self.threshold_db = float(val)

    def get_frame(self):
        with self.lock:
            current_buffer = self.buffer.copy()

        # 1. 计算 STFT
        # 返回形状: (1 + n_fft/2, t)
        f, t, Zxx = signal.stft(current_buffer, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
        
        # 只取最后 HISTORY_LEN 帧以保持图片宽度固定
        if Zxx.shape[1] > HISTORY_LEN:
            Zxx = Zxx[:, -HISTORY_LEN:]
        
        # === Channel 1: Mel Spectrogram ===
        mag = np.abs(Zxx)
        # 转换到 Mel 刻度
        mel_spec = np.dot(self.mel_basis, mag)
        # 转 dB
        mel_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        
        # 归一化 Channel 1 到 0-1 (假设范围 -80dB 到 0dB)
        c1 = (mel_db + 80) / 80
        c1 = np.clip(c1, 0, 1)

        # === 阈值处理 ===
        # 创建掩码：低于阈值的设为 False
        mask = mel_db > self.threshold_db

        # === Channel 2 & 3: Phase (Cos/Sin) ===
        phase = np.angle(Zxx)
        
        # 这里需要注意：Zxx是线性频率，我们需要将其映射到Mel频率对应的相位
        # 简单的做法是：既然 Mel 是线性频带的加权和，相位很难直接"加权"。
        # 为了可视化的连续性，通常直接取 Mel 滤波器中心频率对应的 bin 的相位，
        # 或者为了演示效果，我们这里简单地对相位矩阵做插值(resize)到 N_MELS 高度。
        
        # 这种 resize 方法能保留相位的纹理结构用于 CNN 输入
        # 原始 Phase 形状 (513, Time) -> 目标 (128, Time)
        # 使用 zoom 或 resize，这里手动简易重采样
        phase_resampled = signal.resample(phase, N_MELS, axis=0)

        # 计算 Cos 和 Sin
        c2 = np.cos(phase_resampled)
        c3 = np.sin(phase_resampled)

        # === 应用阈值逻辑 ===
        # 幅度低于阈值时，设相位为 0
        # 此时 Cos(0)=1, Sin(0)=0. 
        # 但通常为了视觉区分"无信号"，我们可能希望它们变黑(0)。
        # 用户需求是"设相位为0"，数学上意味着 Cos=1, Sin=0。
        # 配合 mask 操作：
        
        # 将 mask 也调整大小以匹配 Mel 维度
        mask_resampled = np.zeros((N_MELS, mel_db.shape[1]), dtype=bool)
        # 简单的做法：因为 c1 已经是 Mel 域的，直接用 c1 的阈值判断即可
        mask_resampled = mel_db > self.threshold_db

        # 应用逻辑：如果 mask 为 False (低于阈值)，相位设为 0
        # c2[~mask_resampled] = np.cos(0) # = 1
        # c3[~mask_resampled] = np.sin(0) # = 0
        
        # 另一种解读：如果为了作为神经网络输入，通常希望背景纯黑(0)。
        # 但如果是严格按照"相位=0"，则如下：
        c2[~mask_resampled] = 1.0 
        c3[~mask_resampled] = 0.0

        # 为了让图片显示好看，将 Cos/Sin 从 [-1, 1] 映射到 [0, 1] 用于显示
        # 如果是用于机器读取，请保留原始 float。这里是为了 HTML 显示。
        c2_img = (c2 + 1) / 2
        c3_img = (c3 + 1) / 2

        # 堆叠成 RGB (Channel 1, 2, 3)
        # C1: Mel (Red), C2: Cos (Green), C3: Sin (Blue)
        # 图像通常是 (Height, Width, Channels)
        img_stack = np.dstack((c1, c2_img, c3_img))
        
        # 翻转 Y 轴，让低频在下，高频在上
        img_stack = np.flipud(img_stack)

        # 转为 8-bit 整数
        img_uint8 = (img_stack * 255).astype(np.uint8)
        
        return img_uint8

# ================= Flask Web Server =================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
processor = AudioProcessor()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Audio to 3-Channel Image</title>
    <style>
        body { font-family: sans-serif; background: #222; color: #fff; text-align: center; }
        #container { margin-top: 20px; }
        img { border: 2px solid #555; image-rendering: pixelated; width: 80%; max-width: 800px; }
        .controls { margin: 20px; background: #333; padding: 20px; display: inline-block; border-radius: 8px; }
        input[type=range] { width: 300px; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
</head>
<body>
    <h1>Microphone Signal -> 3-Channel Spec/Phase</h1>
    
    <div id="container">
        <img id="spectrogram" src="" alt="Waiting for signal..." />
    </div>

    <div class="controls">
        <label for="thresh">Amplitude Threshold (dB): <span id="thresh-val">-60</span></label><br><br>
        <input type="range" id="thresh" min="-80" max="0" value="-60" step="1">
    </div>

    <script>
        const socket = io();
        const img = document.getElementById('spectrogram');
        const slider = document.getElementById('thresh');
        const valDisplay = document.getElementById('thresh-val');

        // 接收图片数据
        socket.on('new_image', function(data) {
            img.src = 'data:image/jpeg;base64,' + data;
        });

        // 滑块交互
        slider.oninput = function() {
            valDisplay.innerText = this.value;
            socket.emit('update_threshold', {val: this.value});
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@socketio.on('update_threshold')
def handle_threshold(json):
    processor.set_threshold(json['val'])

# ================= 后台音频线程 =================
def audio_stream_loop():
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        return

    print("Audio stream started...")
    while True:
        try:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            processor.add_data(data)
            
            # 限制生成帧率，例如 20 FPS，以免浏览器卡死
            socketio.sleep(0.05) 
            
            # 获取处理后的图片
            img_array = processor.get_frame()
            
            # 转为 JPEG Base64
            pil_img = Image.fromarray(img_array)
            buff = io.BytesIO()
            pil_img.save(buff, format="JPEG", quality=80)
            b64_str = base64.b64encode(buff.getvalue()).decode('utf-8')
            
            # 推送到前端
            socketio.emit('new_image', b64_str)
            
        except Exception as e:
            print(f"Stream error: {e}")
            break
            
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
    # 启动音频线程
    t = socketio.start_background_task(audio_stream_loop)
    print("Starting server at http://127.0.0.1:5000")
    socketio.run(app, port=5000, debug=False)
