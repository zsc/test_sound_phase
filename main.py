import numpy as np
import librosa
import base64
import io
from PIL import Image
import os
import argparse

def process_audio_to_html(input_path, output_html="output.html", target_width=1000, target_height=400):
    print(f"Loading audio: {input_path}...")
    try:
        y, sr = librosa.load(input_path, sr=None)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # 1. 计算 STFT
    # n_fft 决定了频率分辨率，hop_length 决定了时间分辨率
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # 分离幅度和相位
    magnitude, phase = librosa.magphase(D)
    
    # 2. Channel 1: Spectrogram (Log Scale)
    # 使用对数分贝刻度，更符合人耳听感 (类 Mel 视觉效果)
    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # 归一化幅度到 0-255 用于显示和存储
    # min-max normalization
    mag_min, mag_max = mag_db.min(), mag_db.max()
    mag_norm = (mag_db - mag_min) / (mag_max - mag_min)
    mag_uint8 = (mag_norm * 255).astype(np.uint8)

    # 3. Channel 2 & 3: Phase (Cos & Sin)
    # 相位范围是 -pi 到 pi
    cos_phase = np.cos(phase) # range -1 to 1
    sin_phase = np.sin(phase) # range -1 to 1

    # 将 -1~1 映射到 0~255
    # -1 -> 0, 0 -> 127.5, 1 -> 255
    cos_uint8 = ((cos_phase + 1) / 2 * 255).astype(np.uint8)
    sin_uint8 = ((sin_phase + 1) / 2 * 255).astype(np.uint8)

    # 4. 图像缩放 (Resizing)
    # 原始 STFT 矩阵可能非常大，为了 HTML 流畅度，我们需要缩放图片
    # 注意：为了保持数据对其，我们需要翻转 Y 轴（低频在下，高频在上）
    def resize_matrix(matrix, width, height):
        # 翻转 Y 轴，让低频在底部
        matrix_flipped = np.flipud(matrix) 
        img = Image.fromarray(matrix_flipped)
        img = img.resize((width, height), Image.Resampling.BILINEAR)
        return np.array(img)

    print("Resizing data for visualization...")
    mag_resized = resize_matrix(mag_uint8, target_width, target_height)
    cos_resized = resize_matrix(cos_uint8, target_width, target_height)
    sin_resized = resize_matrix(sin_uint8, target_width, target_height)

    # 5. 数据编码 (Base64)
    # 我们将把这三个通道的 raw bytes 嵌入到 HTML JS 中
    # 这样 JS 就可以根据阈值动态重新组合 RGB
    def encode_data(arr):
        return base64.b64encode(arr.tobytes()).decode('utf-8')

    mag_b64 = encode_data(mag_resized)
    cos_b64 = encode_data(cos_resized)
    sin_b64 = encode_data(sin_resized)

    # 6. 生成 HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Phase Visualization</title>
    <style>
        body {{ font-family: sans-serif; background: #222; color: #eee; display: flex; flex-direction: column; align-items: center; padding: 20px; }}
        h2 {{ margin-bottom: 10px; }}
        .controls {{ background: #333; padding: 20px; border-radius: 8px; margin-bottom: 20px; width: {target_width}px; box-sizing: border-box; }}
        label {{ display: flex; align-items: center; justify-content: space-between; font-weight: bold; }}
        input[type=range] {{ flex-grow: 1; margin: 0 15px; }}
        canvas {{ border: 2px solid #555; background: #000; image-rendering: pixelated; }}
        .info {{ margin-top: 10px; font-size: 0.9em; color: #aaa; }}
        .legend {{ display: flex; gap: 20px; margin-top: 5px; font-size: 0.8em; }}
        .dot {{ width: 10px; height: 10px; display: inline-block; margin-right: 5px; border-radius: 50%; }}
    </style>
</head>
<body>

    <h2>3-Channel Audio Visualization</h2>
    
    <div class="controls">
        <label>
            Magnitude Threshold (Noise Gate): <span id="threshVal">0</span>%
            <input type="range" id="threshold" min="0" max="255" value="0">
        </label>
        <div class="legend">
            <span><span class="dot" style="background:red;"></span>Red: Log Spectrogram</span>
            <span><span class="dot" style="background:green;"></span>Green: Phase Cosine</span>
            <span><span class="dot" style="background:blue;"></span>Blue: Phase Sine</span>
        </div>
        <div class="info">
            When Magnitude < Threshold: Phase is forced to 0 (Cos=1, Sin=0).<br>
            Green(Cos) 255 = +1, 0 = -1. Blue(Sin) 255 = +1, 0 = -1.
        </div>
    </div>

    <canvas id="myCanvas" width="{target_width}" height="{target_height}"></canvas>

    <script>
        // Configuration
        const width = {target_width};
        const height = {target_height};
        const totalPixels = width * height;

        // Decode Base64 Data into Uint8Arrays
        function base64ToUint8(base64) {{
            const binary_string = window.atob(base64);
            const len = binary_string.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) {{
                bytes[i] = binary_string.charCodeAt(i);
            }}
            return bytes;
        }}

        const magData = base64ToUint8("{mag_b64}");
        const cosData = base64ToUint8("{cos_b64}");
        const sinData = base64ToUint8("{sin_b64}");

        const canvas = document.getElementById('myCanvas');
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(width, height);
        const slider = document.getElementById('threshold');
        const threshDisplay = document.getElementById('threshVal');

        function render() {{
            const threshold = parseInt(slider.value);
            threshDisplay.innerText = Math.round((threshold / 255) * 100);
            
            const data = imgData.data; // R, G, B, A interleaved

            for (let i = 0; i < totalPixels; i++) {{
                const mag = magData[i];
                let cosVal, sinVal;

                if (mag < threshold) {{
                    // Magnitude is below threshold
                    // Requirement: Set Phase to 0.
                    // If Phase = 0: Cos(0) = 1, Sin(0) = 0
                    
                    // Map 1.0 to byte: (1 + 1)/2 * 255 = 255
                    cosVal = 255; 
                    
                    // Map 0.0 to byte: (0 + 1)/2 * 255 = 127.5 -> 127
                    sinVal = 127;
                }} else {{
                    cosVal = cosData[i];
                    sinVal = sinData[i];
                }}

                const ptr = i * 4;
                data[ptr] = mag;      // Red Channel: Magnitude
                data[ptr + 1] = cosVal; // Green Channel: Cos
                data[ptr + 2] = sinVal; // Blue Channel: Sin
                data[ptr + 3] = 255;    // Alpha: Opaque
            }}
            
            ctx.putImageData(imgData, 0, 0);
        }}

        // Initial render
        render();

        // Event listener
        slider.addEventListener('input', render);
    </script>
</body>
</html>
    """

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Done! HTML saved to: {os.path.abspath(output_html)}")

if __name__ == "__main__":
    # 创建简单的命令行参数
    parser = argparse.ArgumentParser(description="Convert Audio to 3-Channel Phase HTML")
    parser.add_argument("input_file", help="Path to input wav/mp3 file")
    parser.add_argument("--output", default="audio_viz.html", help="Output HTML file name")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input_file):
        process_audio_to_html(args.input_file, args.output)
    else:
        print("Input file not found.")
