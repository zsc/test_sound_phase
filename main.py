import librosa
import numpy as np
import base64
from PIL import Image
from io import BytesIO

def audio_to_3channel_image(audio_path, output_html_path="output.html", 
                            sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    将音频转换为3通道图像（Mel谱 + Cos相位 + Sin相位）并生成HTML预览。
    """
    print(f"正在处理音频: {audio_path} ...")
    
    # 1. 加载音频
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"加载音频失败: {e}")
        return

    # 2. 计算 STFT (得到复数矩阵 D)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # 3. 分离幅度与相位
    # magnitude: 振幅
    # phase_angle: 相位角 (弧度)
    magnitude, phase_angle = librosa.magphase(D)
    
    # 4. 计算 Cos 和 Sin 相位分量
    # 这将相位的周期性映射为连续的数值 [-1, 1]
    phase_cos = np.cos(phase_angle)
    phase_sin = np.sin(phase_angle)

    # 5. 创建 Mel 滤波器组
    # 我们不仅对幅度做 Mel 变换，对相位分量也做同样的变换以保持空间维度对齐
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    # 6. 处理 Channel 1: Mel Spectrogram (Log-Magnitude)
    # 矩阵乘法: (n_mels, n_fft/2 + 1) dot (n_fft/2 + 1, time_steps)
    mel_mag = np.dot(mel_basis, magnitude)
    # 转为分贝 (Log scale)，这是听觉感知的标准做法
    mel_mag_db = librosa.amplitude_to_db(mel_mag, ref=np.max)
    
    # 归一化 Ch1 到 [0, 255]
    #通常音频范围在 -80dB 到 0dB 之间
    min_db = -80.0
    max_db = 0.0
    mel_mag_db = np.clip(mel_mag_db, min_db, max_db)
    ch1 = 255 * (mel_mag_db - min_db) / (max_db - min_db)

    # 7. 处理 Channel 2 & 3: Mel-scaled Phase
    # 注意：直接对相位做线性滤波(Mel)在物理上是有损的，但为了图像对齐是必须的。
    # 我们分别对 cos 和 sin 分量进行 Mel 投影
    mel_cos = np.dot(mel_basis, phase_cos)
    mel_sin = np.dot(mel_basis, phase_sin)

    # 归一化 Ch2, Ch3 到 [0, 255]
    # Cos/Sin 经过 Mel 加权后数值范围可能会变，我们需要重新归一化
    # 简单的 Min-Max 归一化策略，或者假设它们大致在 [-1, 1] 范围内
    def normalize_phase_channel(data):
        # 激进的归一化，确保利用满色阶
        d_min, d_max = data.min(), data.max()
        if d_max - d_min == 0:
            return np.zeros_like(data)
        return 255 * (data - d_min) / (d_max - d_min)

    ch2 = normalize_phase_channel(mel_cos)
    ch3 = normalize_phase_channel(mel_sin)

    # 8. 堆叠通道 (Height, Width, 3)
    # 图像数据通常需要是 uint8 类型
    img_data = np.stack([ch1, ch2, ch3], axis=-1).astype(np.uint8)
    
    # 9. 图像反转 (可选)
    # librosa 的频率轴默认是从低到高 (index 0 是低频)，作为图片通常希望低频在下方
    # origin='lower'。但在数组中 index 0 是顶部。
    # 为了让图片看起来像标准的声谱图（低频在下，我们需要上下翻转
    img_data = np.flipud(img_data)

    # 10. 生成图片对象
    img = Image.fromarray(img_data, 'RGB')
    
    # 转换为 Base64 以嵌入 HTML
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # 11. 生成 HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio to 3-Channel Image</title>
        <style>
            body {{ font-family: sans-serif; background: #1a1a1a; color: #fff; text-align: center; padding: 20px; }}
            .container {{ display: inline-block; background: #333; padding: 20px; border-radius: 10px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #555; }}
            .info {{ margin-top: 15px; text-align: left; font-size: 0.9em; color: #ccc; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Audio Feature Visualization</h2>
            <p><strong>Source:</strong> {audio_path}</p>
            <img src="data:image/png;base64,{img_str}" alt="Spectrogram Representation" />
            <div class="info">
                <p><strong>R channel (Red):</strong> Mel-Spectrogram (Log Magnitude)</p>
                <p><strong>G channel (Green):</strong> Cosine of Phase (Mel-scaled)</p>
                <p><strong>B channel (Blue):</strong> Sine of Phase (Mel-scaled)</p>
                <p><strong>Dimensions:</strong> {img_data.shape[1]}px (Time) x {img_data.shape[0]}px (Mel Freq)</p>
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"完成! HTML 已保存至: {output_html_path}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 请替换为你的 wav 或 mp3 文件路径
    #input_audio = "../test_wav/gaoshi_zaomiao.wav"
    input_audio = "../chord_progression.wav"
    
    '''
    # 如果没有文件，生成一个假的测试音频（正弦波）
    import soundfile as sf
    sr = 22050
    t = np.linspace(0, 5, sr * 5)
    y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    sf.write('test_audio_440hz.wav', y, sr)
    input_audio = 'test_audio_440hz.wav'
    '''

    audio_to_3channel_image(input_audio, "spectrogram_view.html")
