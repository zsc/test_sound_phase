写一段 python，能实时把输入 microphone 声音信号 转成一个 3-channel 图，第一个 channel 为 spectrogram（类 mel）, 第二、三个channel 分别为 cos/sin 相位（这样能更连续地表示相位的周期性），最后输出一个 html 包含图片。幅度低于阈值时，设相位为0. 在 html 增加设阈值的控件。

写一个 all-in-one html，能实时把输入 microphone 声音信号 转成两种之一：
A. 一个 3-channel 图，第一个 channel 为 spectrogram（类 mel）, 第二、三个channel 分别为 cos/sin 相位（这样能更连续地表示相位的周期性）。注意是三通道合在一起的图。
幅度低于阈值时，设相位为0. 在 html 增加设阈值的控件。
B. mel 频谱图
