# Kilosort 使用指南与常见问题解决方案

> 基于 Kilosort GitHub 仓库 1000+ 个已关闭 Issues 的总结分析。
> 涵盖 Kilosort 1/2/2.5/3/4 各版本。

---

## 一、安装与环境配置

### 1.1 标准安装方法

**Kilosort4 (Python 版) — 推荐方式：**

```bash
# 创建 conda 环境
conda create --name kilosort python=3.9
conda activate kilosort

# 安装（含 GUI）
python -m pip install kilosort[gui]

# 仅 API（无 GUI）
python -m pip install kilosort
```

**关键注意事项：**
- 推荐 Python 3.9–3.11（Python 3.13 可能存在兼容性问题）
- 使用 Intel/AMD GPU 必须有 NVIDIA CUDA 支持的 PyTorch
- Apple Silicon (M1/M2/M3/M4) 目前不完全支持，MPS 后端有已知问题

### 1.2 PyTorch GPU 安装

```bash
# 先卸载 CPU 版本的 PyTorch（如有）
pip uninstall torch torchvision torchaudio

# 安装 CUDA 版本（以 CUDA 12.1 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 常见安装问题及解决

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| `pip install kilosort[gui]` 失败 (Linux) | PyQt6 在旧 Ubuntu (18.04) 上无法编译 | 升级到 Ubuntu 20.04+，或先手动 `pip install PyQt5` |
| `conda create` 失败或环境不可用 | conda 不再推荐安装 PyTorch | 使用 `pip` 安装 PyTorch，而非 `conda install` |
| `no Qt platform plugin` 错误 | 缺少 Qt 系统库 | `sudo apt install libxcb-cursor0` (Linux)；或安装 `PyQt5` 替代 |
| PyTorch 不识别 CUDA | CUDA 版本不匹配 | 确保 PyTorch CUDA 版本与系统驱动兼容；检查 `torch.cuda.is_available()` |
| `numpy.dtype size changed` | NumPy 版本不兼容（由依赖包混用引起） | 在全新 conda 环境中安装，避免与系统包冲突 |
| Apple Silicon 不支持 | PyTorch MPS 后端 FAISS 不兼容 | 目前无完整解决方案，建议使用 Intel/AMD + NVIDIA GPU |
| `threadpoolctl` 警告 | scikit-learn 依赖版本 | 无害警告，可以忽略 |
| 首次启动 404 错误 | 下载 `wTEMP.npz` 模板文件失败 | 确保网络连接正常；文件托管在 `kilosort.org` |

---

## 二、GUI 启动与使用

### 2.1 启动 GUI

```bash
conda activate kilosort
python -m kilosort
```

### 2.2 GUI 启动失败常见问题

| 问题 | 解决方法 |
|------|----------|
| `python -m kilosort` 无反应 | 检查 conda 环境是否激活；检查 PyQt 是否正确安装 |
| `UnboundLocalError: local variable 'path' referenced before assignment` | 重新安装/更新到最新版本；或在启动前手动设置 `data_path` |
| `qt.qpa.screen` 错误 | 忽略（仅在远程/无头服务器上影响），或设置 `QT_QPA_PLATFORM=offscreen` |
| 大数据文件加载后 GUI 卡死 | 分批加载，耐心等待 |
| GUI 启动后无法再次启动 | 重启终端/conda 环境 |

### 2.3 GUI 使用流程

1. **选择二进制文件** — 支持 `.bin`, `.dat`, `.bat`, `.raw` 格式
2. **选择探针布局** — 预设或自定义（`.mat`, `.prb`, `.json`）
3. **检查/调整参数** — 主参数面板 + 额外设置窗口
4. **加载数据** — 点击 LOAD 按钮加载并验证数据
5. **预览** — 查看原始数据、白化数据、探针布局
6. **运行排序** — 点击 Run 开始全流程排序
7. **查看结果** — 漂移图、诊断图、spike 位置图
8. **在 Phy 中策展** — 运行 `phy template-gui params.py`

---

## 三、数据准备

### 3.1 支持的格式

- **二进制文件**: `.bin`, `.dat`, `.bat`, `.raw`
- **数据类型**: `int16`（默认），`uint16`, `int32`, `float32` 等
- **数据要求**: 形状为 `(n_channels, n_samples)` 的连续二进制文件（C 序）

### 3.2 数据转换

Kilosort4 GUI 内置转换器支持从以下格式转换：
- Intan `.rhd` / `.rhs`
- OpenEphys 二进制
- SpikeGLX 二进制
- Neuralynx `.ncs`

```python
# 通过 API 使用 SpikeInterface 转换
from spikeinterface.extractors import read_openephys
from spikeinterface.preprocessing import bandpass_filter, common_reference
```

### 3.3 数据准备常见问题

| 问题 | 解决方法 |
|------|----------|
| `Bytes in binary file did not divide evenly` | `n_chan_bin` 设置不正确，确认总通道数 |
| float32 数据 spike 检测失败 | 设置 `shift` 和 `scale` 参数使数据范围在 [-100, +100] |
| 多文件记录（同一 session 分段文件） | 将文件名以列表形式传入 `filename` 参数 |
| `.rhd` 文件转换失败 | 确保文件完整且使用最新版 SpikeInterface；或先导出为 `.dat` |
| HDF5 `.h5` 文件 | 先通过 SpikeInterface 转换为 `.bin` 格式 |
| 需要排除同步通道 | 使用 `bad_channels` 参数排除 SYNC 通道 |
| 数据过大（200+ GB） | 考虑分析部分数据（设置 `tmin`, `tmax`），或使用 `batch_downsampling` |

---

## 四、探针配置

### 4.1 内置探针

Kilosort4 内置支持：
- Neuropixels 1.0 (多种模式)
- Neuropixels 2.0（单/多shank）
- Neuropixels 1.0 NHP（非人灵长类长探头）

### 4.2 自定义探针

三种格式：`.json` / `.mat` / `.prb`

**JSON 格式（推荐）：**
```json
{
    "chanMap": [0, 1, 2, ..., 383],
    "xc": [0.0, 0.0, ..., 0.0],
    "yc": [0.0, 20.0, ..., 7660.0],
    "kcoords": [1.0, 1.0, ..., 1.0],
    "n_chan": 384
}
```

**关键字段：**
- `chanMap`: 数据中每个通道在二进制文件中的行索引（0-indexed）
- `xc`, `yc`: 通道的 x, y 坐标（微米）
- `kcoords`: shank 编号（同一 shank 上所有通道使用相同的值，如 1, 2, 3...）
- `n_chan`: 通道总数

### 4.3 探针配置常见问题

| 问题 | 解决方法 |
|------|----------|
| `chanMap` 超出二进制通道范围 | 确保 `chanMap` 为 0-indexed，且最大值 < `n_chan_bin` |
| 新建探针 GUI 功能不可用 | 使用 JSON 文件手动创建 |
| 从 SpikeInterface 生成的探针无法加载 | KS4 探针格式略有不同，需转换为简单字典格式 |
| 多 shank 探针配置 | 每个 shank 使用不同的 `kcoords` 值；可结合 `shank_idx` 分别排序 |
| 双面探针（double-sided） | 分 shank 分别排序 |
| 通道间距过大（如 300μm）无法检测 spike | KS4 针对高密度探针优化，低密度需调整参数 |
| Neuropixels 2.0 4-shank | 使用内置探针，或设置 `nblocks` 和 `x_centers` |
| 2D 阵列探针（MEA/Utah阵列） | 必须手动设置 `x_centers`，并考虑增大 `dminx` |

---

## 五、排序结果解读

### 5.1 输出文件

| 文件 | 说明 |
|------|------|
| `spike_times.npy` | spike 时间（采样点） |
| `spike_clusters.npy` | 每个 spike 的 cluster ID |
| `spike_templates.npy` | 每个 spike 的 template ID |
| `amplitudes.npy` | spike 振幅 |
| `templates.npy` | 每个 cluster 的平均模板波形 |
| `channel_positions.npy` | 通道位置 |
| `channel_map.npy` | 通道映射 |
| `pc_features.npy` | PC 特征 |
| `pc_feature_ind.npy` | PC 特征对应的通道索引 |
| `similar_templates.npy` | cluster 间相似度矩阵 |
| `params.py` | Phy 参数文件 |
| `kilosort4.log` | 详细运行日志 |

### 5.2 Cluster 标签

Phy 中将 cluster 分为三类：
- **good**: 通过不应期测试，可能是单个神经元
- **mua** (multi-unit activity): 未通过不应期测试，可能是多神经元混合
- **noise**: 噪声 cluster

### 5.3 常用 Phy 操作

```bash
# 启动 Phy
phy template-gui params.py

# 常用快捷键
# 在 Phy 中：
#   f — 切换 feature view
#   w — 切换波形 view
#   c — 切换 correlogram view
#   g — 标记为 "good"
#   m — 标记为 "mua"
#   alt+click — 分裂 cluster
#   space — 合并选中的 clusters
```

---

## 六、各版本差异与兼容性

| 特性 | KS2/2.5 (MATLAB) | KS3 (MATLAB) | KS4 (Python) |
|------|-------------------|--------------|--------------|
| 语言 | MATLAB + CUDA | MATLAB + CUDA | Python + PyTorch |
| GPU 要求 | NVIDIA CUDA | NVIDIA CUDA | NVIDIA CUDA（CPU 也可） |
| 漂移校正 | KS2.5 有，KS2 无 | 有 | 有（改进版） |
| 输出格式 | Phy | Phy | Phy |
| GUI | MATLAB GUI | MATLAB GUI | Qt5/Qt6 GUI |
| 安装难度 | 高（需编译CUDA） | 高 | 低 |

---

## 七、常见问题分类及解决方案

### 7.1 Spike 检测问题

#### 7.1.1 未检测到 spike / "No spikes detected"

**最常见的错误之一，原因多样：**

| 可能原因 | 解决方案 |
|----------|----------|
| `Th_universal` / `Th_learned` 太高 | 降低阈值至 `7-8`；低 SNR 数据降至 `6` |
| 通道间距过大（稀疏探针） | 减小 `dmin`（垂直间距）和 `dminx`（水平间距）为实际通道间距的 0.5-1 倍 |
| 数据比例问题（float32） | 设置 `shift` 和 `scale` 使数据落入 [-100, +100] |
| 白化后数据全为 0 / 空白 | 检查探针 `chanMap` 是否正确对应数据通道；检查 `n_chan_bin` |
| `TruncatedSVD` 错误 (0 samples) | 降低 `Th_single_ch` 到 `4-5`；确保数据有 spike 信号 |
| 通道数太少（4-16 通道） | KS4 针对高密度探针优化；设置 `nearest_chans < 通道数`，降低 `dminx` |
| 高通滤波过于激进 | 降低 `highpass_cutoff` 至 `150-200` Hz |
| 信号极性反转 | 设置 `invert_sign=True` |

#### 7.1.2 检测到太少 spike / 漏检

| 可能原因 | 解决方案 |
|----------|----------|
| 阈值过高 | 降低 `Th_universal` (如 7-8) 和 `Th_learned` (如 6-7) |
| 窄波形（fast-spiking）未被检测 | 减小 `min_template_size` (如 5) |
| 模板覆盖不足 | 减小 `dmin`、`dminx`；增加 `template_sizes` (如 7) |
| 2D MEA / Utah 阵列 | 增大 `dminx` 至等于通道间距；确保 `x_centers` 正确 |
| 长时间记录中丢失 cluster | 增加 `nblocks` (如 5-10) 助力漂移跟踪 |

#### 7.1.3 检测到太多噪声 / 伪迹

| 可能原因 | 解决方案 |
|----------|----------|
| 阈值太低 | 提高 `Th_universal` 至 `10-12`；提高 `Th_learned` 至 `9-10` |
| 光遗传学/电刺激伪迹 | 设置 `artifact_threshold` (如 500-1000) 归零超过阈值的 batch |
| 电容式舔舐传感器干扰 | 屏蔽/移除干扰通道；在记录时暂时关闭传感器 |
| 出现 ripple 振荡时的异常检测 | 提高 `highpass_cutoff` 至 `400-500` Hz |

### 7.2 聚类问题

#### 7.2.1 过度合并 (Over-merging)

**症状：不同神经元的 spike 被合并到同一 cluster**

| 解决方案 |
|----------|
| 降低 `ccg_threshold` (如 `0.15-0.2`)，使合并条件更严格 |
| 降低 `acg_threshold` (如 `0.1`)，使 "good" 标准更严格 |
| 在 Phy 中手动分裂合并的 cluster |
| 确保 `cluster_neighbors` 足够大（≥10） |

#### 7.2.2 过度分裂 (Over-splitting)

**症状：同一神经元的 spike 被分成多个 cluster**

| 解决方案 |
|----------|
| 提高 `ccg_threshold` (如 `0.3-0.4`)，使合并条件更宽松 |
| 提高 `acg_threshold` (如 `0.25-0.3`) |
| 在 Phy 中手动合并相似的 cluster |
| 如果问题在 KS4 中比 KS2 严重，尝试降低 `Th_learned` |

#### 7.2.3 KS4 比 KS2.5 产出更少的 "good" unit

**这是一个被多次报告的常见问题：**

| 解决方案 |
|----------|
| KS4 的标签标准比 KS2.5 更保守，"good" 单位质量更高但数量可能更少 |
| 降低 `acg_threshold` 和 `ccg_threshold` |
| 确认探针配置相同（KS4 与 KS2 探针格式略有不同） |
| 对于非 Neuropixels 探针（如 32ch 多 shank），结果可能差距较大 |
| 在 Phy 中手动重新标记：KS4 的 "mua" cluster 中可能包含实际可用的 unit |

#### 7.2.4 跨通道分裂/重复检测

**症状：同一 spike 被检测到出现在多个不同 channel 的 cluster 中**

| 解决方案 |
|----------|
| 增大 `max_channel_distance` 可以有助于合并 |
| 在 Phy 中通过 CCG 检查并手动合并 |
| 这是一种保守策略的副作用，通常无害 |

#### 7.2.5 所有 cluster 被标记为噪声

| 解决方案 |
|----------|
| 检查数据质量：是否有大量噪声/伪迹 |
| 检查白化是否正常（查看 GUI 白化数据视图） |
| 提高 `Th_universal` 以减少噪声进入聚类 |
| 降低 `acg_threshold` 使更多 cluster 通过不应期测试 |

### 7.3 漂移校正问题

#### 7.3.1 漂移校正方向错误 / 反而变差

| 现象 | 解决方案 |
|------|----------|
| 漂移校正将数据沿错误方向移动 | 检查探针 yc 坐标方向是否正确（通常 y 值越大代表越深）；确保 shank 配置正确 |
| 漂移估计过大 | 增大 `drift_smoothing` (如 `[0.5, 1.0, 1.0]`) |
| `nblocks=1` 漂移不准确 | 增加至 `3-5` 可获得更精细的块内漂移估计 |
| 慢速漂移未被跟踪 | `nblocks=0` 关闭漂移校正时，模板仍有缓慢更新，但校正不如开 `nblocks>1` 好 |

#### 7.3.2 禁用漂移校正

设置 `nblocks=0` 可完全跳过漂移校正步骤。适用于：
- 短时间记录（<5 分钟）
- 急性记录，无显著漂移
- 漂移校正产生错误结果时

### 7.4 内存与性能问题

#### 7.4.1 GPU 内存不足 (CUDA OOM)

| 解决方案 |
|----------|
| 减小 `batch_size`（如 `30000` 或 `20000`） |
| 减小 `cluster_neighbors`（如 `5`） |
| 减小 `max_cluster_subset`（如 `10000`） |
| 设置 `clear_cache=True`（不推荐，会降低性能） |
| 使用 `batch_downsampling > 1` 减少处理的数据量 |
| 通过 `shank_idx` 按 shank 分别排序 |
| 升级 GPU 显存（推荐 ≥ 8GB，大通道数推荐 ≥ 24GB） |

#### 7.4.2 系统内存 (RAM) 使用过高

| 现象 | 解决方案 |
|------|----------|
| RAM 随时间逐渐增长至 128GB+ | 可能存在内存泄漏；更新到最新版本；重启 Python 进程 |
| 大文件 RAM 持续高位 | 通过 `tmin`/`tmax` 分段处理 |
| 32GB RAM 不足 | 对于 1000+ 通道 × 数小时记录，建议 ≥ 64GB RAM |

#### 7.4.3 排序速度太慢

| 可能原因 | 解决方案 |
|----------|----------|
| 未使用 GPU | 确认 PyTorch 正确检测到 CUDA；`torch.cuda.is_available()` 应为 `True` |
| 使用 CPU 版本 PyTorch | 卸载后重新安装 CUDA 版 PyTorch |
| GPU 显存太小 (< 8GB) | 减小 `batch_size`，减小 `cluster_downsampling` |
| `max_cluster_subset` 太大 | 设为 `25000` 或更低 |
| GUI 比 API 慢 | GUI 有额外开销；大批量排序推荐用 API |
| 聚类阶段特别慢 | 检查 `cluster_downsampling`（默认 `20` 即可）；减少 `cluster_neighbors` |
| NVIDIA RTX 5060 系列 | 确保 CUDA 与 PyTorch 版本兼容 |
| 同一时间运行多个 KS 实例 | GPU 内存竞争；串行运行 |

#### 7.4.4 硬件推荐

| 通道数 | GPU 显存 | 系统 RAM | 存储 |
|--------|----------|----------|------|
| < 64 通道 | ≥ 4 GB | 16 GB | 快速 NVMe SSD |
| 64–384 通道 | ≥ 8 GB | 32 GB | 快速 NVMe SSD |
| 384–1024 通道 | ≥ 16 GB | 64 GB | 快速 NVMe SSD + 大容量 HDD |
| > 1000 通道 | ≥ 24 GB | 128 GB | 快速 NVMe SSD + 大容量 HDD |

---

### 7.5 运行时错误及解决方案

#### 7.5.1 常见错误速查表

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `ValueError: Found array with 0 sample(s) ... TruncatedSVD` | 未找到足够好的单通道阈值穿越，无 spike 样本用于 PCA | 降低 `Th_single_ch`；检查数据是否有信号；检查探针配置 |
| `n_samples=5 should be >= n_clusters=6` | 检测到的 spike 过少（少于 `n_templates`） | 降低 `Th_single_ch`、提高检测灵敏度 |
| `ZeroDivisionError: float division by zero` | 除零错误，通常发生在聚类或振幅归一化阶段 | 更新到最新版本（此 bug 在多个版本中修复过） |
| `IndexError: shape of mask does not match` | 张量维度不匹配 | 通常是数据问题或版本 bug，检查数据格式 |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | `nt` 与 `wPCA` 维度不匹配 | 确保 `nt` 为奇数；`nt=60` 会导致此错误 |
| `The size of tensor a must match the size of tensor b` | 通道数和模板维度不匹配 | 通常与探针配置错误有关 |
| `cannot reshape tensor` | pytorch reshape 失败 | 检查数据格式和 `n_chan_bin` |
| `CUDNN_STATUS_NOT_SUPPORTED` | cuDNN 操作不支持 | 通常可忽略，结果应该仍然有效 |
| `DLL load failed while importing _multiarray_umath` | NumPy/PyTorch DLL 冲突 | 在全新 conda 环境中重新安装 |
| `OMP_NUM_THREADS` / `OpenBLAS` 警告 | 多线程库冲突 | 设置环境变量 `OMP_NUM_THREADS=1` |
| `qt.qpa.window: SetProcessDpiAwarenessContext() failed` | Qt DPI 设置问题（Windows） | 无害警告，可忽略 |
| `ValueError: Unrecognized settings` | 传入了不支持的参数名 | 检查拼写错误；确保参数名与 `DEFAULT_SETTINGS` 一致 |

#### 7.5.2 路径与文件问题

| 问题 | 解决方法 |
|------|----------|
| 日志文件缺失（`kilosort4.log`） | 确保 results 目录有写权限 |
| 结果保存到错误路径 | 明确指定 `results_dir` 参数 |
| 排序后文件无法移动/删除 | Python 进程可能仍持有文件句柄，重启 Python |
| 跨操作系统加载 `ops.npy` 失败 | KS4 将路径转为字符串存储；若仍有问题，手动修改 `params.py` |
| `wTEMP.npz` 下载失败 (403/404) | 网络问题或服务器暂时不可用；手动下载放置到 `~/.kilosort/` 目录 |
| `.prb` 文件中 geometry 键的排序问题 | 确保 geometry 和 chanMap 的顺序匹配 |

#### 7.5.3 Phy 兼容性问题

| 问题 | 解决方法 |
|------|----------|
| `IndexError` 在 Phy 中打开 KS4 结果 | 检查 `params.py` 中的 `dat_path` 是否正确 |
| Phy 中 `TemplateFeatureView` 不可用 | 确认 KS4 保存了 `pc_features.npy` |
| KS4 结果在 Phy 中看起来比 KS2 差 | KS4 的预处理和特征计算方式不同；在 Phy 中可能需要不同的策展策略 |
| 无法安装 Phy | `pip install phy --pre`（安装预发布版本） |
| `cluster_Amplitude.tsv` 解释 | 每行包含 cluster_id 和该 cluster 的振幅值 |

---

### 7.6 数据质量问题

#### 7.6.1 处理后看到的奇怪波形

| 现象 | 可能原因 |
|------|----------|
| 波形呈阶梯状或块状 | `nblocks` 与 `temp_wh.dat` 的交互导致 |
| 簇的振幅范围异常大（混合高低振幅 spike） | KS4 可能欠合并；在 Phy 中分裂或调整 `ccg_threshold` |
| spike 时间超过记录长度 | batch 边界填充导致的正常现象，可以安全截断超出的 spike |
| 错误通道上的波形 | 探针 `chanMap` 与数据通道顺序不匹配 |
| KS4 和 KS2 的模板不同 | 两个版本的模板提取方法不同，这可能是正常的 |

#### 7.6.2 生物记录相关问题

| 场景 | 建议 |
|------|------|
| 自由移动动物 + 稀疏通道图 | KS4 对此场景可能不如 KS2.5；尝试恢复默认密集通道图 |
| 小脑浦肯野细胞（复杂 spike）| 增加 `nt`（如 81-101）捕获更长的复杂 spike 波形 |
| 强光遗传学刺激 | 设置 `artifact_threshold` 归零光刺激期间的 batch |
| Opto-electrode 光伪迹 | 使用 `artifact_threshold` 或在预处理阶段手动移除光刺激时段 |
| 麻醉状态慢性记录 | 漂移较小，`nblocks=1-2` 可能足够 |
| 清醒行为记录（显著漂移） | 增加 `nblocks` 到 `5`，可能需要手动检查漂移校正质量 |

---

## 八、调参策略总结

### 8.1 快速调参流程图

```
排序结果不理想
    │
    ├─ 没有检测到 spike？
    │   ├─ 降低 Th_universal 和 Th_learned
    │   ├─ 检查数据比例（设置 shift/scale）
    │   ├─ 检查探针 chanMap 和 n_chan_bin
    │   └─ 减小 dmin/dminx
    │
    ├─ 检测到太多噪声？
    │   ├─ 提高 Th_universal 和 Th_learned
    │   ├─ 设置 artifact_threshold
    │   └─ 排除坏通道（bad_channels）
    │
    ├─ 过度分裂？
    │   ├─ 提高 ccg_threshold
    │   └─ 在 Phy 中手动合并
    │
    ├─ 过度合并？
    │   ├─ 降低 ccg_threshold
    │   └─ 在 Phy 中分裂
    │
    ├─ 漂移校正有问题？
    │   ├─ 调整 nblocks
    │   ├─ 调整 drift_smoothing
    │   └─ 或直接禁用：nblocks=0
    │
    ├─ 内存不足？
    │   ├─ 减小 batch_size
    │   ├─ 减小 cluster_neighbors
    │   ├─ 减小 max_cluster_subset
    │   └─ 按 shank 分别排序
    │
    └─ 运行太慢？
        ├─ 确认使用 GPU
        ├─ 增大 cluster_downsampling
        ├─ 减小 max_cluster_subset
        └─ 使用更快的硬件
```

### 8.2 各探针类型推荐参数

| 参数 | Neuropixels 1/2 | 高密度线阵 (32-128ch) | Tetrode (4ch) | MEA 2D 阵列 | Utah 阵列 |
|------|-----------------|----------------------|---------------|-------------|-----------|
| `dminx` | 32 | 32-50 | 20 | 实际间距或更大 | 实际间距 |
| `dmin` | auto | auto | 10-20 | 1 (无深度) | auto |
| `nearest_chans` | 10 | 10 | 4 | 8-10 | 10-16 |
| `max_channel_distance` | 32 | 40-80 | 50 | 100-200 | 100-200 |
| `nblocks` | 1-5 | 1-5 | 1 | 1 | 1 |
| `x_centers` | auto (1) | 1 | 1 | 手动指定 | 手动指定 |
| `Th_universal` | 9 | 9 | 6-7 | 8-10 | 8-10 |

---

## 九、最佳实践与建议

### 9.1 排序建议

1. **先用默认参数运行**，再根据结果调整
2. **检查白化数据视图**：正常白化后应清晰看到 spike 波形
3. **检查漂移散点图**：确认漂移估计是否合理
4. **始终在 Phy 中策展**：KS 的自动标签是初筛，手动确认至关重要
5. **保存 log 文件**：`kilosort4.log` 包含所有运行细节
6. **每次参数改动后重新排序**，对比效果

### 9.2 版本建议

- 始终使用最新稳定版：`pip install --upgrade kilosort`
- 如果最新版本有问题，可以回退：`pip install kilosort==4.0.3`
- KS4 vs KS2.5 选择：
  - KS4 适合 Neuropixels 高密度探针、自动化流程
  - KS2.5 适合非标准探针、需要更灵活参数控制的场景

### 9.3 常用调试技巧

```python
# 1. 检查 PyTorch 是否正确使用 GPU
import torch
print(torch.cuda.is_available())  # 应为 True
print(torch.cuda.get_device_name(0))

# 2. 检查默认设置
from kilosort import DEFAULT_SETTINGS
print(DEFAULT_SETTINGS.keys())

# 3. 查看已修改的参数
from kilosort.parameters import compare_settings
modified, extra = compare_settings(your_settings)
print("Modified:", modified)
print("Extra keys:", extra)

# 4. 加载已有结果
from kilosort import run_kilosort, load_sorting
ops, st, clu, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    load_sorting('path/to/kilosort4/')
```

---

## 十、参考资源

- **官方文档**: https://kilosort.readthedocs.io/
- **GitHub 仓库**: https://github.com/MouseLand/Kilosort
- **Phy 文档**: https://phy.readthedocs.io/
- **SpikeInterface**: https://spikeinterface.readthedocs.io/
- **示例数据**: https://www.kilosort.org/downloads

---

> 本文档基于 Kilosort GitHub 仓库 1000+ 个已关闭 Issues 的分析整理（截至 2026 年 5 月）。
> 涉及关键 Issue 编号范围：#401–#1034，覆盖 Kilosort 1/2/2.5/3/4 各版本。
