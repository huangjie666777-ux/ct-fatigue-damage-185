# 平行束 CT 滤波反投影重建服务

纯后端实现：Python 3.10 + FastAPI 0.115.12 + NumPy 2.2.6（滤波反投影 FBP 全部用 NumPy 手写，不调用任何现成重建函数）。

## 运行

    .venv/bin/python -m uvicorn ctrecon.app:app --host 127.0.0.1 --port 8000

## 输入格式

`POST /reconstruct`，`multipart/form-data`：

| 字段 | 说明 |
| --- | --- |
| `file` | NPZ，含三个数组 |
| `detector_spacing_mm` | 探测器单元间距（毫米，>0） |
| `center_index` | 旋转中心对应的小数探测器索引 |
| `output_size` | 输出图像边长（像素，1–256） |
| `pixel_spacing_mm` | 输出像素间距（毫米，>0） |
| `filter` | `ram-lak`（默认）或 `hann` |

NPZ 数组：

- `intensity`：二维数组，形状 **角度 × 探测器**，即 `(n_angles, n_detectors)`。
- `dark`、`flat`：一维数组，长度等于探测器宽度。

角度约定：**从 0 开始、等间距覆盖 180°、不含终点**，即
`theta_k = k * pi / n_angles`（k = 0 … n_angles−1）。

限制与校验（全部返回 HTTP 422）：

- 角度数 2–360，探测器数 2–512，输出边长 1–256。
- 拒绝对象数组（`allow_pickle=False`）、形状不匹配、含 NaN/Inf 的数组。
- 拒绝非有限或非正的间距、非有限中心索引、未知滤波器。
- ZIP/NPZ 解压后总大小上限 64 MiB。

## 标定（线积分）

逐探测器执行

    T(theta, i) = (I(theta, i) - dark[i]) / (flat[i] - dark[i])
    p(theta, i) = -ln(T)

- 要求每个探测器 `flat > dark`，否则拒绝。
- 透射率必须有限且严格为正，否则拒绝。
- 透射率 > 1 时**保留负线积分**，不裁剪，也不把坏值当零。

## 重建（FBP）

- 滤波器在 FFT 网格上按**真实探测器间距**构造频率 `f = k / (N * d)`（cycles/mm），Ram-Lak 响应 `2|f|`（截止于探测器 Nyquist `1/(2d)`）；Hann 在 Ram-Lak 上乘 Hann 窗。
- 每行 sinogram FFT **补零到至少 2 倍探测器长度**（取 2 的幂），避免循环卷积混叠。
- 反投影按 `t = x*cos(theta) + y*sin(theta)` 线性插值，探测器范围外取 0；按角度步长 `pi/n_angles` 积分。卷积与角度积分均带真实物理步长，输出单位为**每毫米线性衰减系数 mm^-1**。
- 不做逐图归一化；保留负重建值。

### 坐标与平行束覆盖范围

- 图像以**几何中心为原点**：列向右为 +x，行向上为 +y（数组第 0 行对应 +y）。
- 0° 时射线法向为 +x，探测器坐标 `t = (detector_index - center_index) * detector_spacing_mm`。
- 平行束在 0–180°（不含 180°）内等角距采样即可覆盖完整物体：平行射线在 `theta` 与 `theta+180°` 方向的投影等价，因此只需半圆采样。FOV 半径约为旋转中心两侧的有效探测器宽度；偏心或超出探测器轨迹的结构不会被完整采样。

## 输出（ZIP）

响应为 `application/zip`，包含：

- `reconstruction.npy`：float64 二维数组（`numpy.save` 格式），物理单位 mm^-1。
- `preview.png`：8 位灰度预览，按图像 min/max 做**仅用于显示**的线性拉伸，不影响 NPY。
- `metadata.json`：回显参数及图像/sinogram 数值范围与单位。

## 解析示例（偏心圆盘）

均匀圆盘（半径 R、衰减 μ、圆心 (cx, cy)）的解析投影为弦长公式：

    p(theta, t) = 2 * mu * sqrt(R^2 - (t - p0)^2),  p0 = cx*cos(theta) + cy*sin(theta)
                 （|t - p0| <= R，否则为 0）

生成示例 NPZ：

    .venv/bin/python examples/offcenter_disk_demo.py

## 测试

    .venv/bin/python -m compileall -q ctrecon examples tests
    .venv/bin/python -m pytest -q

## FBP 单位修正

滤波响应为 |f|（cycles/mm），FFT/IFFT 对本身携带连续傅里叶变换的 1/(N·d) 与 d 因子，因此滤波结果**不再额外乘探测器间距**。此前版本响应 2|f| 再乘 d，仅在 d=0.5 时碰巧正确；修正后任意探测器间距下输出均为正确的 mm^-1。

## 双能材料分解：POST /decompose

接收**已对齐**的低、高能扫描（不做图像配准），逐像素把两幅衰减图分解为两种材料的非负密度图。

`multipart/form-data` 字段：

| 字段 | 说明 |
| --- | --- |
| `low_file` / `high_file` | 低/高能 NPZ，格式同 /reconstruct，两者 intensity 形状必须相同 |
| `detector_spacing_mm`、`center_index`、`output_size`、`pixel_spacing_mm`、`filter` | 共同几何参数，限制同 /reconstruct |
| `materials` | JSON，两个唯一非空材料名，如 `["aluminum","plastic"]` |
| `mu_matrix` | JSON 2×2 质量衰减系数矩阵，**行为低/高能、列为材料**，单位 mm²/mg，元素须有限且严格为正 |
| `slice_thickness_mm` | 截面厚度（毫米，>0） |
| `rois` | JSON，1–8 个唯一命名矩形 `{"name","x0","y0","x1","y1"}`，像素索引，左上含、右下不含 |

校验（均返回 422）：

- 两份 NPZ 各自沿用 /reconstruct 的全部大小与非法输入限制，且形状必须一致。
- `mu_matrix` 的 2 范数条件数 > 10000 直接拒绝，不使用正则化掩盖不可辨识性。
- ROI 越界、空区、重名、数量超出 1–8 均拒绝。

### 线性双材料假设

忽略射束硬化，假设能量 e 的每毫米衰减为材料分密度的线性组合：

    mu_e(x, y) = sum_m mu_matrix[e][m] * rho_m(x, y)

逐像素求精确 2×2 非负最小二乘解 rho >= 0（mg/mm³）：无约束解可行时取之；否则在两个单材料边界解中选平方误差最小者，**不是**把负分量简单截零。衰减图负值全部保留；输出每个能量的残差图（预测 − 观测，mm^-1）。

### 区域计量与输出（ZIP）

每个 ROI 按 `质量 = Σ rho × 像素面积 × 截面厚度` 积分各材料质量（mg），并给出各能量平均残差。ZIP 内容：

- `density_<材料名>.npy`：两份 float64 密度图（mg/mm³）。
- `residual_low.npy` / `residual_high.npy`：两能量残差图（mm^-1）。
- `preview_<材料名>.png`：各材料 8 位灰度预览，仅显示用 min/max 拉伸，不改定量数据。
- `metadata.json`：回显参数、单位及各 ROI 的质量与平均残差。

### 解析示例（两材料混合）

`examples/dual_material_demo.py` 生成两个不同材料圆盘的解析投影（弦长公式按密度加权，再经 mu_matrix 线性组合成低/高能线积分）：

    .venv/bin/python examples/dual_material_demo.py

脚本打印可直接使用的 curl 命令。
