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
