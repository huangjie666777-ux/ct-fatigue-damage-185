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

## 复合梁截面载荷校核：POST /section_check

在双能分解的基础上做截面强度校核：材料分布 → 体积分数 → 组合截面刚度 → 轴力 + 双向弯曲联合求解 → 各材料拉/压许用比与超限位置。

`multipart/form-data` 字段（几何与 `mu_matrix` 同 /decompose）：

| 字段 | 说明 |
| --- | --- |
| `low_file` / `high_file` | 低/高能 NPZ，格式同 /decompose |
| `mask_file` | NPZ，含与重建同尺寸的**布尔**数组 `mask`；掩膜外不参与截面 |
| `materials` | JSON，两个唯一命名对象，各含 `reference_density_mg_per_mm3`、`elastic_modulus_mpa`、`allowable_tension_mpa`、`allowable_compression_mpa`（均须有限且严格为正） |
| `load_cases` | JSON，1–8 个唯一命名工况 `{"name","N","Mx","My"}`，单位 N 与 N·mm，须有限 |

### 力学模型与假设

- **线弹性、完全粘结、平截面假设**：`ε = ε0 + κx·y − κy·x`，有效应力为各材料 `Σ φ_m E_m ε`。
- 体积分数 `φ_m = ρ_m / ρ_ref_m`；**总和 > 1 时按比例归一，< 1 时保留空隙**；掩膜外置零，原密度图不改。
- 坐标以图像中心为原点，x 向右、y 向上，取像素中心；`N = ∫σ dA`、`Mx = ∫yσ dA`、`My = ∫−xσ dA`，3×3 刚度矩阵**联合求解**，不忽略耦合项。
- 每工况、每材料在其存在区域（φ>0 且掩膜内）统计拉/压极值与最大许用比 `max(σ/σ_allow)`，比值 ≤ 1 为合格，并给出超限位置（像素与毫米坐标）。

### 校验（均返回 422，不返回伪结果）

- 掩膜非布尔、尺寸不符、为空；载荷非有限、工况重名或数量超出 1–8；材料参数非正或非有限。
- 截面奇异（掩膜内材料分布无法平衡轴力与双向弯曲，如单像素截面）直接拒绝。
- `mu_matrix` 行不齐等非法 JSON 结构同样返回 422（此前会触发 HTTP 500）。

### 输出（ZIP）

- `stress_<工况>_<材料>.npy`：该材料应力图（MPa，float64），材料不存在处为 NaN。
- `exceedance_<工况>.png`：许用比预览，0–1 灰度、超限像素标红，仅显示用。
- `report.json`：3×3 刚度矩阵、各工况应变/曲率（ε0、κx、κy）、平衡残差、各材料拉压极值与许用比及超限位置、总体合格结论。

### 解析示例（偏心双材料截面）

    .venv/bin/python examples/section_check_demo.py

生成偏心铝盘 + 塑料盘的低/高能 NPZ 与圆形布尔掩膜，并打印 curl 命令；`service` 工况合格、`overload` 工况超限。

## 复合梁循环疲劳：POST /fatigue

在 `/section_check` 同一材料分解、体积分数、坐标和耦合刚度之后新增疲劳分析；旧接口路径、表单字段与 ZIP 文件保持不变。疲劳端点只接收 `output_size <= 64` 的正方形网格。

### 输入（multipart/form-data）

| 字段 | 说明 |
| --- | --- |
| `low_file` / `high_file` | 低/高能 NPZ，沿用 `/reconstruct` 的 `intensity`、`dark`、`flat` |
| `mask_file` | NPZ，布尔数组 `mask`，尺寸必须等于重建网格 |
| `history_file` | NPZ，含同长度、有序、有限的一维数组 `N`、`Mx`、`My` |
| `R` | 块重复次数；必须是纯十进制正整数（无小数点、无科学计数法） |
| 几何字段 | `detector_spacing_mm`、`center_index`、`output_size`、`pixel_spacing_mm`、`filter` |
| `mu_matrix` | 与 `/section_check` 相同的 2×2 衰减矩阵 |
| `materials` | 两个材料对象，除原有物性外，还需正有限值 `Sref`、`Nref`、`m`、`Su`，Sref 与 Su 均为 MPa |

历史数组长度必须为 2–2000，三个数组按相同时间顺序配对。单位为：N（N，拉为正）、Mx/My（N·mm）、所有应力和疲劳强度参数（MPa）。非有限值、对象数组、布尔历史数组、长度不匹配、非正整数 `R`、空掩膜、奇异截面或超过 64×64 的疲劳网格均返回 HTTP 422。

### 计数与损伤假设

- 每个历史步将完整的同步 `[N, Mx, My]` 联合代入同一个 3×3 耦合刚度方程，得到共同应变场；不能分别对 N、Mx、My 计数后相加。
- 各材料在其存在像素（φ>0）的应力历史为 `σ_m(x,y,t)=E_m ε(x,y,t)`。
- 逐像素先删除相邻连续重复值，保留首尾端点和峰谷转折；然后按 ASTM E1049 三反转雨流法计数。闭合循环计数 1，残余反转计数 0.5；不连接首尾。
- 循环幅值 `a=(σmax−σmin)/2`、均值 `s=(σmax+σmin)/2`；零幅值不产生损伤。
- Goodman 校正：`a' = a / (1 − max(s,0)/Su)`；若分母小于等于 0，拒绝整个请求。
- S-N 寿命：`Nf = Nref (Sref/a')^m`；Miner 单块损伤 `D=Σ count/Nf`，累计损伤为 `R·D`。
- 按相互独立的重复块假设，JSON 中 `total_block_life = 1/D` 表示该载荷块可重复的块数；若最大损伤点 `D=0`，返回 `null` 且 `zero_damage=true`。另给 `cumulative_life_in_repeated_blocks=(1/D)/R`，仅用于把寿命换算到“重复后块”尺度。

### 输出（ZIP）

- `damage_<材料名>.npy`：该材料存在区域的累计损伤 `R·D`，不存在处为 NaN。
- `damage_preview.png`：两材料逐像素最大单块损伤的显示预览；0–1 映射灰度，超过 1 标红，不替代 NPY。
- `fatigue_report.json`：输入 SHA-256、刚度矩阵、材料参数、最大累计损伤、材料名、像素行列与毫米坐标、单块损伤、总块寿命、该最大点的全部雨流循环（幅值、均值、Goodman 幅值、count、Nf、损伤）、单位和假设。

### 解析示例与 curl

    .venv/bin/python examples/fatigue_demo.py

脚本生成 64×64 疲劳截面用低/高能 NPZ、掩膜和同步载荷历史 NPZ，并打印可直接执行的 curl 命令。
