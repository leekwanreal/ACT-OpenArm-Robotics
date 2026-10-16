# 🚀 HƯỚNG DẪN KẾT NỐI LAN, CÀI ĐẶT MÔI TRƯỜNG, HUẤN LUYỆN ACT VÀ TỐI ƯU ĐIỀU KHIỂN (WINDOWS ⟷ UBUNTU SERVER)

> **Mục đích**: Hướng dẫn toàn diện từ cấu hình mạng LAN trực tiếp (`toann@192.168.0.107`), cài đặt môi trường, đăng ký WandB, đồng bộ mã nguồn 2 camera ($480 \times 640$), huấn luyện lưu đầy đủ weights (PyTorch `.pth` + ONNX `.onnx`), cơ chế Train & Eval đồng thời, và phân tích khắc phục triệt để hiện tượng giật rung khi suy luận (Inference).

---

## 📑 MỤC LỤC

1. [Bước 1: Cấu hình IP mạng LAN trên máy Windows (192.168.0.x)](#bước-1-cấu-hình-ip-mạng-lan-trên-máy-windows-1921680x)
2. [Bước 2: Kết nối SSH vào Server Ubuntu (`toann@192.168.0.107`)](#bước-2-kết-nối-ssh-vào-server-ubuntu-toann1921680107)
3. [Bước 3: Đồng bộ mã nguồn mới (2 Camera 480x640, YOLOv8 pure PyTorch) lên Server](#bước-3-đồng-bộ-mã-nguồn-mới-từ-windows-lên-server)
4. [Bước 4: Kiểm tra môi trường Server & Cài đặt bổ sung](#bước-4-kiểm-tra-môi-trường-server--cài-đặt-bổ-sung)
5. [Bước 5: Lấy Model Weights về máy (Tải trực tiếp hoặc qua WandB Cloud)](#bước-5-lấy-model-weights-về-máy-tải-trực-tiếp-hoặc-qua-wandb-cloud)
6. [Bước 6: Cơ chế Train & Eval đồng thời (Tính điểm khi Save Checkpoint)](#bước-6-cơ-chế-train--eval-đồng-thời-tính-điểm-khi-save-checkpoint)
7. [Bước 6.5: Kiểm tra định dạng file HDF5 trước khi huấn luyện (HDF5 Inspector)](#bước-65-kiểm-tra-định-dạng-file-hdf5-trước-khi-huấn-luyện-hdf5-inspector)
8. [Bước 7: Lệnh chạy huấn luyện GPU hoàn chỉnh trên Server (Chuẩn 2 Camera 480x640)](#bước-7-lệnh-chạy-huấn-luyện-gpu-hoàn-chỉnh-trên-server-chuẩn-2-camera-480x640)
9. [Bước 8: Lệnh chạy đánh giá mô hình (Evaluation trên tập Test/Val)](#bước-8-lệnh-chạy-đánh-giá-mô-hình-evaluation-trên-tập-testval)
10. [Bước 9: Phân tích & Khắc phục hiện tượng Giật khi Inference Robot](#bước-9-phân-tích--khắc-phục-hiện-tượng-giật-khi-inference-robot)
11. [Phụ lục: Trả lại cấu hình IP mặc định ban đầu cho Windows](#phụ-lục-trả-lại-cấu-hình-ip-mặc-định-ban-đầu-cho-windows)

---

## BƯỚC 1: CẤU HÌNH IP MẠNG LAN TRÊN MÁY WINDOWS (192.168.0.x)

Khi cắm dây mạng LAN nối trực tiếp giữa cổng mạng của máy Windows và máy Server Ubuntu (`192.168.0.107`), bạn cần gán cho card mạng Ethernet trên máy Windows một địa chỉ IP tĩnh cùng dải `192.168.0.x` (ví dụ `192.168.0.100`).

### Cách 1: Đổi nhanh bằng PowerShell (Khuyên dùng - 3 giây)
1. Nhấn phím `Windows`, gõ **PowerShell**, nhấp chuột phải chọn **Run as Administrator**.
2. Card mạng cắm dây LAN (ASIX USB to Gigabit) trên máy bạn có tên là **`Ethernet 3`**. Chạy lệnh sau:
   ```powershell
   New-NetIPAddress -InterfaceAlias "Ethernet 3" -IPAddress 192.168.0.100 -PrefixLength 24
   ```
   *(Nếu báo IP đã tồn tại, chạy lệnh cập nhật: `Set-NetIPAddress -InterfaceAlias "Ethernet 3" -IPAddress 192.168.0.100 -PrefixLength 24`)*

### Cách 2: Đổi qua Giao diện Đồ họa (GUI Windows)
1. Nhấn tổ hợp phím **`Windows + R`**, nhập:
   ```text
   ncpa.cpl
   ```
   rồi nhấn **Enter**.
2. Nhấp chuột phải vào biểu tượng card mạng **Ethernet** -> chọn **Properties**.
3. Trong danh sách, nhấp đúp chuột vào dòng **Internet Protocol Version 4 (TCP/IPv4)**.
4. Tích chọn vào mục **Use the following IP address**:
   - **IP address**: `192.168.0.100` (hoặc số bất kỳ từ 2 đến 254, khác 107)
   - **Subnet mask**: `255.255.255.0`
   - **Default gateway**: Để trống
5. Bấm **OK** -> Bấm tiếp **OK** để lưu lại.

### Kiểm tra kết nối thông mạng:
Mở Terminal / PowerShell thường, gõ:
```bash
ping 192.168.0.107
```
Nếu thấy phản hồi dạng:
```text
Reply from 192.168.0.107: bytes=32 time<1ms TTL=64
```
👉 **Hai máy đã thông mạng LAN trực tiếp với băng thông 1Gbps!**

---

## BƯỚC 2: KẾT NỐI SSH VÀO SERVER UBUNTU (`toann@192.168.0.107`)

### 1. Lệnh kết nối SSH
Mở PowerShell hoặc Command Prompt trên Windows và gõ:
```bash
ssh toann@192.168.0.107
```
Nhập mật khẩu của user `toann` khi được hỏi.

> [!TIP]
> **Xử lý nếu bị lỗi "WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!":**  
> Chạy lệnh xóa key cũ:
> ```bash
> ssh-keygen -R 192.168.0.107
> ```
> rồi thực hiện lại lệnh `ssh toann@192.168.0.107`.

### 2. Khởi tạo phiên làm việc `tmux` trên Server
Để tiến trình huấn luyện tiếp tục chạy ngầm ổn định ngay cả khi bạn rút dây LAN hoặc gập máy:
```bash
# 1. Tạo phiên làm việc tmux mới
tmux new -s act_train

# 2. Để thoát ra màn hình ngoài mà train vẫn chạy: Nhấn Ctrl + B, rồi nhấn D
# 3. Để vào lại xem tiến trình train:
tmux attach -t act_train
```

---

## BƯỚC 3: ĐỒNG BỘ MÃ NGUỒN MỚI TỪ WINDOWS LÊN SERVER

Thư mục `act_training` đã được cập nhật toàn diện:
- **Chuẩn 2 Camera RGB-D**: Chest (`chest_rgb`, `chest_depth`) và Front (`front_rgb`, `front_depth`) ở độ phân giải **$480 \times 640$**.
- **Vision Backbone YOLOv8-Nano Pure PyTorch**: Tự động trích xuất đặc trưng 4 kênh RGB-D mà không cần cài thêm ultralytics.
- **7-layer Transformer Decoder**: Chuẩn Stanford ALOHA ACT bimanual (16 động cơ).
- **Lưu đồng thời đầy đủ weights**: PyTorch checkpoint (`.pth`) + ONNX model (`.onnx`) + Metadata (`.json`) + Stats (`.pkl`).
- **Tự động đồng bộ weights lên WandB Artifacts (Cloud)**.

### Lệnh đẩy thư mục code mới từ Windows lên Server:
Mở một cửa sổ PowerShell mới trên máy Windows (tại thư mục gốc của repository):
```powershell
scp -r c:\Users\Admin\Desktop\20261\VinRobotics\OpenArmReal\act_training toann@192.168.0.107:/home/toann/code/OpenArm_ACT_Training/
```
*(Nếu server dùng đường dẫn khác, hãy thay `/home/toann/code/OpenArm_ACT_Training/` bằng đường dẫn thư mục code thực tế trên server)*.

---

## BƯỚC 4: KIỂM TRA MÔI TRƯỜNG SERVER & CÀI ĐẶT BỔ SUNG

### 1. Kích hoạt môi trường ảo Python trên Server:
```bash
# Truy cập vào thư mục dự án trên server:
cd /home/toann/code/OpenArm_ACT_Training/act_training

# Kích hoạt virtual environment:
source venv/bin/activate
```

### 2. Các thư viện CẦN CÀI THÊM:
Chạy lệnh sau trong môi trường `(venv)`:
```bash
pip install wandb matplotlib
```
- **`wandb`**: Ghi nhận metrics thời gian thực và tự động lưu trữ model weights lên Cloud.
- **`matplotlib`**: Vẽ biểu đồ cột sai số 16 khớp (Per-Joint MAE bar chart) tự động trong quá trình eval.

### 3. Những thư viện KHÔNG CẦN CÀI:
- ❌ **Không cần cài `ultralytics`**: Backbone YOLOv8 RGB-D đã được viết bằng **Pure PyTorch** độc lập.
- ❌ **Không cần cài `timm`**: Mặc định mô hình dùng `yolov8_nano` siêu nhẹ (~2.1M params), không phụ thuộc vào `timm`.

---

## BƯỚC 5: LẤY MODEL WEIGHTS VỀ MÁY (TẢI TRỰC TIẾP HOẶC QUA WANDB CLOUD)

Mô hình sau khi huấn luyện sẽ lưu toàn bộ weights tại `--output_dir` (mặc định: `checkpoints/act_openarm/`):

### 📋 Danh sách file weights được tạo ra:
| Tên File | Dung lượng | Công dụng |
| :--- | :---: | :--- |
| **`best_checkpoint.pth`** | ~437 MB | **PyTorch weights tốt nhất**: Chứa toàn bộ state_dict, optimizer, scheduler, scaler và dataset stats. |
| **`latest_checkpoint.pth`** | ~437 MB | **PyTorch weights mới nhất**: Dùng để tiếp tục huấn luyện nếu bị ngắt quãng (`--resume auto`). |
| **`best_checkpoint.onnx`** | ~124 MB | **Mô hình ONNX triển khai robot**: Dùng trực tiếp cho Node ROS 2 inference 50Hz trên robot thực tế. |
| **`best_checkpoint_metadata.json`** | ~4 KB | Chứa toàn bộ thông số chuẩn hóa Z-score (mean/std), min/max độ sâu và kích thước ảnh 480x640. |
| **`dataset_stats.pkl`** | ~1 KB | Thống kê Z-score chuẩn của dataset để load trực tiếp trong Python. |

### Cách 1: Tải trực tiếp bằng lệnh SCP trên PowerShell Laptop (Khuyên dùng khi cắm LAN)
```powershell
# Tạo thư mục lưu weights trên máy:
New-Item -ItemType Directory -Force -Path "C:\Users\Admin\Desktop\20261\VinRobotics\OpenArmReal\act_training\checkpoints\act_openarm"

# Tải toàn bộ weights từ server về laptop:
scp -r toann@192.168.0.107:/home/toann/code/OpenArm_ACT_Training/act_training/checkpoints/act_openarm/* "C:\Users\Admin\Desktop\20261\VinRobotics\OpenArmReal\act_training\checkpoints\act_openarm"
```

### Cách 2: Tải qua Cloud WandB (Không cần cắm lại dây LAN)
Nếu bạn bật cờ `--use_wandb`, hệ thống tự động đẩy toàn bộ file weights lên **WandB Artifacts**:
1. Mở trình duyệt web tại nhà vào: **[wandb.ai](https://wandb.ai)**
2. Chọn Project **`act-openarm`** -> Nhấp vào biểu tượng **Artifacts** 📦 ở menu bên trái.
3. Chọn Artifact **`act_best_model`** -> Chọn tab **Files** -> Bấm **Download** để tải về máy.

---

## BƯỚC 6: CƠ CHẾ TRAIN & EVAL ĐỒNG THỜI (TÍNH ĐIỂM KHI SAVE CHECKPOINT)

Hệ thống huấn luyện mới đã tích hợp sẵn cơ chế đánh giá song song tự động:

1. **Đánh giá mỗi khi lưu Checkpoint (`--eval_every 10` hoặc `--save_every 10`)**:
   - Tự động chạy đánh giá trên tập Validation không có gradient (`torch.no_grad()`).
   - Tính toán đầy đủ: **Total Loss**, **Reconstruction L1 Loss**, **KL Divergence Loss**, và **Per-Joint MAE** (sai số từng khớp).
   - Tự động vẽ biểu đồ bar chart 16 khớp đẩy lên WandB (`val_analysis/per_joint_mae`).
   - Ghi ảnh mẫu RGB-D của cả 2 camera ngực và đối diện lên WandB (`val_samples/chest_rgbd`, `val_samples/front_rgbd`).
   - So sánh nếu Loss tốt hơn kỷ lục trước $\rightarrow$ cập nhật `best_checkpoint.pth`, xuất `best_checkpoint.onnx`, và đồng bộ ngay lên WandB Cloud.

2. **Đánh giá toàn diện trên Checkpoint cuối cùng (Final Evaluation)**:
   - Khi hoàn thành toàn bộ epochs, pipeline chạy final evaluation chi tiết.
   - Hiển thị bảng đánh giá độ chính xác từng khớp theo chuẩn ALOHA:
     - Khớp tay: sai số mục tiêu $< 3^\circ$ (Radian & Độ).
     - Kẹp gắp Gripper: sai số mục tiêu $< 1.5\text{ mm}$.

---

## BƯỚC 6.5: KIỂM TRA ĐỊNH DẠNG FILE HDF5 TRƯỚC KHI HUẤN LUYỆN (HDF5 INSPECTOR)

Trước khi bấm chạy huấn luyện hàng trăm epochs, bạn có thể chạy công cụ kiểm tra tự động [inspect_hdf5.py](file:///c:/Users/Admin/Desktop/20261/VinRobotics/OpenArmReal/inspect_hdf5.py) để bóc thử 1 file HDF5 trong thư mục dataset và kiểm tra toàn bộ cấu trúc:

```bash
# 1. Bóc thử cấu trúc file HDF5 đầu tiên trong folder:
python inspect_hdf5.py --folder /đường/dẫn/tới/thư_mục_data_hdf5

# 2. Hoặc kiểm tra trực tiếp 1 file chỉ định:
python inspect_hdf5.py --file /đường/dẫn/tới/file_episode_0.hdf5

# 3. Chạy kiểm tra kết hợp Forward Pass qua Dataset và Mô hình ACT:
python inspect_hdf5.py --folder /đường/dẫn/tới/thư_mục_data_hdf5 --test_forward
```

**Bảng tiêu chí được script tự động kiểm tra:**
- `action`: Phải có shape `[T, 16]` (chuẩn 16 bậc tự do 2 tay OpenArm), không chứa NaN/Inf.
- `observations/qpos`: Phải có shape `[T, 16]` (chuẩn 16 góc khớp thực tế).
- `observations/images/chest`: Ảnh camera trước ngực chuẩn $480 \times 640$.
- `observations/images/front`: Ảnh camera đối diện (nếu thiếu, pipeline tự động duplicate camera chest).
- `frequency / timing`: Kiểm tra tần số 50.0 Hz và độ giật lag/rớt frame (tự động bật `auto_resample` nếu jitter).

---

## BƯỚC 7: LỆNH CHẠY HUẤN LUYỆN GPU HOÀN CHỈNH TRÊN SERVER (CHUẨN 2 CAMERA 480x640)

Chạy lệnh sau bên trong phiên `tmux` trên Server:

```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --dataset_dir /đường/dẫn/tới/thư_mục_data_hdf5 \
    --output_dir checkpoints/act_openarm \
    --backbone_type yolov8_nano \
    --yolo_stride 16 \
    --decoder_layers 7 \
    --img_height 480 \
    --img_width 640 \
    --num_cameras 2 \
    --chunk_size 50 \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 4 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --kl_weight 10.0 \
    --warmup_epochs 15 \
    --val_split 0.15 \
    --eval_every 10 \
    --save_every 10 \
    --use_wandb \
    --wandb_project act-openarm \
    --wandb_name act_openarm_2cam_480x640 \
    --device cuda
```

> [!NOTE]
> - Mặc định cờ lưu file PyTorch (`--save_pth`) đã được **BẬT sẵn**, bạn không cần thêm cờ `--save_pth`.
> - Nếu GPU của server có VRAM $\ge 24\text{GB}$ (ví dụ RTX 3090, RTX 4090, A5000), bạn có thể nâng `--batch_size` lên `48` hoặc `64`.
> - **Cờ bổ sung đặc trưng 3D (`--use_3d_features`)**:
>   - **Mặc định (Không có cờ)**: Mô hình chỉ chạy 2D Backbone đã chọn (YOLOv8-Nano thuần túy) với chi phí tính toán thấp nhất.
>   - **Khi thêm cờ `--use_3d_features`**: Mô hình tự động kích hoạt `Point3DEncoder`, tạo ra 600 tokens điểm 3D Plücker $P = (\hat{u} Z, \hat{v} Z, Z)$ từ độ sâu cả 2 camera để đưa vào Transformer Memory cùng Visual tokens.

---

## BƯỚC 8: LỆNH CHẠY ĐÁNH GIÁ MÔ HÌNH (EVALUATION TRÊN TẬP TEST/VAL)

Bạn có thể chạy kiểm thử độc lập mô hình đã huấn luyện trên tập dataset bằng `eval.py`:

### Đánh giá trên file checkpoint PyTorch (`.pth`):
```bash
python3 eval.py \
    --checkpoint_path checkpoints/act_openarm/best_checkpoint.pth \
    --dataset_dir /đường/dẫn/tới/thư_mục_data_hdf5 \
    --output_dir evaluation_results \
    --batch_size 16 \
    --use_wandb
```

### Đánh giá trên file mô hình ONNX (`.onnx`):
```bash
python3 eval.py \
    --checkpoint_path checkpoints/act_openarm/best_checkpoint.onnx \
    --dataset_dir /đường/dẫn/tới/thư_mục_data_hdf5 \
    --output_dir evaluation_results \
    --batch_size 16
```
Kết quả đánh giá bao gồm:
- Độ chính xác L1, MSE cho từng khớp.
- Kiểm tra giả lập **Temporal Ensembling** $50\text{ Hz}$ thời gian thực.
- Tỷ lệ giảm giật quỹ đạo (**Jerk Reduction %**).
- File báo cáo đầy đủ: `evaluation_results/eval_report.json`.

---

## BƯỚC 9: PHÂN TÍCH & KHẮC PHỤC HIỆN TƯỢNG GIẬT KHI INFERENCE ROBOT

### 1. Hiện tượng thực tế gặp phải:
- **Khi để profile `default`**: Robot chạy bị **rung giật cơ học**, nhưng lực đủ mạnh, vẫn vươn tay tới vị trí vật thể và kẹp được.
- **Khi chuyển sang PD gains thấp (hoặc dùng `interpolator: filter4`)**: Chuyển động nhìn **mượt hơn**, nhưng robot **không còn chính xác, thậm chí không giơ nổi tay lên để kẹp**.

---

### 2. Nguyên nhân kỹ thuật cốt lõi (Root Cause):

Động cơ Damiao trên OpenArm điều khiển ở chế độ **MIT Impedance Mode** qua CAN bus:
$$\tau = K_p (q_{des} - q_{cur}) + K_d (\dot{q}_{des} - \dot{q}_{cur}) + \tau_{ff}$$
Trong code điều khiển hiện tại (`controller_node.py`), tham số feedforward torque luôn đặt là: $\tau_{ff} = 0.0$ (không có bù trọng lực Gravity Compensation).

1. **Tại sao `default` bị GIẬT nhưng NÂNG ĐƯỢC TAY?**
   - Profile `default` có $K_p$ lớn: Khớp vai J1, J2 có $K_p = 75.0$, khuỷu J3, J4 có $K_p = 50.0$.
   - Khi $K_p$ cao, mô-men phục hồi $\tau = K_p (q_{des} - q)$ đủ lớn để **thắng được trọng lực cánh tay** (trọng lượng các link kim loại và 7 động cơ Damiao). Vì vậy sai số do trọng lực $e_{gravity} \approx \frac{\tau_{gravity}}{K_p}$ nhỏ, tay nâng lên được.
   - **Lý do giật**: Policy ACT dự đoán các action chunk rời rạc. Mỗi khi có bước nhảy nhỏ giữa các chunk hoặc nhiễu nội suy ở 400Hz, với $K_p = 75$ và $K_d = 2.8$, motor sinh ra biến thiên mô-men đột ngột $\Delta \tau$ rất lớn (High Jerk) gây rung giật cơ học.

2. **Tại sao hạ PD gains / dùng `filter4` lại MƯỢT nhưng "KHÔNG GIƠ NỔI TAY LÊN"?**
   - **Hiện tượng kéo trĩu do trọng lực (Gravity Sagging)**: Khi hạ $K_p$ xuống thấp (ví dụ profile `act` có $K_p = 20.0$), độ cứng lò xo của khớp giảm gần 4 lần! Không có bù trọng lực $\tau_{ff}$, sai số tĩnh do trọng lực:
     $$e_{ss} = \frac{\tau_{gravity}(q)}{K_p}$$
     tăng vọt gấp 4 lần. Dù policy ra lệnh $q_{des}$ là "giơ tay lên cao", motor sinh lực quá yếu, tay bị trọng lực kéo trĩu xuống, không bao giờ đạt tới vị trí kẹp.
   - **Hiện tượng trễ pha do bộ lọc (`filter4`)**: Bộ lọc `filter4` (4th-order critically damped filter) với băng thông mặc định $w = 30\text{ rad/s}$ gây ra độ trễ pha:
     $$\Delta t_{lag} \approx \frac{4}{w} \approx \frac{4}{30} \approx 133\text{ ms}$$
     Độ trễ 133ms tương đương với gần 7 chu kỳ điều khiển ACT ($50\text{ Hz}$). Độ trễ này khiến tín hiệu setpoint bám sau lệnh thực tế quá xa, kết hợp với $K_p$ thấp khiến robot phản ứng chậm chạp và mất độ chính xác.

---

### 3. Giải pháp giải quyết triệt để (Standard Best Practices):

> [!CAUTION]
> **RÀNG BUỘC AN TOÀN PHẦN CỨNG BẮT BUỘC:**  
> **TUYỆT ĐỐI KHÔNG ĐIỀU CHỈNH HOẶC BƠM MÔ-MEN XOẮN TRỰC TIẾP ($\tau_{ff} \equiv 0.0$)**. Việc điều chỉnh sai torque có thể làm vỡ bánh răng hộp số của động cơ Damiao. Toàn bộ quá trình làm mượt **chỉ thực hiện ở miền vị trí setpoint $q(t)$** và cân bằng tỉ số Damping ảo $K_d / K_p$, giữ nguyên chế độ MIT Impedance chuẩn của OpenArm.

Để robot **vừa mượt mà, vừa sắc bén, không bị ngu đi và đủ lực nâng tay lên kẹp**, áp dụng 3 giải pháp an toàn sau:

#### Giải pháp 1: Tinh chỉnh PD Gains phân tầng theo khớp (`act_tuned`)
Không cào bằng $K_p$ cho tất cả các khớp. Các khớp gốc (vai, khuỷu) giữ độ cứng cao để chống sụt trọng lực, đồng thời **tăng $K_d$ (damping)** để đưa hệ thống về trạng thái tắt dần tới hạn ($\zeta \approx 1.0$), triệt tiêu rung giật mà không làm mềm tay:

Mở file: `openarm_mc/src/openarm_controller/config/gains.yaml`, thiết lập `active_profile: act_tuned`:
```yaml
      act_tuned:
        left_j1:          # DM8009 Vai (nâng cả cánh tay)
          kp: 65.0
          kd: 3.6         # Tăng Damping để triệt tiêu rung giật
        left_j2:          # DM8009 Vai
          kp: 65.0
          kd: 3.6
        left_j3:          # DM4340 Khuỷu
          kp: 45.0
          kd: 2.8
        left_j4:          # DM4340 Khuỷu
          kp: 45.0
          kd: 2.8
        left_j5:          # DM4310 Cổ tay
          kp: 25.0
          kd: 1.6
        left_j6:          # DM4310 Cổ tay
          kp: 25.0
          kd: 1.6
        left_j7:          # DM4310 Cổ tay
          kp: 25.0
          kd: 1.6
        left_gripper:
          speed: 10.0
          effort: 0.15
        # (Áp dụng tương tự cho cánh tay phải right_j1 .. right_gripper)
```

#### Giải pháp 2: Sử dụng Quintic Spline Interpolator (Độ trễ pha = 0 ms)
Trong `openarm_controller`, giữ nguyên `interpolator: quintic` (mặc định). Bộ nội suy Quintic khớp đa thức bậc 5 mượt mà tại 400Hz liên tục $C^2$ cả về vị trí, vận tốc và gia tốc mà **hoàn toàn KHÔNG tạo ra độ trễ pha 133ms như `filter4`**. Nhờ đó robot phản xạ tức thời theo quan sát hình ảnh thời gian thực ("không bị ngu đi").

#### Giải pháp 3: Bật Temporal Ensembling với hệ số $k$ chuẩn trong ACT Node
Trong Node `openarm_act` (`act.yaml`), đảm bảo Temporal Ensembling luôn được kích hoạt:
- Action Chunking tổng hợp các chunk gối đầu nhau theo trọng số hàm mũ $w_i = \exp(-k \cdot i)$.
- Đặt `ensemble_k: 0.01` và `weighting: newest`. Dự đoán mới nhất luôn có trọng số cao nhất (đáp ứng nhanh), đồng thời làm phẳng hoàn toàn các bước nhảy không liên tục giữa các chunk.
- Duy trì `vel_scale: 1.0` để robot hoạt động đúng nhịp độ thao tác 50Hz.

---

### 4. Về Camera Intrinsics & Tọa độ Tia Chuẩn Hóa (Plücker Lines)

File HDF5 không lưu camera intrinsics $K$ vì camera là phần cứng cố định (fixed rig). Khi triển khai nâng cấp 3D sau này, có thể xử lý như sau:
1. **Lấy trực tiếp từ RealSense ROS 2**:
   ```bash
   ros2 topic echo /camera/chest/color/camera_info --once
   ros2 topic echo /camera/front/color/camera_info --once
   ```
2. **Dùng Tọa độ tia chuẩn hóa / Plücker Lines $(\mathbf{d}, \mathbf{m}) \in \mathbb{R}^6$ (KHÔNG CẦN INTRINSICS)**:
   Chuẩn hóa pixel $(u, v)$ về $[-1, 1]$:
   $$\hat{u} = \frac{u - 320}{320}, \quad \hat{v} = \frac{v - 240}{240}, \quad P_{ray} = (\hat{u} \cdot Z, \; \hat{v} \cdot Z, \; Z)$$
   Vector này đồng dạng affine với không gian 3D metric thực tế. Lớp MLP/Linear trong mạng nơ-ron tự động học được hệ số tỉ lệ tiêu cự $f_x, f_y$ trong quá trình huấn luyện End-to-End mà không cần biết chính xác thông số calib.

---

## PHỤ LỤC: TRẢ LẠI CẤU HÌNH IP MẶC ĐỊNH BAN ĐẦU CHO WINDOWS

Khi huấn luyện xong hoặc muốn rút dây LAN để cắm vào modem mạng / Wi-Fi bình thường, bạn trả lại cấu hình IP tự động (DHCP):

### Cách 1: Bằng PowerShell (Run as Administrator) - 1 giây:
```powershell
Set-NetIPInterface -InterfaceAlias "Ethernet 3" -Dhcp Enabled
Set-DnsClientServerAddress -InterfaceAlias "Ethernet 3" -ResetServerAddresses
```

### Cách 2: Bằng Giao diện Đồ họa (GUI):
1. Bấm `Win + R`, gõ `ncpa.cpl` -> Enter.
2. Chuột phải vào card mạng **Ethernet 3** (ASIX USB to Gigabit) -> **Properties** -> nhấp đúp **Internet Protocol Version 4 (TCP/IPv4)**.
3. Chọn lại 2 dòng:
   - 🔘 **Obtain an IP address automatically**
   - 🔘 **Obtain DNS server address automatically**
4. Bấm **OK** -> **OK**. Card mạng sẽ tự nhận lại mạng bình thường.
