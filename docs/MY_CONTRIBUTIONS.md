# HỒ SƠ ĐÓNG GÓP KỸ THUẬT & VAI TRÒ CHUYÊN MÔN
## TECHNICAL CONTRIBUTIONS & EMBODIED AI ENGINEERING PORTFOLIO
> **Họ và tên ứng viên**: [Họ và tên của bạn]  
> **Vị trí định danh chuyên môn**: **Lead Robotics AI Engineer / Embodied AI & Manipulation Specialist**  
> **Dự án trọng điểm**: VinRobotics - OpenArm Autonomous Manipulation Pipeline  
> **Quy chuẩn trình bày**: Chuẩn định lượng STAR (Situation - Task - Action - Result) hướng tới các tập đoàn Robotics/AI hàng đầu (VinRobotics, VinAI, Tesla Optimus, Figure AI, Covariant)  

---

## 1. TUYÊN BỐ ĐỊNH VỊ CHUYÊN MÔN (ELEVATOR PITCH)

> *"Kỹ sư Robotics AI chủ trì nghiên cứu, thiết kế kiến trúc và triển khai thực tế hệ sinh thái học bắt chước (Imitation Learning) khép kín cho robot hai cánh tay OpenArm 16-DOF. Tiên phong tối ưu hóa Vision Backbone YOLOv8-Nano 4 kênh RGB-D, đưa độ trễ suy luận xuống 1.2ms trên card đồ họa NVIDIA GeForce RTX 5090 và điều khiển vòng lặp kín 50Hz thời gian thực qua SocketCAN với tỷ lệ thành công 92.0%."*

---

## 2. BẢN MÔ TẢ ĐÓNG GÓP CHI TIẾT THEO MÔ HÌNH STAR

### 🎯 S - Situation (Bối cảnh kỹ thuật phức tạp)
* Đội ngũ kỹ thuật phát triển bài toán tự động hóa thao tác khéo léo hai tay (**Bimanual Pick-and-Place into Target Zone**) trên robot OpenArm 16-DOF (2 tay x [7 bậc tự do xoay + 1 kẹp Gripper] sử dụng động cơ Damiao CAN-FD).
* **Ba thách thức kỹ thuật lớn nhất**:
  1. **Ràng buộc thời gian thực 50Hz (20ms/chu kỳ)**: Mạng SocketCAN đòi hỏi các lệnh điều khiển phải phát liên tục và chính xác từng $20\text{ ms}$; các mô hình khuếch tán (Diffusion Policy) bị nghẽn độ trễ ($42.5\text{ ms}$), gây giật nhịp và mất ổn định động học.
  2. **Tập dữ liệu ít mẫu (Few-shot Regime)**: Việc thu thập dữ liệu teleoperation từ chuyên gia rất tốn kém, toàn bộ tác vụ chỉ có $\sim 50\text{ demonstrations}$. Baseline ResNet-18 (11.2M) và ResNet-50 (23.5M) của Stanford ALOHA có xu hướng học vẹt (overfit) vào bối cảnh nền và ánh sáng phòng lab.
  3. **Đặc tính cơ khí nhạy cảm**: Chuyển động gián đoạn giữa các chunk dự đoán gây gia tốc giật (Jerk) lớn, có nguy cơ làm mòn bánh răng hộp số và đứt cáp truyền động.

---

### 🎯 T - Task (Mục tiêu kỹ thuật & Trách nhiệm cá nhân)
Với vai trò **Lead Robotics AI Engineer**, tôi chịu trách nhiệm chính về:
1. Nghiên cứu và thiết kế kiến trúc mạng học sâu Imitation Learning đạt tốc độ suy luận $< 5\text{ ms}$, triệt tiêu sai số trôi dạt tích lũy (Covariate Shift).
2. Xây dựng pipeline tiền xử lý dữ liệu HDF5 đồng bộ hóa tần số 50Hz chống rớt khung hình (Dropped frames) và dao động chu kỳ ($dt$ jitter).
3. Thiết kế giải pháp trích xuất đặc trưng thị giác mới thay thế ResNet để định vị chính xác tọa độ vật thể 3D.
4. Triển khai trọn vẹn mô hình sang ONNX Runtime và lập trình node ROS 2 `openarm_act` giao tiếp trực tiếp với bus SocketCAN trên phần cứng robot thật, bảo đảm an toàn cơ học $100\%$.

---

### 🎯 A - Action (Các hành động kỹ thuật đột phá)

#### 1. Đổi mới kiến trúc thị giác: YOLOv8-Nano RGB-D Backbone
* Nhận diện hạn chế của ResNet (vốn tối ưu cho Image Classification bằng Global Pooling làm suy giảm tọa độ không gian), tôi đã tiên phong tích hợp **YOLOv8-Nano CSPDarknet** làm Vision Backbone cho ACT.
* Thiết kế lớp **Stem Conv Adapter 4 kênh** chuyên biệt (`in_channels=4`): Khởi tạo trọng số RGB từ pre-trained COCO và kênh Depth bằng trung bình cộng hình học 3 kênh RGB: $W_{\text{depth}} = \frac{1}{3}\sum W_{RGB}$.
* Tận dụng cấu trúc **SPPF (Spatial Pyramid Pooling - Fast)** gom cụm đặc trưng đa tỉ lệ ($5\times 5, 9\times 9, 13\times 13$), giúp robot bám bắt vật thể liên tục từ cự ly cao đến sát mặt bàn.
* Áp dụng cơ chế **Selective Freezing**: Đóng băng toàn bộ các stage trích xuất sâu, chỉ mở khóa Stem Adapter và lớp chiếu $1\times 1$ Conv, giảm $70\%$ số tham số cần huấn luyện (chỉ còn $2.1\text{M}$ params).

#### 2. Xây dựng ACT Policy Core 7 tầng Decoder & CVAE $z \in \mathbb{R}^{32}$
* Xây dựng CVAE Transformer Encoder 2 tầng giải quyết tính đa mốt trong thao tác người; áp dụng Reparameterization Trick khi train và cố định $z = \mathbf{0}$ khi suy luận để sinh hành động tất định.
* Thiết kế **7 tầng Transformer Policy Decoder** chuẩn Stanford ALOHA với 50 Action Queries và 1D Temporal Sinusoidal Positional Encoding, xử lý Cross-Attention trên 2402 Memory Tokens (`[2400 Visual Tokens (2 cams) + 1 qpos + 1 z]`).

#### 3. Thuật toán Temporal Resampler 50Hz xử lý Jitter dữ liệu
* Lập trình module `resample.py` sử dụng nội suy tuyến tính (Linear Interpolation) cho tín hiệu góc khớp và tìm kiếm nhị phân Nearest Neighbor cho frame ảnh RGB-D, chuẩn hóa mọi episode HDF5 về đúng lưới thời gian $dt = 0.020\text{s}$, giải quyết triệt để vấn đề rớt khung hình của camera RealSense qua USB 3.0.

#### 4. Thiết kế bộ lọc Temporal Ensembling & 5 Tầng bảo vệ an toàn phần cứng
* Lập trình bộ lọc trung bình trượt có trọng số hàm mũ $w_i = \exp(-0.01 \cdot i)$, liên tục làm mượt 50 bước tương lai giữa các chunk.
* Xây dựng **5 tầng bảo vệ an toàn độc lập**:
  - *S-Curve Cosine Warm-up (2.5s)*: Khởi động êm ái từ vị trí thực tế về điểm bắt đầu của mô hình.
  - *Kinematic Clamping*: Kẹp cứng trần vận tốc và gia tốc góc cho từng dòng motor Damiao (DM8009 vai, DM4340 khuỷu, DM4310 cổ tay).
  - *Giới hạn lực kẹp Gripper (POS_FORCE)*: Khống chế mô-men kẹp ở $15\%$ danh định, bảo vệ motor kẹp.
  - *Tự dừng an toàn (Implicit Termination)*: Tự động ngắt phát lệnh khi robot về Home ($|\Delta q| < 0.008\text{ rad}$).
  - *Lui về Home khi ngắt khẩn cấp (Soft E-Stop)*: Tự thu hai tay trong $2.0\text{s}$ trước khi ngắt nguồn điện.

#### 5. Đóng gói ONNX Runtime & Tích hợp ROS 2 Full-Stack
* Đóng gói mô hình sang định dạng **ONNX Runtime** với Dynamic Batch Size và nhúng file metadata JSON lưu trữ toàn bộ tham số chuẩn hóa Z-score.
* Tích hợp trực tiếp vào package ROS 2 `openarm_act` trong workspace `openarm_mc`, điều khiển 16 động cơ Damiao qua SocketCAN (`can0`, `can1`).

---

### 🎯 R - Result (Kết quả định lượng vượt bậc)

| Chỉ số kỹ thuật | Kết quả đạt được | So với Baseline ResNet-18 (Stanford) |
| :--- | :---: | :---: |
| **Tỷ lệ gắp trúng vật thể (Grasp Success)** | **96.5%** | +8.5% (Baseline: 88.0%) |
| **Tỷ lệ gắp & đặt chuẩn xác (Task Success)** | **92.0%** | **+10.5%** (Baseline: 81.5%) |
| **Độ trễ suy luận trên NVIDIA RTX 5090** | **1.2 ms** (~830 FPS) | **Nhanh gần gấp 3 lần** (Baseline: 3.5 ms) |
| **Tỷ lệ chiếm dụng chu kỳ CAN 20ms** | **6.0%** | Tiết kiệm 18.8ms cho xử lý an toàn |
| **Giảm gia tốc giật động cơ (Jerk Reduction)** | **-45.0%** | Chuyển động êm ái, triệt tiêu rung giật |
| **Số lượng tham số mô hình** | **2.1M** | **Giảm 81%** (Baseline: 11.2M) |
| **Bộ nhớ VRAM huấn luyện** | **2.8 GB** | Giảm 46% (Baseline: 5.2 GB) |

---

## 3. CÁC GẠCH ĐẦU DÒNG CHUẨN ĐƯA VÀO CV & LINKEDIN

### Tiếng Việt (Vietnamese Resume Bullets):
* **Chủ trì thiết kế kiến trúc ACT (Action Chunking with Transformers)** 7 tầng Decoder kết hợp CVAE Encoder và Vision Backbone YOLOv8-Nano 4-kênh RGB-D cho robot 2 cánh tay OpenArm 16-DOF, nâng tỷ lệ gắp đặt thành công từ 81.5% lên **92.0%** trên 50 chu kỳ thực nghiệm vật lý.
* **Tối ưu hóa độ trễ suy luận xuống 1.2 ms trên GPU NVIDIA GeForce RTX 5090** (~830 FPS), chỉ chiếm 6% chu kỳ 20ms của bus SocketCAN, bảo đảm điều khiển vòng lặp kín thời gian thực không độ trễ (Zero Control Jitter).
* **Lập trình module Temporal Resampler 50Hz** (Linear Interpolation + Nearest Neighbor) xử lý triệt để hiện tượng rớt khung hình và dao động tần số trong dữ liệu teleoperation HDF5.
* **Triển khai bộ lọc Temporal Ensembling và cấu trúc 5 tầng bảo vệ động học an toàn**, giảm 45% gia tốc giật (Jerk) trên 16 động cơ Damiao CAN-FD, bảo vệ hộp số cơ khí khi vận hành cường độ cao.
* **Đóng gói mô hình sang ONNX Runtime** và xây dựng trực tiếp node ROS 2 `openarm_act`, làm chủ toàn bộ chuỗi giá trị từ thu thập dữ liệu, huấn luyện mô hình đến triển khai trên phần cứng robot thực tế.

### Tiếng Anh (English Resume Bullets):
* **Architected and trained an end-to-end Bimanual ACT Policy** (7-layer Transformer Decoder + CVAE) featuring a lightweight **4-channel RGB-D YOLOv8-Nano backbone**, achieving **92.0% physical task success** on 16-DOF OpenArm manipulation (outperforming Stanford ALOHA ResNet-18 baseline at 81.5%).
* **Achieved ultra-fast 1.2 ms inference latency on NVIDIA GeForce RTX 5090** (~830 FPS) via ONNX Runtime optimization, consuming only 6% of the 20ms CAN bus period to ensure zero-jitter 50Hz real-time closed-loop control.
* **Engineered a 50Hz Temporal Resampler** using linear trajectory interpolation and nearest-neighbor visual alignment to resolve multi-camera frame drops and timestamp jitter across HDF5 teleoperation datasets.
* **Implemented an exponential Temporal Ensembling filter alongside a 5-layer hardware safety suite** (S-curve warm-up, kinematic velocity/acceleration clamping, and force limits), reducing motor jerk acceleration by 45% on 16 Damiao CAN-FD actuators.
* **Bridged deep learning to physical robotics** by creating a modular ROS 2 deployment node (`openarm_act`) communicating directly via SocketCAN, delivering a robust, production-grade Embodied AI monorepo.

---

## 4. KỊCH BẢN TRẢ LỜI PHỎNG VẤN KỸ THUẬT CHUYÊN SÂU (TECHNICAL INTERVIEW PREPARATION)

### Q1: "Tại sao bạn quyết định đổi Backbone ResNet-18 của Stanford ALOHA sang YOLOv8-Nano?"
> **Câu trả lời xuất sắc:**  
> *"ResNet-18 được thiết kế ban đầu cho bài toán Image Classification, nơi mạng hướng tới việc tóm tắt ngữ nghĩa toàn cục và loại bỏ thông tin không gian qua các tầng pooling sâu. Tuy nhiên, trong bài toán Pick-and-Place của robot hai cánh tay, việc xác định chính xác tâm vật thể và hướng tiếp cận của ngón kẹp với độ lệch dưới vài milimét là yếu tố quyết định.  
> YOLOv8 được tối ưu hóa từ gốc cho Object Detection và Localization. Tôi đã khai thác khối SPPF (Spatial Pyramid Pooling - Fast) để bắt các đặc trưng đa tỉ lệ, giúp mô hình bám sát vật thể ở cả cự ly quan sát xa lẫn cự ly cực gần khi ngón kẹp hạ xuống. Hơn nữa, với kích thước chỉ 2.1M tham số (nhẹ hơn 5.3 lần so với ResNet-18), YOLOv8-Nano hoạt động như một bộ điều hòa chống overfit xuất sắc trên tập dữ liệu ít mẫu (~50 demo), giúp tỷ lệ gắp đặt thành công tăng thêm 10.5%."*

### Q2: "Tại sao không sử dụng Diffusion Policy đang rất phổ biến hiện nay?"
> **Câu trả lời xuất sắc:**  
> *"Diffusion Policy rất xuất sắc trong việc biểu diễn các phân phối hành động phức tạp, nhưng đối với bài toán điều khiển thời gian thực vòng lặp kín 50Hz (chu kỳ 20ms) trên robot thật, Diffusion Policy bộc lộ điểm nghẽn nghiêm trọng về độ trễ: Quá trình lấy mẫu lặp qua 8 - 16 bước khử nhiễu (Denoising steps) tiêu tốn 30ms đến 80ms trên GPU máy tính, vi phạm ràng buộc thời gian thực của bus CAN và buộc người ta phải dùng các kỹ thuật bất đồng bộ phức tạp.  
> Trong khi đó, ACT sử dụng Transformer Decoder sinh toàn bộ 50 bước tương lai chỉ trong một lần forward pass duy nhất (1.2ms trên RTX 5090). Ngoài ra, ACT có hiệu suất mẫu rất cao, chỉ cần 50 demo là hội tụ ổn định, trong khi Diffusion Policy đòi hỏi nhiều dữ liệu hơn và thời gian huấn luyện dài gấp 4-5 lần."*

### Q3: "Làm thế nào để đảm bảo hệ thống không bị gián đoạn hoặc giật nảy cơ học khi chuyển giao giữa các Action Chunk?"
> **Câu trả lời xuất sắc:**  
> *"Mỗi chu kỳ điều khiển 20ms, mô hình ACT sinh ra một chunk 50 bước tương lai. Các chunk này gối đầu lên nhau. Tôi áp dụng thuật toán Temporal Ensembling hòa trộn các dự đoán với trọng số suy giảm theo hàm mũ: $w_i = \exp(-m \cdot i)$ với $m = 0.01$.  
> Dự đoán mới nhất nhận trọng số lớn nhất, trong khi các dự đoán từ các bước trước đóng vai trò bộ giảm chấn. Thuật toán này giúp triệt tiêu hiện tượng giật cục tại biên của chunk, giảm tới 45% gia tốc giật (Jerk) đo được trên encoder động cơ Damiao, bảo vệ bộ truyền động bánh răng và giúp hai cánh tay chuyển động tự nhiên như con người."*
