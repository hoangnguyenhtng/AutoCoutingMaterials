# Product Retrieval v0.1

Prototype nhận dạng và xác minh **mã sản phẩm từ một ảnh crop**. Phiên bản này dùng
DINOv2 để trích xuất đặc trưng, tìm code gần nhất trong gallery và trả về một trong
năm trạng thái:

- `MATCH`: ảnh phù hợp code công nhân đã chọn.
- `MISMATCH`: một code khác vượt trội đủ rõ.
- `REVIEW`: kết quả chưa đủ cách biệt, cần con người xác nhận.
- `UNKNOWN`: ảnh không đủ giống bất kỳ code đã đăng ký nào.
- `NOT_REGISTERED`: code công nhân chọn chưa có trong gallery.

Phạm vi v0.1 chưa gồm phát hiện/đếm sản phẩm trên tray, YOLO, Excel, GUI và PLC.
Mục tiêu là kiểm chứng retrieval trước khi mở rộng hệ thống.

## Cài đặt

Yêu cầu Python 3.10 trở lên. Nên dùng môi trường ảo:

```powershell
git clone https://github.com/hoangnguyenhtng/AutoCoutingMaterials.git
cd AutoCoutingMaterials
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Cách trên không cần chạy `Activate.ps1`, nên cũng dùng được khi PowerShell chặn
script bởi execution policy. Sau khi cài, thay `python` trong các lệnh bên dưới
bằng `.venv\Scripts\python.exe`, hoặc kích hoạt môi trường bằng
`.venv\Scripts\Activate.ps1` nếu máy cho phép.

Lần encode đầu tiên, `torch.hub` sẽ tải trọng số DINOv2 từ repository chính thức
của Meta. Những lần sau PyTorch dùng bản cache. Nếu máy chạy hoàn toàn offline,
cần chuẩn bị cache/model trước.

## Chuẩn bị dữ liệu

Đặt ảnh crop vào `data/crops/<CODE>/` và điền
`data/manifests/local_data.csv`. Manifest có các cột:

```csv
image_path,product_code,model_family,physical_part_id,capture_session,split,orientation,source
data/crops/A001/001.jpg,A001,M01,P001,S01,gallery,0,local
data/crops/A001/002.jpg,A001,M01,P002,S02,query,90,local
data/crops/A002/001.jpg,A002,M01,P003,S01,gallery,0,local
data/crops/UNKNOWN/001.jpg,U001,M99,P900,S03,unknown,45,local
```

Quy tắc quan trọng:

- `gallery`: ảnh tham chiếu đã đăng ký.
- `query`: ảnh độc lập dùng để đánh giá code đã biết.
- `unknown`: toàn bộ ảnh của code bị giữ ngoài gallery để đo khả năng từ chối.
- Không đặt ảnh gốc ở gallery và augmentation của chính ảnh đó ở query.
- Nên tách gallery/query theo cả `physical_part_id` và `capture_session`.
- Ảnh nên được crop sát sản phẩm nhưng vẫn chừa một biên nhỏ nhất quán.

Kiểm tra manifest:

```powershell
python -m product_retrieval --config configs/experiment.yaml validate-manifest
```

Lệnh trả exit code `2` nếu phát hiện nguy cơ rò rỉ mẫu vật/phiên chụp.

## Chạy thử nghiệm

Tạo gallery từ các dòng có `split=gallery`:

```powershell
python -m product_retrieval --config configs/experiment.yaml build-gallery
```

Tìm năm code gần nhất cho một ảnh:

```powershell
python -m product_retrieval --config configs/experiment.yaml search data/crops/A001/query.jpg --top-k 5
```

Kiểm tra code do công nhân chọn:

```powershell
python -m product_retrieval --config configs/experiment.yaml verify data/crops/A001/query.jpg --selected-code A001
```

Đánh giá toàn bộ các dòng `query` và `unknown`:

```powershell
python -m product_retrieval --config configs/experiment.yaml evaluate
```

Kết quả được ghi vào:

- `outputs/metrics.json`: Recall@1, Recall@5, MRR, FAR, FRR, review rate,
  unknown rejection rate và thời gian xử lý.
- `outputs/predictions.csv`: dự đoán của từng ảnh.
- `outputs/difficult_pairs.csv`: các cặp code thường bị nhầm.

## Cấu hình thử nghiệm

Tất cả tham số nằm trong `configs/experiment.yaml`. Nên tạo một file cấu hình cho
mỗi thử nghiệm và chỉ đổi một nhóm biến tại một thời điểm:

1. `feature_type`: so sánh `cls` và `patch_mean`.
2. `image_size`: so sánh 224, 336 và 448 (phải chia hết cho patch size 14).
3. `aggregation`: `top_mean`, `max` hoặc `centroid`.
4. `references_per_code`: số ảnh tốt nhất dùng cho `top_mean`.
5. Các ngưỡng verification.

Các ngưỡng có sẵn chỉ giúp pipeline chạy được, **không phải ngưỡng sản xuất**.
Cần chọn chúng trên validation độc lập, khóa cấu hình, rồi đo lại trên test chưa
từng dùng để chỉnh tham số. Với hệ thống giao hàng, ưu tiên kiểm soát FAR; đồng
thời theo dõi review rate để không làm giảm năng suất vận hành.

Khi đổi model, loại feature, kích thước ảnh hoặc tiền xử lý, chương trình sẽ từ
chối dùng gallery cũ. Hãy chạy lại `build-gallery` để tránh so sánh vector không
tương thích.

## Kiểm thử

```powershell
python -m pytest -q
```

Test dùng model giả cục bộ nên không tải trọng số DINOv2 và không cần GPU.

## Hướng phát triển sau v0.1

Sau khi retrieval đạt mục tiêu trên 10–20 code thật (bao gồm các cặp rất giống
nhau), bước tiếp theo là detector/segmenter cho tray. Mỗi crop từ detector sẽ tái
sử dụng nguyên pipeline ở đây. Phần quản lý lượt tray, tổng số lượng theo code,
đọc kế hoạch Excel và giao tiếp PLC nên tiếp tục là các module độc lập.
