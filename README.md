# Object Segmentation với Ultralytics YOLO

Repo fine-tune YOLO instance segmentation trên bộ Oxford-IIIT Pet. Dataset gốc nằm trong `The Oxford-IIIT Pet.zip`; archive gốc chỉ chứa ảnh. Bộ 100 ảnh để gán nhãn đã được xuất từ Roboflow thành `coco-segmentation.zip` và chuyển sang YOLO segmentation.

## Dataset đã gán nhãn

`coco-segmentation.zip` có 100 ảnh và polygon class `pet`. Script import giữ nguyên split của Roboflow: **70 train, 15 validation, 15 test**. Tọa độ COCO được chuẩn hóa theo kích thước từng ảnh và lưu ở định dạng polygon YOLO. Ảnh và nhãn nằm trong `data/processed/`; cấu hình là `configs/data.yaml`. Cả mèo lẫn chó dùng cùng class `pet`.

Để import lại archive (script từ chối ghi đè dataset đã có file):

```powershell
python scripts/import_roboflow_coco.py
```

## Cài đặt và huấn luyện

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --size n --epochs 100
python train.py --size s --epochs 100
python train.py --size m --epochs 100 --batch 4
```

Script dùng checkpoint segmentation pretrained `yolo11n-seg.pt`, `yolo11s-seg.pt`, `yolo11m-seg.pt`. Mặc định ảnh đầu vào train ở `imgsz=640`; giữ cùng kích thước, split và epochs khi so sánh n/s/m. Nếu thiếu VRAM, giảm `--batch`, hoặc thêm `--device cpu` / `--device 0` tùy máy. Ultralytics lưu weights, biểu đồ loss, mAP và ảnh dự đoán trong `runs/segment/yolo11{n,s,m}/`.

## Huấn luyện và đánh giá YOLO11n/s/m trên Kaggle GPU

Notebook Kaggle và gói dữ liệu đã chuẩn bị trong [`kaggle/`](kaggle/). Upload `kaggle/kaggle_yolo_seg_dataset.zip` thành Kaggle Dataset, attach dataset vào notebook, bật GPU và Internet, rồi chạy `kaggle/train_yolo11_n_s_m_on_kaggle.ipynb`. Notebook train cả ba model với cùng cấu hình, đánh giá trên test rồi lưu `summary.csv`, biểu đồ, đồ thị huấn luyện và weights vào `yolo11_nsm_results.zip` để tải xuống. Xem [`kaggle/README.md`](kaggle/README.md) để biết các bước cụ thể.

## Đánh giá

Trong lúc train, Ultralytics đánh giá trên split validation. Đánh giá cuối cùng trên test bằng checkpoint tốt nhất:

```powershell
python val.py runs/segment/yolo11n/weights/best.pt
```

Mặc định `val.py` dùng split `test`; thêm `--split val` để xem lại validation. Script in box mAP50/mAP50-95 và mask mAP50/mAP50-95. Báo cáo nên so sánh các metric và loss giữa n/s/m. Với chỉ 100 ảnh, kết quả mang tính thử nghiệm và có thể dao động.

## Bố cục

- `labeling/images/`: bản 100 ảnh ban đầu để gán nhãn; `labeling/manifest.csv` ghi giống Oxford-IIIT Pet.
- `coco-segmentation.zip`: export COCO Segmentation từ Roboflow (bỏ qua trong Git).
- `scripts/import_roboflow_coco.py`: chuyển export COCO polygon sang YOLO và giữ train/valid/test.
- `data/processed/`: ảnh cùng nhãn YOLO đã chuyển (bỏ qua trong Git).
- `configs/data.yaml`: cấu hình dataset một lớp `pet`.
- `train.py`, `val.py`: huấn luyện và đánh giá.

Nguồn dataset: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). Hướng dẫn segmentation: [Ultralytics segmentation docs](https://docs.ultralytics.com/tasks/segment/).
