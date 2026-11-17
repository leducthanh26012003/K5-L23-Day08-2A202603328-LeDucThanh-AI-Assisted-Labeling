# Báo cáo Day 08 — 2A202603328 - Lê Đức Thanh

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m (`yolo26m.pt`), CPU | `person=0.5`; các lớp còn lại `--conf 0.3` | `threshold_sweep`: person giữ recall 90% (9/10) ở mọi ngưỡng 0.10–0.50 nên lấy 0.5 cho precision cao nhất (56% = 9/16). Motorbike không ngưỡng nào đạt recall 90% (tối đa 67% = 6/9 ở 0.10; 56% = 5/9 ở 0.3), car tối đa 88% (7/8) → để mặc định 0.3. Bicycle (1 vật), truck (1), bus (0) quá ít mẫu để chọn riêng. Rủi ro: r02 cho thấy lên 0.5 mất nhiều người ở xa trên pilot (img_007 mất 9 người) → khi duyệt phải quét kỹ người nhỏ bị sót |
| three_wheeler | YOLOE-26s (`yoloe-26s-seg.pt`), prompt: `tuk-tuk`, `cyclo`, `cargo tricycle` | `three_wheeler=0.15` | `threshold_sweep`: recall 100% (7/7), precision 78% (7/9) ở 0.15; lên 0.25 recall còn 71% (5/7). r03: xe thật có điểm thấp (xích lô phải img_032 chỉ 0.27). r01: YOLO26 không có lớp này (tuk-tuk / xích lô thành car, bicycle, motorbike) nên file YOLOE đứng **trước** trong lệnh gộp để `three_wheeler` thắng box car / truck chồng lên (gộp đã bỏ 11 car, 6 truck, 2 bus, 1 bicycle của YOLO26) |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.5,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json` → 510 box trên 28 ảnh (24 batch + 4 gold): person 215, motorbike 154, car 65, three_wheeler 52, bicycle 13, truck 7, bus 4.
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: **SAM 2.1 base** (`sam2.1_b.pt`). r04: khi box đúng thì mask người rất sát (img_004 người đàn ông 0.93, img_013 tài xế 0.91). Chạy được trên CPU (5.12 s/ảnh), đủ cho 3 vật ở Part 5. SAM 3 cần quyền HuggingFace và GPU nên không dùng; SAM 1 (`mobile_sam.pt`) nhẹ hơn nhưng chưa so chất lượng mask.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? **Lỗi lớp.** YOLO26 gọi tuk-tuk là `car` (img_004 0.37 / 0.34, img_013 0.42), xích lô là `bicycle` (img_032 0.70) hoặc `motorbike` (img_007 0.28–0.38), SUV đen là `truck` 0.88 (img_001, theo contract SUV → `car`). Đổi ngưỡng chỉ giữ hay bỏ cả box, không đổi được lớp (r02 ở 0.5 vẫn còn các lỗi này). Sửa: dùng YOLOE cho `three_wheeler` và cho YOLOE ưu tiên khi gộp (IoU ≥ 0.6 thì bỏ box YOLO26); lỗi còn lại (SUV → truck, pickup bị YOLOE gọi `three_wheeler` ở img_002 0.37) phải sửa tay khi duyệt.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): **Cao mà sai:** img_001 SUV đen `truck` 0.88; img_032 xích lô nhìn từ phía sau `bicycle` 0.70; img_011 xe xanh cắt mép trái `truck` 0.77; img_012 hàng rào chắn `three_wheeler` 0.45. **Thấp mà đúng:** img_032 xích lô phải `three_wheeler` 0.27; img_030 xe tay ga trắng phía trước 0.47; img_007 người lái phía sau 0.35–0.49; img_002 người đi bộ bên trái 0.27–0.50.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: ngưỡng được chọn và chấm trên cùng 4 ảnh gold (43 đối tượng: 10 person, 9 motorbike, 8 car, 7 three_wheeler, 1 bicycle, 1 truck, 7 ignore_region), nên precision / recall của máy trên gold là **lạc quan** và chỉ là ước lượng thô; bicycle / truck / bus có < 5 vật nên không chọn được ngưỡng riêng. Thêm nữa, khi đối chiếu máy với gold tôi thấy gold của chính tôi có chỗ lệch contract: img_036 box `motorbike` bao cả người lái (mục 4 của contract: xe và người là hai box riêng); img_012 và img_020 chưa gán người ngồi trong tuk-tuk dù nhìn rõ qua cạnh hở; img_020 xe tím góc phải gán `truck` trong khi máy gọi `bus` 0.69. Một phần "sót / thừa" của máy trên gold vì vậy là lỗi gold, sẽ đưa vào calibration (mục 6).

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<>> | <<>> | <<>> | <<>> | <<>> |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
