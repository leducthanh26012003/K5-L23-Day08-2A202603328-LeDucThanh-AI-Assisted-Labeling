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
| accepted | 396 | 90.4% | 438 box máy |
| edited | 5 | 1.1% | 438 box máy |
| deleted | 30 | 6.8% | 438 box máy |
| added | 22 | 5.2% | 423 đối tượng sau duyệt |

Nguồn: `qc/review_stats.md` (24 ảnh batch; đã loại 21 đối tượng quanh 12 bẫy vì lỗi nhân tạo không phản ánh máy). Trong deleted + added có 7 trường hợp là **đổi lớp** (box đúng chỗ, sai lớp), nên số vật máy **sót thật** khoảng 22 − 7 = 15.

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? **`three_wheeler`**: trong 40 box máy chỉ giữ nguyên 15 (37.5%), xoá 21, sửa 4, phải vẽ thêm 5; các lớp khác giữ nguyên trên 90% (person 188/190, motorbike 132/134, car 49/51). Box thừa dồn ở img_003 (máy 8 box, giữ 4), img_013 (4 → 1), img_014 (3 → 0), img_018 (2 → 0). Nguyên nhân: ngưỡng `three_wheeler=0.15` chọn trên 4 ảnh gold quá thấp cho cả lô; ở 0.15 YOLOE bắn cả vào vật không phải xe ba bánh (đã thấy ở pilot: pickup img_002 0.37, và trên gold: hàng rào img_012 0.45). Ngưỡng in-sample đẹp trên gold (precision 78%) nhưng ra lô thì hơn nửa box là thừa.
- So với gold: precision / recall của **máy** 61.7% (29/47 box máy, KTC 95%: 47–74%) / 80.6% (29/36, KTC 95%: 65–90%) (mẫu số: 36 đối tượng gold, không tính 7 `ignore_region`). Số này lạc quan vì ngưỡng chọn trên chính 4 ảnh này. Ngược lại, một phần FP là do gold của tôi sót (người ngồi trong tuk-tuk img_012, img_020; xem mục 1), nên precision thật có thể cao hơn một chút.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **4/12 = 33.3%** (KTC 95%: 14–61%; ngưỡng đạt 80%); loại nào lọt: mỗi loại lọt 2/3 — **missing** img_018 (người 32×48 px), img_021 (xe máy 41×85 px); **loose** img_029, img_006 (xe máy khoảng 50×65 px, box lệch khoảng 10 px); **wrong_class** img_027 (người 7×21 px bị đổi thành `bicycle`), img_024 (`car` bị đổi thành `truck`); **extra** img_017 (box `motorbike` trên mái hiên, không có xe), img_013 (box `person` trên nóc tuk-tuk, chỗ trống). Bẫy bắt được phần lớn là vật to hoặc lệch lớn: img_009 xe đạp bị xoá (212×155 px), img_005 box `truck` thừa (94×188 px), img_023 tuk-tuk box lệch mạnh; thêm img_003 người bị gọi `bicycle`. Vì sao: tôi duyệt 24 ảnh trong 20 phút (≈ 50 giây/ảnh, checklist gợi ý ~3 phút/ảnh), nên bỏ **bước 0** (ẩn nhãn máy, tự đếm trước) và không đi qua **từng box** ở bước 2 (thừa) và bước 4 (lỏng). Hai box thừa nằm trên nền trống (mái hiên, nóc xe) chỉ cần nhìn từng box là thấy. 4/8 bẫy lọt nằm ở ảnh dày xe máy (018, 021, 024, 029) — đúng nhóm ảnh guide dặn phải quét theo góc phần tư.
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Accepted 90.4% đi cùng bắt bẫy 33% nghĩa là phần lớn "accepted" là **bấm chấp nhận chứ chưa kiểm**: nếu tôi thật sự kiểm từng box thì phải thấy 2 box thừa trên nền trống. Đây là thiên kiến tự động hóa: box máy trông hợp lý nên mắt không quay lại ảnh. Vì vậy 90.4% accepted không chứng minh máy tốt; con số máy đáng tin hơn là recall 80.6% trên gold (máy sót khoảng 1/5 vật) và lỗi `three_wheeler` ở mục 2.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay **3.75 phút/ảnh** (15 phút / 4 ảnh gold) vs duyệt **0.83 phút/ảnh** (20 phút / 24 ảnh). Thiên lệch (bạn gán gold trước nên đã quen cảnh): duyệt làm sau gán tay nên đã quen CVAT và label contract, phần nhanh hơn không hoàn toàn do máy; thêm nữa tốc độ duyệt này đi kèm bắt bẫy chỉ 33%, nên 0.83 phút/ảnh là tốc độ của một lần duyệt **chưa đủ kỹ**, không phải tốc độ chuẩn. Ngay cả nếu duyệt đúng checklist (~3 phút/ảnh) thì vẫn nhanh hơn gán tay một chút, và ảnh batch dày hơn gold (trung bình 18.0 đối tượng/ảnh, 433/24 không tính `ignore_region`, so với 9 ở gold, 36/4).
- Kết luận: **có, nhưng chỉ cho person / motorbike / car / bicycle**; **không** dùng `three_wheeler` ở ngưỡng 0.15. Dựa trên: các lớp COCO được giữ nguyên trên 90% (person 188/190, motorbike 132/134, car 49/51) và recall máy trên gold 80.6% (29/36); còn `three_wheeler` bị xoá 21/40 box. Với `three_wheeler` nên nâng ngưỡng (gold cho thấy 0.3 vẫn giữ precision 83%, recall 71%) hoặc vẽ tay. Điều kiện: phải duyệt đúng checklist, vì ở tốc độ hiện tại tôi để lọt 8/12 lỗi của máy.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Bước 0 trên 24/24 ảnh:** ẩn nhãn máy, nhẩm đếm vật theo lớp và ghi số đếm vào một bảng trước khi bật nhãn; cuối buổi so số đếm với số box sau duyệt, ảnh nào lệch > 2 vật thì duyệt lại. Kiểm: bảng có đủ 24 dòng.
  2. **Đi hết danh sách object của CVAT cho từng ảnh**, mở to mọi box cao < 60 px và mọi box ở vùng nền (trời, mái, tường); với 6 ảnh dày (005, 018, 021, 024, 028, 029) quét theo 4 góc phần tư. Kiểm: số box đã kiểm = số box máy của ảnh; mục tiêu lần sau bắt bẫy `extra` và `loose` 3/3.
  3. **Không duyệt dưới 2 phút/ảnh** (24 ảnh ≥ 48 phút) và nghỉ 3–5 phút sau mỗi 8 ảnh; nếu `time_log` cho thấy nhanh hơn thì duyệt lại lô. Kiểm: `batch_review` trong `time_log.csv` ≥ 48 phút và bắt bẫy ≥ 80% (≥ 10/12).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_010 (xích lô phía trước, người đạp ngồi sau) | SAM 2.1 base (`sam2.1_b.pt`) | 4 (3 dương: mui, bánh sau, bánh trái; 1 âm: lưng người đạp) | Không gồm người đạp (đúng 4.2 mục 4: người ở phía trước khung xe) | Lấm tấm vài đốm trên sọc trắng áo người đạp và trên cột mui. Thử thêm click âm thứ 5 lên sọc trắng áo thì mask **tệ hơn**: lỗ chỗ và lan sang xích lô phía sau, nên giữ bản 4 click | `cyclo_img010.jpg` |
| img_023 (tuk-tuk giữa, có khách ngồi trong, người đứng chạm xe) | SAM 2.1 base | 4 (2 dương: mui, bánh trước; 2 âm: người đứng, khách ngồi) | Không gồm khách ngồi trong xe và người đứng cạnh | Lần 1 (3 click) mask nuốt cả khách ngồi bên trong; thêm click âm lên khách thì tách được. Mui, thân, hai bánh bám biên tốt | `tuktuk_img023.jpg` |
| img_009 (xe máy đỗ chở hàng: bao, túi, chuối) | SAM 2.1 base | 7 (4 dương: bánh trước, bánh sau, tay lái, giỏ; 3 âm: đống chuối, bao trắng, túi xanh) | Không gồm hàng (bao, túi, chuối); có giỏ xe vì là bộ phận gắn liền | Lần 1 (3 click) chỉ ra bánh trước và **một túi xanh** (click dương rơi trúng túi). Hàng che gần hết thân xe nên phải chấm từng phần nhìn thấy; mask cuối có 2 polygon (thân xe bị túi xanh cắt đôi) | `moto_cargo_img009.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? **Giúp:** mục 4 ("hai vật chạm nhau: vùng chạm thuộc vật ở phía trước") quyết định rõ cho img_010 và img_023: người đạp, khách ngồi, người đứng đều ở phía trước khung xe nên bỏ khỏi mask xe. Mục 2 (lỗ thủng nhỏ hơn ~5% thì không cần vẽ) cho phép bỏ qua các khoảng hở nhỏ giữa nan bánh và khung xích lô. **Thiếu:** 4.2 không nói **hàng hoá chở trên xe máy** có thuộc mask không (bảng lớp chỉ nói `three_wheeler` gồm "thùng hàng gắn liền xe"); với img_009 tôi chọn bỏ hàng, giữ giỏ, nhưng đây là câu hỏi cần chốt ở v1.1. Cũng chưa có quy tắc khi hàng che xe thành nhiều mảnh: một mask nhiều polygon hay nối lại (img_009 ra 2 polygon).
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: chưa thử (CVAT trên máy chưa cài serverless/Nuclio cho AI Tools).

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
