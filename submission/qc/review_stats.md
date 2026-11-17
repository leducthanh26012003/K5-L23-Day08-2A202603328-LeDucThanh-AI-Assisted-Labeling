# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 438; đối tượng sau duyệt: 423; ngưỡng accepted: IoU >= 0.9
- Đã **loại 21 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 396 | 396/438 = 90.4% | 438 box máy |
| edited (sửa) | 5 | 5/438 = 1.1% | 438 box máy |
| deleted (xoá) | 30 | 30/438 = 6.8% | 438 box máy |
| added (vẽ thêm) | 22 | 22/423 = 5.2% | 423 đối tượng sau duyệt |

Trong số deleted + added, **7 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 15.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 188 | 0 | 2 | 6 |
| bicycle | 5 | 0 | 2 | 1 |
| motorbike | 132 | 1 | 1 | 2 |
| car | 49 | 0 | 2 | 3 |
| bus | 2 | 0 | 1 | 2 |
| truck | 5 | 0 | 1 | 3 |
| three_wheeler | 15 | 4 | 21 | 5 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 36 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 29 | 18 | 7 | 61.7% (KTC 95%: 47–74%) | 80.6% (KTC 95%: 65–90%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 4/12 = 33.3% (KTC 95%: 14–61%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - missing: 1/3
  - loose: 1/3
  - wrong_class: 1/3
  - extra: 1/3
  - lọt: img_029.jpg (loose), img_018.jpg (missing), img_027.jpg (wrong_class), img_017.jpg (extra), img_006.jpg (loose), img_021.jpg (missing), img_024.jpg (wrong_class), img_013.jpg (extra)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
