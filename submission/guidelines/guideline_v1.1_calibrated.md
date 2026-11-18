# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603328 - Lê Đức Thanh

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 2 | Khối xe máy đỗ dày đặc **gần camera**: từng xe hay `ignore_region`? | Xe nào thấy được **bánh hoặc yên cùng tay lái** và cao ≥ 20 px thì vẽ box `motorbike` riêng. Phần xe chồng phía sau không tách được biên, từ 5 xe trở lên, thì vẽ **một** `ignore_region`, **không** phủ lên xe đã có box. Bổ sung mục 4.1.6 (v1.0 chỉ nói vật ở xa). | img_006 (hàng xe đỗ bên trái hẻm) | CAL-07 |
| 3 | Hàng nhô ra / hàng chở trên xe: box và mask xe có gồm hàng không? | Box và mask `motorbike`, `bicycle` **không** gồm hàng chất lên hay treo vào xe (bao, túi, chuối, thúng). **Gồm** bộ phận gắn cố định: giỏ, baga, chân chống. Hàng che xe thành nhiều mảnh thì mask giữ **nhiều polygon**, không nối qua hàng. `three_wheeler` vẫn gồm thùng hàng gắn liền xe như v1.0 mục 3. | img_009 (xe máy đỗ chở chuối và bao) | CAL-06 |
| 4 | Người ngồi trong xe ba bánh có mui: ngưỡng "thấy ≥ 20%" có hợp lý? | Thay ước lượng 20% (khó đo bằng mắt, tôi đã làm không nhất quán) bằng tiêu chí nhìn được: gán `person` khi thấy được **đầu và vai / thân trên** qua cạnh hở. Chỉ thấy tay, chân, hoặc bóng người sau kính, sau mui thì **không** gán. Phân vân thì tick `can_review`. Sửa mục 4.1.2 và dòng "tuk-tuk có hành khách" ở mục 5. | img_023 (khách ngồi trong tuk-tuk giữa: gán), img_020 (khách thấy rõ qua cạnh hở: gán) | CAL-05 |
| 7 | Có đặt `conf` tối thiểu để chấp nhận box máy không cần sửa? | **Không đặt.** Box điểm cao vẫn có thể sai (img_001: `truck` 0.88 cho một SUV) và hai box thừa bị lọt có `conf` 0.67 và 0.60. `conf` chỉ dùng để **ưu tiên nhìn kỹ** box điểm thấp, không bao giờ để bỏ qua box điểm cao. | img_001, img_017, img_013 | CAL-08 |

**Chưa chốt** (lô 24 ảnh và 4 ảnh gold không có ca đủ rõ để làm bằng chứng, nên giữ nguyên như v1.0 và chờ lô sau):
- Câu 1 (xe đạp điện, xe máy điện nhỏ): chưa gặp ca nào.
- Câu 5 (kính chắn gió có thuộc mask xe không): 3 mask ở Part 5 đều là xe hở (xích lô, tuk-tuk, xe máy), chưa gặp ca kính.
- Câu 6 (box máy chồng nhau quá nhiều trong ảnh đêm): chưa gặp ca nào.

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_001, img_002 | SUV đen (img_001), bán tải trắng (img_002) | YOLO26m gọi `truck` 0.88 và 0.73 | Đổi thành `car`; chỉ giữ `truck` khi có buồng lái tách thùng hàng và từ 4 bánh (CAL-01) | Không đổi quy tắc (mục 3 đã ghi SUV, bán tải là `car`); thêm img_001 làm ví dụ và nhắc người duyệt kiểm **mọi** box `truck` của YOLO26 |
| img_003, img_013, img_014, img_018 | Ảnh không có hoặc có ít xe ba bánh | YOLOE-26s ở ngưỡng 0.15 vẽ nhiều box `three_wheeler` thừa: 21/40 box bị xoá khi duyệt | Lô sau dùng ngưỡng `three_wheeler=0.3` và duyệt từng box `three_wheeler` (CAL-02) | Không phải quy tắc gán nhãn mà là recipe của máy; ghi ở đây để lô sau không lặp lại |
| img_003 | Một xe ba gác chở rau có mui | YOLOE-26s vẽ 3 box cho cùng một xe: một box chỉ bao mui, một box chỉ bao thùng, một box bao cả xe | Mỗi xe ba bánh **một** box bao cả mui và thùng hàng gắn liền xe; xoá box chỉ bao một phần (CAL-03) | Mục 3 (`three_wheeler`): thêm câu "một xe = một box, kể cả khi mui và thùng tách màu" |
| img_018, img_027, img_033, img_034 | Xe máy, xe đạp, ô tô ở xa cao 14–19 px | YOLO26m vẽ box (conf 0.31–0.82), tôi giữ nguyên khi duyệt | Xoá; nếu là cụm từ 5 vật nhỏ sát nhau thì thay bằng một `ignore_region` (CAL-04) | Không đổi quy tắc (mục 4.1.6 đã rõ); thêm img_018 làm ví dụ |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu 2 (khối xe đỗ gần camera), câu 3 (hàng chở trên xe), câu 4 (người trong tuk-tuk: tiêu chí đầu và vai thay cho 20%), câu 7 (không đặt `conf` tối thiểu). Thêm ca khó: SUV / bán tải bị gọi `truck`, nhiều box cho một xe ba bánh, box dưới 20 px, ngưỡng `three_wheeler` của máy. Câu 1, 5, 6 chưa chốt vì chưa có ca. | CAL-01 đến CAL-08 (CAL-09 là lỗi duyệt, không đổi guideline) |
