# Thông tin học viên

- Họ và tên: Lê Đức Thanh
- MSHV: 2A202603328
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: leducthanh26012003
- Repo bài nộp (URL): https://github.com/leducthanh26012003/K5-L23-Day08-2A202603328-LeDucThanh-AI-Assisted-Labeling
- Commit chốt bài (hash): 6724b71438b769ab8a55737968825a60776f4d3b
- Máy: Windows 11 + WSL2 Ubuntu, Python 3.10.12; GPU: không dùng (torch bản CPU, máy có NVIDIA MX330 2 GB)
- CVAT (Docker): v2.74.1

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 396 / 5 / 30 = 90.4% / 1.1% / 6.8% (mẫu số: 438 box máy); added: 22 = 5.2% (mẫu số: 423 đối tượng)
- So với gold: precision / recall của **máy** 61.7% (29/47) / 80.6% (29/36)
- Bắt bẫy: 4/12 = 33.3%
- Thời gian: gán tay 3.75 phút/ảnh (15 phút / 4 ảnh); duyệt có pre-label 0.83 phút/ảnh (20 phút / 24 ảnh)
- Lỗi QC: 40.0% theo ảnh (2/5), 5.6% theo đối tượng (5/90)
- **Quyết định:** Rework — lý do một dòng: ảnh lỗi 2/5 > 2%, lỗi hệ thống "thừa" 3/5 lỗi, bắt bẫy 4/12 < 80%
