# 🇻🇳 AIDEOM-VN — AI-Driven Decision Optimization Model for Vietnam

Web app Streamlit giải **12 bài toán mô hình ra quyết định** phát triển kinh tế Việt Nam trong kỉ nguyên AI, dùng dữ liệu thực 2020–2025 (NSO, MoST, MIC, MPI, World Bank, GII).

**Tác giả: Đỗ Tiến Đạt**

## ✨ Tính năng

- **Trang chủ**: chỉ số vĩ mô VN 2025 + bản đồ 12 bài theo 4 cấp độ.
- **Bài 1–11**: mỗi bài là một trang tương tác (slider/toggle), hiển thị kết quả + biểu đồ Plotly **và phần trả lời câu hỏi thảo luận chính sách**.
- **Bài 12**: dashboard tích hợp 4 tab (Tổng quan M1-M2 · Phân bổ M3 · 5 Kịch bản M6 · Cảnh báo rủi ro M4-M5).
- Tất cả phép tối ưu (LP, MIP, TOPSIS, NSGA-II, động học, SP, Q-learning) chạy **trực tiếp** trên dữ liệu thực.

## 📂 Cấu trúc

```
aideom_vn/
├── app.py                 # Toàn bộ ứng dụng
├── requirements.txt
├── README.md
└── data/
    ├── vietnam_macro_2020_2025.csv
    ├── vietnam_sectors_2024.csv
    └── vietnam_regions_2024.csv
```

## 🚀 Chạy

```bash
cd aideom_vn
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`.

## ⚙️ Ghi chú kỹ thuật

- **Bài 7 (NSGA-II)**: dùng `pymoo` nếu có; nếu chưa cài, app **tự fallback** sang random-sampling + lọc Pareto (không lỗi).
- **Bài 10 (Stochastic SP)**: dùng `pyomo + glpk` nếu có; nếu không, **tự fallback** sang `scipy.linprog` cho cùng mô hình tuyến tính.
- Đổi tham số bằng slider/toggle để xem độ nhạy theo thời gian thực.
- Các số liệu chính khớp notebook gốc (ví dụ MAPE Bài 1 = 6,42%).

## 🌐 Deploy (Streamlit Cloud)

1. Đẩy thư mục lên GitHub (giữ nguyên `data/`).
2. streamlit.io → New app → trỏ tới `app.py`.

---
© 2025 AIDEOM-VN · Đỗ Tiến Đạt
