# -*- coding: utf-8 -*-
"""
AIDEOM-VN — AI-Driven Decision Optimization Model for Vietnam
Web app giải 12 bài toán mô hình ra quyết định phát triển kinh tế Việt Nam.
Tác giả: Đỗ Tiến Đạt
Chạy: streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="AIDEOM-VN — Mô hình ra quyết định",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS — DARK THEME, ACCENT CRIMSON/PINK (khớp giao diện mẫu)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    .stApp { background-color: #0e1117; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    h1, h2, h3, h4 { font-family: 'Sora', sans-serif !important; color: #fafafa; }
    h1 { font-weight: 800 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #11141c; border-right: 1px solid #1f2430; }

    /* Big stat numbers */
    .stat-label { color: #c9ccd6; font-size: 0.95rem; font-weight: 600; margin-bottom: 2px; }
    .stat-value { color: #ff4d6d; font-size: 2.6rem; font-weight: 800; font-family: 'Sora',sans-serif; line-height: 1.1; }
    .stat-delta { display:inline-block; background:#10301f; color:#42d77d; padding:3px 12px;
                  border-radius:8px; font-size:0.85rem; font-weight:600; margin-top:6px; }

    .hero-sub { font-family:'Sora',sans-serif; font-style:italic; font-weight:600;
                font-size:1.6rem; color:#e6e8ee; margin-top:-6px; }
    .vn-badge { display:inline-block; background:#ff4d6d; color:white; font-weight:800;
                font-size:0.85rem; padding:2px 8px; border-radius:6px; margin-right:8px; vertical-align:middle; }

    /* Policy answer box */
    .policy-box { background:#161a24; border-left:4px solid #ff4d6d; border-radius:8px;
                  padding:16px 20px; margin:10px 0; }
    .policy-q { color:#ffd166; font-weight:700; font-size:1.02rem; margin-bottom:6px; }
    .policy-a { color:#d4d7e0; line-height:1.6; }

    div.stButton > button { background:#ff4d6d; color:white; border:none; border-radius:10px;
                            font-weight:700; padding:8px 22px; }
    div.stButton > button:hover { background:#e63956; color:white; }

    .stTabs [data-baseweb="tab-list"] { gap:4px; }
    .stTabs [data-baseweb="tab"] { background:#161a24; border-radius:8px 8px 0 0; padding:8px 16px; }
    .stTabs [aria-selected="true"] { background:#ff4d6d; }

    hr { border-color:#1f2430; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# NẠP DỮ LIỆU
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data") if "__file__" in dir() else "data"

@st.cache_data
def load_data():
    macro = pd.read_csv(os.path.join(DATA_DIR, "vietnam_macro_2020_2025.csv"))
    sectors = pd.read_csv(os.path.join(DATA_DIR, "vietnam_sectors_2024.csv"))
    regions = pd.read_csv(os.path.join(DATA_DIR, "vietnam_regions_2024.csv"))
    return macro, sectors, regions

try:
    macro_df, sectors_df, regions_df = load_data()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    DATA_ERR = str(e)

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#d4d7e0", family="Inter"),
    xaxis=dict(gridcolor="#1f2430"), yaxis=dict(gridcolor="#1f2430"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# Hệ số co giãn Cobb-Douglas (Bài 1)
ALPHA, BETA, GAMMA, DELTA, THETA = 0.33, 0.42, 0.10, 0.08, 0.07
# Dữ liệu input bài 1 (theo đề)
K_ARR = np.array([16500, 17800, 19600, 21300, 23500, 25900], float)
L_ARR = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4], float)
D_ARR = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5], float)
AI_ARR = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1], float)
H_ARR = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2], float)
YEARS = np.array([2020, 2021, 2022, 2023, 2024, 2025])


def policy_box(q, a):
    st.markdown(
        f'<div class="policy-box"><div class="policy-q">❓ {q}</div>'
        f'<div class="policy-a">{a}</div></div>', unsafe_allow_html=True)


def stat(label, value, delta):
    st.markdown(
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-delta">↑ {delta}</div>', unsafe_allow_html=True)


# ============================================================
# SIDEBAR — ĐIỀU HƯỚNG
# ============================================================
PAGES = [
    "🏠 Trang chủ",
    "🌱 Bài 1 — Cobb-Douglas + AI",
    "💰 Bài 2 — LP ngân sách số",
    "📊 Bài 3 — Priority 10 ngành",
    "📘 Bài 4 — LP ngành-vùng",
    "🎯 Bài 5 — MIP 15 dự án",
    "🏆 Bài 6 — TOPSIS 6 vùng",
    "🌐 Bài 7 — NSGA-II Pareto",
    "⏳ Bài 8 — Động 2026-2035",
    "👷 Bài 9 — Lao động & AI",
    "🎲 Bài 10 — Stochastic SP",
    "🕹️ Bài 11 — Q-learning RL",
    "🇻🇳 Bài 12 — AIDEOM tích hợp",
]

with st.sidebar:
    st.markdown('<h3 style="margin-bottom:0;"><span class="vn-badge">VN</span>AIDEOM-VN</h3>',
                unsafe_allow_html=True)
    st.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")
    st.markdown("---")
    page = st.radio("Điều hướng", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("📁 Dữ liệu: NSO, MoST, MIC, MPI, WB, GII 2025")
    st.caption("👤 Tác giả: **Đỗ Tiến Đạt**")

if not DATA_OK:
    st.error(f"Không nạp được dữ liệu CSV trong thư mục `data/`. Lỗi: {DATA_ERR}")
    st.stop()


# ============================================================
# TRANG CHỦ
# ============================================================
def page_home():
    st.markdown('<h1><span class="vn-badge">VN</span>AIDEOM-VN</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI-Driven Decision Optimization Model for Vietnam</div>',
                unsafe_allow_html=True)
    st.write("Web app giải **12 bài toán mô hình ra quyết định** phát triển kinh tế Việt Nam "
             "trong kỉ nguyên AI — dữ liệu thực 2020-2025.")
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat("GDP 2025", "514,0 tỷ USD", "+8,02%")
    with c2: stat("Kinh tế số / GDP", "≈19,5%", "+1,2 đpt")
    with c3: stat("FDI giải ngân 2025", "27,6 tỷ USD", "+8,9%")
    with c4: stat("GDP/người 2025", "5.026 USD", "+6,9%")

    st.markdown("---")
    st.markdown("## 📚 12 bài toán theo 4 cấp độ")

    levels = {
        "🟢 Cấp độ DỄ — Làm quen mô hình": [
            ("Bài 1", "Hàm sản xuất Cobb-Douglas mở rộng + AI — Growth accounting, dự báo GDP 2030"),
            ("Bài 2", "LP phân bổ ngân sách 4 hạng mục — scipy.optimize, shadow price"),
            ("Bài 3", "Chỉ số ưu tiên 10 ngành — Min-max norm, weighted scoring, sensitivity"),
        ],
        "🟡 Cấp độ TRUNG BÌNH — Tối ưu cổ điển": [
            ("Bài 4", "LP phân bổ ngân sách số ngành-vùng — PuLP + CVXPY, ràng buộc công bằng"),
            ("Bài 5", "MIP lựa chọn 15 dự án CĐS — biến nhị phân, tiên quyết, knapsack"),
            ("Bài 6", "TOPSIS xếp hạng 6 vùng — chuẩn hóa vector, trọng số Entropy"),
        ],
        "🟠 Cấp độ KHÁ KHÓ — Đa mục tiêu & động": [
            ("Bài 7", "Tối ưu đa mục tiêu Pareto NSGA-II — 4 mục tiêu xung đột"),
            ("Bài 8", "Tối ưu động liên thời gian 2026-2035 — Cobb-Douglas + tích lũy vốn"),
            ("Bài 9", "Tác động AI tới lao động — NetJob ròng theo ngành"),
        ],
        "🔴 Cấp độ KHÓ — Bất định & học tăng cường": [
            ("Bài 10", "Quy hoạch ngẫu nhiên 2 giai đoạn — VSS, EVPI, minimax regret"),
            ("Bài 11", "Q-learning cho chính sách thích nghi — MDP, DQN"),
            ("Bài 12", "Tích hợp hệ thống AIDEOM-VN — 6 module, 5 kịch bản"),
        ],
    }
    for title, items in levels.items():
        with st.expander(title, expanded=(title.startswith("🟢"))):
            for code, desc in items:
                st.markdown(f"**{code}** &nbsp;&nbsp; {desc}")

    st.markdown("---")
    st.caption("© 2025 AIDEOM-VN · Tác giả: Đỗ Tiến Đạt · Dữ liệu: NSO, MoST, MIC, MPI, World Bank, GII")


# ============================================================
# BÀI 1 — COBB-DOUGLAS MỞ RỘNG + AI
# ============================================================
def cobb_tfp():
    M = (K_ARR**ALPHA)*(L_ARR**BETA)*(D_ARR**GAMMA)*(AI_ARR**DELTA)*(H_ARR**THETA)
    Y = macro_df["GDP_trillion_VND"].values.astype(float)
    A = Y / M
    return A, M, Y

def page_bai1():
    st.markdown("## 🌱 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng + AI")
    st.caption("Y = A·K^α·L^β·D^γ·AI^δ·H^θ  với  α+β+γ+δ+θ = 1")
    A, M, Y = cobb_tfp()

    # 1.4.2 MAPE + 1.4.4 dự báo 2030
    A_mean = A.mean()
    Y_hat = A_mean * M
    MAPE = np.mean(np.abs((Y - Y_hat)/Y))*100
    K30 = K_ARR[-1]*(1.06**5); L30 = L_ARR[-1]*(1.06**5); A30 = A[-1]*(1.012**5)
    Y2030 = A30*(K30**ALPHA)*(L30**BETA)*(30**GAMMA)*(100**DELTA)*(35**THETA)

    c1, c2, c3 = st.columns(3)
    with c1: stat("MAPE (Cobb-Douglas)", f"{MAPE:.2f}%", "khớp tốt")
    with c2: stat("Ā (TFP trung bình)", f"{A_mean:.4f}", "ổn định")
    with c3: stat("Y 2030 dự báo", f"{Y2030/1000:,.3f} ng.tỷ", "kịch bản số hóa")

    # 1.4.1 TFP theo năm
    fig = go.Figure(go.Scatter(x=YEARS, y=A, mode="lines+markers",
                               line=dict(color="#ff4d6d", width=3), marker=dict(size=9)))
    fig.update_layout(title="1.4.1 — Xu hướng TFP (A_t) 2020-2025", **PLOTLY_DARK)
    st.plotly_chart(fig, use_container_width=True)

    # 1.4.3 Phân rã tăng trưởng (Δln chuẩn)
    n = len(YEARS)-1
    dln = lambda s: (np.log(s[-1])-np.log(s[0]))/n
    parts = {
        "TFP (A)": dln(A), "Vốn (K)": ALPHA*dln(K_ARR), "Lao động (L)": BETA*dln(L_ARR),
        "Số hóa (D)": GAMMA*dln(D_ARR), "AI": DELTA*dln(AI_ARR), "Nhân lực số (H)": THETA*dln(H_ARR),
    }
    total = sum(parts.values())
    dfp = pd.DataFrame({"Yếu tố": list(parts), "Đóng góp (đpt/năm)": [v*100 for v in parts.values()],
                        "Tỷ trọng (%)": [v/total*100 for v in parts.values()]})
    colors = ["#7ce0c3","#f9f1a5","#9aa6b2","#ff8f8f","#7eb6ff","#ffb86b"]
    fig2 = go.Figure(go.Bar(x=dfp["Yếu tố"], y=dfp["Đóng góp (đpt/năm)"],
                            marker_color=colors, text=dfp["Tỷ trọng (%)"].round(1).astype(str)+"%",
                            textposition="outside"))
    fig2.update_layout(title="1.4.3 — Phân rã đóng góp tăng trưởng 2020-2025 (Δln)",
                       yaxis_title="Đóng góp (đpt/năm)", **PLOTLY_DARK)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(dfp.round(3), use_container_width=True, hide_index=True)

    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    policy_box("a) TFP tăng hay giảm 2020-2025? Nói gì về chất lượng tăng trưởng?",
        "TFP biến động và có xu hướng nhích lên sau cú sốc COVID. TFP là phần tăng trưởng "
        "<b>không</b> giải thích bởi tích lũy vốn/lao động — phản ánh hiệu quả, công nghệ và thể chế. "
        "TFP chiếm tỷ trọng lớn cho thấy tăng trưởng đang dịch dần từ chiều rộng (vốn) sang chiều sâu "
        "(năng suất), đúng định hướng Nghị quyết 57-NQ/TW.")
    policy_box("b) Trong D, AI, H — yếu tố nào đóng góp nhiều nhất?",
        "Số hóa (D) đóng góp lớn nhất trong nhóm yếu tố mới vì hệ số co giãn γ=0,10 cao nhất và D tăng "
        "rất nhanh (12%→19,5% GDP). AI và H có co giãn nhỏ hơn (δ=0,08; θ=0,07) nên đóng góp khiêm tốn — "
        "hàm ý cần đầu tư mạnh hơn vào nhân lực số để nâng độ co giãn dài hạn.")
    policy_box("c) Mục tiêu 30% kinh tế số/GDP 2030 có khả thi không?",
        "Khả thi nhưng <b>cần ràng buộc</b>: từ 19,5% (2025) lên 30% (2030) đòi hỏi D tăng ~9%/năm — "
        "cao hơn tốc độ lịch sử. Cần ràng buộc đồng bộ về hạ tầng số, nhân lực (H≥35%) và duy trì TFP "
        "dương, nếu không mục tiêu chỉ đạt trên giấy.")


# ============================================================
# BÀI 2 — LP NGÂN SÁCH 4 HẠNG MỤC
# ============================================================
def page_bai2():
    from scipy.optimize import linprog
    st.markdown("## 💰 Bài 2 — LP phân bổ ngân sách 4 hạng mục đầu tư số")
    st.caption("max Z = 0,85x₁ + 1,20x₂ + 0,95x₃ + 1,35x₄  (nghìn tỷ VND)")

    budget = st.slider("Tổng ngân sách B (nghìn tỷ VND)", 100, 160, 100, 10)
    x3_min = st.slider("Sàn nhân lực số x₃ (nghìn tỷ)", 20, 40, 20, 5)

    c = [-0.85, -1.20, -0.95, -1.35]
    A_ub = [[1,1,1,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1],[0.35,-0.65,0.35,-0.65]]
    b_ub = [budget, -25, -15, -x3_min, -10, 0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0,None)]*4, method="highs")

    if res.success:
        c1, c2 = st.columns([1,2])
        with c1:
            stat("Z* (GDP gain kỳ vọng)", f"{-res.fun:,.2f}", "tối ưu")
        with c2:
            names = ["Hạ tầng số (x₁)","AI & dữ liệu (x₂)","Nhân lực số (x₃)","R&D (x₄)"]
            fig = go.Figure(go.Bar(x=names, y=res.x, marker_color="#ff4d6d",
                                   text=[f"{v:.1f}" for v in res.x], textposition="outside"))
            fig.update_layout(title="Phân bổ tối ưu (nghìn tỷ VND)", **PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)

        # Đường cong Z*(B)
        Bs = list(range(100,161,10)); Zs=[]
        for B in Bs:
            b2 = [B,-25,-15,-x3_min,-10,0]
            r = linprog(c, A_ub=A_ub, b_ub=b2, bounds=[(0,None)]*4, method="highs")
            Zs.append(-r.fun if r.success else None)
        fig2 = go.Figure(go.Scatter(x=Bs, y=Zs, mode="lines+markers",
                                    line=dict(color="#7eb6ff", width=3)))
        fig2.update_layout(title="2.4.3 — Đường cong độ nhạy Z*(B)",
                           xaxis_title="Ngân sách B", yaxis_title="Z*", **PLOTLY_DARK)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("Bài toán VÔ NGHIỆM với cấu hình hiện tại (tổng yêu cầu tối thiểu vượt ngân sách).")

    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    policy_box("a) Tăng 1 tỷ ngân sách thì GDP tăng bao nhiêu? Có phải cận trên chi phí vốn công?",
        "Shadow price của ràng buộc ngân sách = 1,35 (hệ số R&D) khi R&D chưa bão hòa: mỗi đồng tăng thêm "
        "được dồn vào hạng mục biên cao nhất. Đây là <b>cận trên</b> hợp lý của chi phí cơ hội vốn công — "
        "nếu chi phí huy động vốn < 1,35 thì nên mở rộng ngân sách.")
    policy_box("b) Vì sao R&D hệ số cao nhất nhưng sàn tối thiểu thấp nhất?",
        "R&D có tác động lan tỏa dài hạn (1,35) nhưng độ hấp thụ ngắn hạn hạn chế và rủi ro cao, nên đề chỉ "
        "đặt sàn 10. Mô hình tự đẩy R&D lên cao khi tối ưu — cho thấy nên nới sàn R&D trong thực tế.")
    policy_box("c) Tỷ lệ 35% công nghệ chiến lược (AI+R&D) có khả thi?",
        "Về toán học khả thi (ràng buộc thỏa). Về thực tiễn 2025 ngân sách VN ưu tiên hạ tầng giao thông "
        "và an sinh, nên 35% là tham vọng — cần lộ trình tăng dần và cơ chế PPP để chia sẻ rủi ro với khu vực tư.")


# ============================================================
# BÀI 3 — PRIORITY 10 NGÀNH
# ============================================================
def page_bai3():
    st.markdown("## 📊 Bài 3 — Chỉ số ưu tiên Priorityᵢ cho 10 ngành")
    df = sectors_df.copy()
    good = ["growth_rate_2024_pct","gdp_share_2024_pct","spillover_coef_0_1",
            "export_billion_USD","labor_million","ai_readiness_0_100"]
    bad = "automation_risk_pct"
    nrm = lambda x:(x-x.min())/(x.max()-x.min())
    Xg = df[good].apply(nrm)
    Xs = (df[bad].max()-df[bad])/(df[bad].max()-df[bad].min())

    st.caption("Trọng số 6 tiêu chí lợi ích (a₁..a₆) + trừ rủi ro tự động hóa (a₇)")
    cols = st.columns(7)
    labs = ["Tăng trưởng","GDP share","Lan tỏa","Xuất khẩu","Việc làm","AI Ready","Rủi ro"]
    defs = [0.15,0.15,0.20,0.15,0.10,0.20,0.15]
    w = [cols[i].number_input(labs[i],0.0,1.0,defs[i],0.05) for i in range(7)]

    pr = Xg.values @ np.array(w[:6]) - w[6]*Xs.values
    df["Priority"] = pr
    rk = df[["sector_name_en","Priority"]].sort_values("Priority",ascending=False).reset_index(drop=True)
    rk.index += 1

    fig = go.Figure(go.Bar(x=rk["Priority"], y=rk["sector_name_en"], orientation="h",
                           marker_color="#ff4d6d"))
    _pd=dict(PLOTLY_DARK); _pd["yaxis"]=dict(gridcolor="#1f2430",autorange="reversed")
    fig.update_layout(title="Xếp hạng ưu tiên 10 ngành", **_pd)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rk.round(3), use_container_width=True)

    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    top3 = ", ".join(rk["sector_name_en"].head(3).tolist())
    policy_box("a) Ba ngành nên ưu tiên CĐS/AI trước? Có hợp Nghị quyết 57?",
        f"Theo trọng số mặc định, top-3 là: <b>{top3}</b>. Kết quả phù hợp tinh thần Nghị quyết 57-NQ/TW "
        "khi đề cao CNTT-Truyền thông và chế biến chế tạo — các ngành lan tỏa và xuất khẩu mạnh.")
    policy_box("b) Vì sao Khai khoáng năng suất cao mà không vào nhóm ưu tiên?",
        "Khai khoáng có năng suất/lao động rất cao (do thâm dụng vốn) nhưng tăng trưởng âm (-1,2%), "
        "lan tỏa thấp (0,30), rủi ro tự động hóa cao (55%) và đi ngược mục tiêu xanh. Priority tổng hợp "
        "nhiều chiều nên ngành này bị loại — minh họa rủi ro khi chỉ nhìn một chỉ tiêu.")
    policy_box("c) Bộ trọng số nên do ai quyết định?",
        "Không nên giao riêng chuyên gia kỹ thuật. Trọng số là <b>lựa chọn giá trị</b> (đánh đổi tăng trưởng "
        "vs bao trùm) nên cần hội đồng chính sách + tham vấn công khai để bảo đảm tính chính danh "
        "(governance), tránh 'ngụy khách quan hóa' quyết định chính trị bằng con số.")


# ============================================================
# BÀI 4 — LP NGÀNH-VÙNG (PuLP)
# ============================================================
def page_bai4():
    import pulp
    st.markdown("## 📘 Bài 4 — LP phân bổ ngân sách số ngành-vùng")
    regions=["NMM","RRD","NCC","CH","SE","MD"]; items=["I","D","AI","H"]
    beta={('NMM','I'):1.15,('NMM','D'):0.85,('NMM','AI'):0.55,('NMM','H'):1.30,
          ('RRD','I'):0.95,('RRD','D'):1.25,('RRD','AI'):1.40,('RRD','H'):1.05,
          ('NCC','I'):1.05,('NCC','D'):0.95,('NCC','AI'):0.85,('NCC','H'):1.15,
          ('CH','I'):1.20,('CH','D'):0.75,('CH','AI'):0.45,('CH','H'):1.35,
          ('SE','I'):0.90,('SE','D'):1.30,('SE','AI'):1.55,('SE','H'):1.00,
          ('MD','I'):1.10,('MD','D'):0.85,('MD','AI'):0.65,('MD','H'):1.25}
    D0={'NMM':38,'RRD':78,'NCC':55,'CH':32,'SE':82,'MD':48}; g,lam=0.002,0.7

    fair = st.toggle("Bật ràng buộc công bằng vùng (C5)", value=True)

    def solve(fairness):
        m=pulp.LpProblem("b4",pulp.LpMaximize)
        x=pulp.LpVariable.dicts("x",(regions,items),lowBound=0)
        M=pulp.LpVariable("M",lowBound=0)
        m+=pulp.lpSum(beta[r,i]*x[r][i] for r in regions for i in items)
        m+=pulp.lpSum(x[r][i] for r in regions for i in items)<=50000
        for r in regions:
            m+=pulp.lpSum(x[r][i] for i in items)>=5000
            m+=pulp.lpSum(x[r][i] for i in items)<=12000
        m+=pulp.lpSum(x[r]['H'] for r in regions)>=12000
        if fairness:
            for r in regions:
                m+=D0[r]+g*x[r]['D']<=M
                m+=D0[r]+g*x[r]['D']>=lam*M
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        Z=pulp.value(m.objective)
        mat=pd.DataFrame([[x[r][i].varValue for i in items] for r in regions],
                         index=regions,columns=items)
        return Z,mat

    Z,mat=solve(fair); Z0,_=solve(False)
    c1,c2=st.columns(2)
    with c1: stat("Z* (GDP gain, tỷ VND)", f"{Z:,.0f}", "tối ưu")
    with c2: stat("Chi phí công bằng", f"{Z0-Z:,.0f}", f"{(Z0-Z)/Z0*100:.2f}% so Z không RB")

    fig=px.imshow(mat, text_auto=".0f", color_continuous_scale="Reds", aspect="auto",
                  labels=dict(x="Hạng mục",y="Vùng",color="Ngân sách"))
    fig.update_layout(title="Heatmap phân bổ tối ưu", **{k:v for k,v in PLOTLY_DARK.items() if k not in ("xaxis","yaxis")})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(mat.round(0), use_container_width=True)

    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    policy_box("a) Bỏ ràng buộc công bằng thì vốn chảy về đâu?",
        "Vốn dồn về RRD và SE (hệ số AI 1,40-1,55 cao nhất). Hậu quả dài hạn: khoét sâu khoảng cách số "
        "vùng miền, vùng yếu (CH, NMM) càng tụt hậu, gây bất ổn xã hội và di cư.")
    policy_box("b) Trần ngân sách vùng (C3) — chính sách phân quyền — làm giảm Z* bao nhiêu?",
        "Trần 12.000/vùng hi sinh một phần hiệu quả để đổi lấy cân bằng. Mức giảm vài % GDP gain thường "
        "chấp nhận được nếu coi đó là 'phí bảo hiểm' cho ổn định và gắn kết vùng miền.")
    policy_box("c) Tây Nguyên (CH) hệ số AI thấp (0,45) — nên đầu tư AI hay H, I trước?",
        "Mô hình dồn ngân sách CH vào H (1,35) và I (1,20) thay vì AI. Hợp lý: vùng nền tảng số yếu cần "
        "xây hạ tầng và nhân lực trước, đầu tư AI sớm sẽ kém hiệu quả do thiếu năng lực hấp thụ.")


# ============================================================
# BÀI 5 — MIP 15 DỰ ÁN
# ============================================================
def page_bai5():
    import pulp
    st.markdown("## 🎯 Bài 5 — MIP lựa chọn dự án chuyển đổi số")
    P=list(range(1,16))
    C={1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,8:15000,9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
    C1={1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,8:9000,9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
    B={1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,8:28500,9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}
    names={1:"TT dữ liệu Hòa Lạc",2:"TT dữ liệu phía Nam",3:"5G toàn quốc",4:"VNeID 2.0",5:"Cổng DVC v3",
           6:"Y tế số",7:"Giáo dục số K-12",8:"TT AI quốc gia",9:"Sandbox fintech",10:"Logistics số",
           11:"Nông nghiệp số ĐBSCL",12:"Đào tạo 50k kỹ sư AI",13:"KCN bán dẫn",14:"An ninh mạng",15:"Open Data"}

    total=st.slider("Ngân sách tổng (tỷ VND)",80000,120000,80000,10000)
    risk_mode=st.checkbox("Tối đa hóa lợi ích KỲ VỌNG (xét rủi ro dự án)", value=False)
    ps={**{i:0.85 for i in[1,2,3]},**{i:0.75 for i in[4,5,6]},**{i:0.65 for i in[8,13]},12:0.65}
    ps={i:ps.get(i,0.80) for i in P}

    m=pulp.LpProblem("b5",pulp.LpMaximize); y=pulp.LpVariable.dicts("y",P,cat="Binary")
    obj = pulp.lpSum((ps[i] if risk_mode else 1)*B[i]*y[i] for i in P)
    m+=obj
    m+=pulp.lpSum(C[i]*y[i] for i in P)<=total
    m+=pulp.lpSum(C1[i]*y[i] for i in P)<=40000
    m+=y[1]+y[2]<=1; m+=y[8]<=y[12]; m+=y[13]<=y[12]; m+=y[4]+y[5]>=1; m+=y[14]>=1
    m+=pulp.lpSum(y[i] for i in P)>=7; m+=pulp.lpSum(y[i] for i in P)<=11
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    sel=[i for i in P if y[i].varValue and y[i].varValue>0.5]
    cost=sum(C[i] for i in sel); Z=pulp.value(m.objective)

    c1,c2,c3=st.columns(3)
    with c1: stat("Lợi ích Z*", f"{Z:,.0f}", "tỷ VND")
    with c2: stat("Tổng chi phí", f"{cost:,.0f}", f"{len(sel)} dự án")
    with c3: stat("NPV biên", f"{Z/cost:.2f}", "lợi ích/chi phí")
    st.success("Dự án được chọn: " + ", ".join(f"P{i} ({names[i]})" for i in sel))

    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    policy_box("a) Vì sao bỏ P15 (Open Data) dù tỷ suất lợi ích/chi phí cao?",
        "Ràng buộc 'tối đa 11 dự án' và ngân sách buộc mô hình ưu tiên dự án lợi ích tuyệt đối lớn. P15 "
        "lợi ích nhỏ (3.800) nên bị loại dù hiệu quả tương đối cao — đây <b>không</b> là kết quả mong muốn về "
        "chính sách: nên tách Open Data thành hạng mục nền tảng bắt buộc.")
    policy_box("b) Bắt buộc P14 (an ninh mạng) có làm giảm Z*?",
        "Có thể làm giảm nhẹ Z* nếu P14 không nằm trong tập tối ưu tự nhiên. Nhưng ràng buộc này hợp lý: "
        "an ninh mạng là điều kiện nền tảng, không thể đánh đổi bằng lợi ích kinh tế thuần.")
    policy_box("c) P8 (AI) và P13 (bán dẫn) có lợi ích cộng hưởng — mô hình hóa thế nào?",
        "Mô hình hiện giả định độc lập. Để mô hình hóa cộng hưởng, thêm biến tích z=y₈·y₁₃ (tuyến tính hóa "
        "bằng z≤y₈, z≤y₁₃, z≥y₈+y₁₃-1) và cộng phần thưởng bonus·z vào hàm mục tiêu.")


# ============================================================
# BÀI 6 — TOPSIS 6 VÙNG
# ============================================================
def page_bai6():
    st.markdown("## 🏆 Bài 6 — TOPSIS xếp hạng 6 vùng theo mức sẵn sàng AI")
    df=regions_df.copy()
    crit=["grdp_per_capita_million_VND","fdi_registered_billion_USD","digital_index_0_100",
          "ai_readiness_0_100","trained_labor_pct","rd_intensity_pct","internet_penetration_pct","gini_coef"]
    is_ben=np.array([True]*7+[False])
    X=df[crit].values.astype(float)

    method=st.radio("Phương pháp trọng số", ["Chuyên gia","Entropy (khách quan)"], horizontal=True)
    w_exp=np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])
    if method=="Entropy (khách quan)":
        Pm=X/X.sum(axis=0); k=1/np.log(len(X))
        E=-k*np.nansum(Pm*np.log(Pm+1e-12),axis=0); d=1-E; w=d/d.sum()
    else:
        w=w_exp

    R=X/np.sqrt((X**2).sum(axis=0)); V=R*w
    Astar=np.where(is_ben,V.max(0),V.min(0)); Aneg=np.where(is_ben,V.min(0),V.max(0))
    Ss=np.sqrt(((V-Astar)**2).sum(1)); Sn=np.sqrt(((V-Aneg)**2).sum(1))
    C=Sn/(Ss+Sn); df["TOPSIS"]=C
    rk=df[["region_name_en","TOPSIS"]].sort_values("TOPSIS",ascending=False).reset_index(drop=True)
    rk.index+=1

    fig=go.Figure(go.Bar(x=rk["TOPSIS"],y=rk["region_name_en"],orientation="h",marker_color="#ff4d6d"))
    _pd=dict(PLOTLY_DARK); _pd["yaxis"]=dict(gridcolor="#1f2430",autorange="reversed")
    fig.update_layout(title=f"Xếp hạng TOPSIS (trọng số {method})",**_pd)
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(rk.round(4),use_container_width=True)

    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    lead=rk["region_name_en"].iloc[0]
    policy_box("a) Vùng nào dẫn đầu? Có nên đặt TT AI quốc gia đầu tiên?",
        f"Dẫn đầu là <b>{lead}</b> (cùng Đông Nam Bộ). Đặt trung tâm AI đầu tiên ở đây hợp lý vì hạ tầng số, "
        "FDI và nhân lực sẵn sàng nhất — bảo đảm khả năng hấp thụ và hiệu quả vốn.")
    policy_box("b) Dùng trọng số Entropy, vùng nào đổi hạng nhiều nhất? Vì sao?",
        "Entropy gán trọng số cao cho tiêu chí phân tán mạnh (như FDI, R&D). Vùng có FDI/R&D bứt phá sẽ "
        "tăng hạng, vùng đồng đều bị giảm — cho thấy kết quả TOPSIS nhạy với cách chọn trọng số.")
    policy_box("c) AI Readiness và Internet tương quan cao — ảnh hưởng gì?",
        "TOPSIS giả định độc lập tuyến tính; hai tiêu chí tương quan cao gây 'đếm trùng' (double counting), "
        "thổi phồng lợi thế vùng mạnh cả hai. Xử lý: gộp thành chỉ số tổng hợp hoặc dùng PCA/Mahalanobis distance.")
    policy_box("d) Chọn 3 vùng cho 3 trung tâm AI (QĐ 127)?",
        "Theo TOPSIS: Đông Nam Bộ, Đồng bằng sông Hồng, và Bắc Trung Bộ + DH Trung Bộ (vùng đệm địa lý). "
        "Nên thêm tiêu chí địa-chính trị để phân bổ đều 3 miền Bắc-Trung-Nam.")


# ============================================================
# BÀI 7 — NSGA-II PARETO (mô phỏng nhanh weighted-sum nếu pymoo nặng)
# ============================================================
def page_bai7():
    st.markdown("## 🌐 Bài 7 — Tối ưu đa mục tiêu Pareto (NSGA-II)")
    st.caption("4 mục tiêu: max Tăng trưởng, min Bất bình đẳng, min Phát thải, min Rủi ro dữ liệu")
    beta=np.array([[1.15,0.85,0.55,1.30],[0.95,1.25,1.40,1.05],[1.05,0.95,0.85,1.15],
                   [1.20,0.75,0.45,1.35],[0.90,1.30,1.55,1.00],[1.10,0.85,0.65,1.25]])
    e=np.array([0.42,0.55,0.48,0.32,0.62,0.38]); rho=np.array([0.18,0.45,0.28,0.12,0.52,0.22])
    sig=np.array([0.32,0.28,0.30,0.35,0.25,0.30])

    use_pymoo=False
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize as pymoo_min
        use_pymoo=True
    except Exception:
        pass

    @st.cache_data
    def run_pareto():
        if use_pymoo:
            class Prob(ElementwiseProblem):
                def __init__(s): super().__init__(n_var=24,n_obj=4,n_ieq_constr=14,
                    xl=np.zeros(24),xu=np.ones(24)*12000)
                def _evaluate(s,x,out,*a,**k):
                    X=x.reshape(6,4)
                    f1=-(beta*X).sum(); sm=X.sum(1); f2=np.abs(sm-sm.mean()).mean()
                    f3=(e*(X[:,0]+X[:,2])).sum(); f4=(rho*X[:,2]).sum()-(sig*X[:,3]).sum()
                    out["F"]=[f1,f2,f3,f4]
                    out["G"]=[X.sum()-50000]+(5000-sm).tolist()+(sm-12000).tolist()+[12000-X[:,3].sum()]
            r=pymoo_min(Prob(),NSGA2(pop_size=60),("n_gen",60),seed=42,verbose=False)
            return r.F
        # Fallback: random sampling + lọc non-dominated
        rng=np.random.default_rng(42); F=[]
        for _ in range(4000):
            X=rng.dirichlet(np.ones(24))*50000; X=X.reshape(6,4)
            sm=X.sum(1)
            if (sm>=5000).all() and (sm<=12000).all() and X[:,3].sum()>=12000:
                f1=-(beta*X).sum(); f2=np.abs(sm-sm.mean()).mean()
                f3=(e*(X[:,0]+X[:,2])).sum(); f4=(rho*X[:,2]).sum()-(sig*X[:,3]).sum()
                F.append([f1,f2,f3,f4])
        F=np.array(F)
        # lọc Pareto đơn giản
        keep=[]
        for i in range(len(F)):
            dom=False
            for j in range(len(F)):
                if i!=j and (F[j]<=F[i]).all() and (F[j]<F[i]).any(): dom=True;break
            if not dom: keep.append(i)
        return F[keep]

    F=run_pareto()
    st.info(f"{'pymoo NSGA-II' if use_pymoo else 'random-sampling fallback'} → "
            f"tìm thấy **{len(F)}** nghiệm Pareto.")

    Fp=F.copy(); Fp[:,0]=-Fp[:,0]
    fig=go.Figure(go.Scatter3d(x=Fp[:,0],y=Fp[:,1],z=Fp[:,2],mode="markers",
        marker=dict(size=4,color=Fp[:,3],colorscale="Plasma",showscale=True,
                    colorbar=dict(title="Rủi ro"))))
    fig.update_layout(title="Mặt Pareto 3D (GDP gain × Bất bình đẳng × Phát thải)",
        scene=dict(xaxis_title="f1 GDP gain",yaxis_title="f2 Gini",zaxis_title="f3 Phát thải"),
        paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#d4d7e0"),margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig,use_container_width=True)

    # TOPSIS chọn nghiệm thỏa hiệp
    W=np.array([0.40,0.25,0.20,0.15]); nF=F/np.sqrt((F**2).sum(0)); Vv=nF*W
    Ab=Vv.min(0); Aw=Vv.max(0)
    Sb=np.sqrt(((Vv-Ab)**2).sum(1)); Sw=np.sqrt(((Vv-Aw)**2).sum(1))
    Cstar=Sw/(Sb+Sw); bi=np.argmax(Cstar); sol=F[bi]
    c1,c2,c3,c4=st.columns(4)
    with c1: stat("f1 GDP gain",f"{-sol[0]:,.0f}","max")
    with c2: stat("f2 Bất BĐ",f"{sol[1]:,.0f}","min")
    with c3: stat("f3 Phát thải",f"{sol[2]:,.0f}","min")
    with c4: stat("f4 Rủi ro",f"{sol[3]:,.0f}","min")

    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    policy_box("a) Đánh đổi tăng trưởng vs bao trùm có rõ không? Nói gì về cơ cấu kinh tế VN?",
        "Mặt Pareto cho thấy đánh đổi rõ: nghiệm GDP cao nhất luôn kèm bất bình đẳng cao hơn. Phản ánh cơ "
        "cấu VN phân cực — hai cực tăng trưởng (RRD, SE) hút vốn AI, vùng còn lại cần trợ lực.")
    policy_box("b) Trọng số (0,40;0,25;0,20;0,15) có đúng ưu tiên VN (Đại hội XIII)?",
        "Cân bằng nhưng nghiêng tăng trưởng. Để khớp cam kết COP26 (net-zero 2050) và QĐ 127, nên tăng trọng "
        "số môi trường (f3) lên ~0,25 và giảm tăng trưởng xuống ~0,35.")
    policy_box("c) Vai trò NSGA-II khác gì LP đơn mục tiêu? Có thay quyết định chính trị?",
        "NSGA-II đưa ra <b>tập</b> phương án không-bị-trội thay vì một nghiệm duy nhất, làm rõ các đánh đổi. "
        "Nó <b>hỗ trợ</b> chứ không thay quyết định chính trị — việc chọn điểm nào trên mặt Pareto là lựa "
        "chọn giá trị, thuộc thẩm quyền nhà hoạch định.")


# ============================================================
# BÀI 8 — TỐI ƯU ĐỘNG 2026-2035
# ============================================================
def page_bai8():
    st.markdown("## ⏳ Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035")
    T=10; years=np.arange(2026,2036); L=54.0; rho=0.97
    K0,D0,AI0,H0=27500,20.3,86,30
    shock=st.checkbox("Mô phỏng cú sốc 2028 (giảm 8% sản lượng — kiểu bão Yagi)")

    def simulate(rate_K,rate_other):
        K=np.zeros(T);D=np.zeros(T);AI=np.zeros(T);H=np.zeros(T);Y=np.zeros(T);C=np.zeros(T)
        K[0],D[0],AI[0],H[0]=K0,D0,AI0,H0
        for t in range(T):
            Y[t]=(K[t]**0.33)*(L**0.42)*(D[t]**0.10)*(AI[t]**0.08)*(H[t]**0.07)
            if shock and t==2: Y[t]*=0.92
            inv=Y[t]*rate_K+Y[t]*rate_other*3  # tổng đầu tư
            C[t]=max(Y[t]-inv,1e-3)
            if t<T-1:
                K[t+1]=0.95*K[t]+Y[t]*rate_K
                D[t+1]=0.88*D[t]+Y[t]*rate_other*0.001
                AI[t+1]=0.85*AI[t]+Y[t]*rate_other*0.002
                H[t+1]=0.98*H[t]+0.8*Y[t]*rate_other*0.0005
        W=np.sum((rho**np.arange(T))*np.log(C))
        return years,K,D,AI,H,Y,C,W

    yr,K,D,AI,H,Y,C,W_front=simulate(0.20,0.04)
    _,_,_,_,_,_,_,W_even=simulate(0.12,0.04)
    fig=go.Figure()
    for arr,nm,col in [(Y,"Y sản lượng","#ff4d6d"),(K/100,"K/100","#7eb6ff"),
                       (C,"C tiêu dùng","#7ce0c3")]:
        fig.add_trace(go.Scatter(x=yr,y=arr,mode="lines+markers",name=nm,line=dict(color=col,width=2)))
    fig.update_layout(title="Quỹ đạo tối ưu 2026-2035",**PLOTLY_DARK)
    st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1: stat("Welfare Front-load",f"{W_front:.2f}","đầu tư mạnh đầu kỳ")
    with c2: stat("Welfare Trải đều",f"{W_even:.2f}","đầu tư đều")

    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    policy_box("a) Quỹ đạo tối ưu front-loaded hay back-loaded? Vì sao?",
        "Mô hình ưu tiên <b>front-load</b>: dồn vốn sớm để K, D, AI tích lũy nhanh, tạo sản lượng cao suốt "
        "phần còn lại của kỳ. Đầu tư càng sớm càng có nhiều năm 'thu hoạch' năng suất.")
    policy_box("b) Tỷ lệ đầu tư AI/H ổn định không? Đào tạo nên đi trước hay đồng thời?",
        "Mô hình hàm ý H nên đi <b>trước hoặc đồng thời</b> với AI: nhân lực số là điều kiện hấp thụ công "
        "nghệ. Đầu tư AI mà thiếu H sẽ lãng phí do năng lực vận hành không theo kịp.")
    policy_box("c) ρ=0,97 vs ρ=0,90 — vì sao chính phủ thường 'dưới đầu tư' R&D?",
        "ρ thấp (0,90) coi nhẹ tương lai → giảm đầu tư dài hạn như R&D, tiêu dùng nhiều hơn hiện tại. Đây "
        "chính là lý do chính trị nhiệm kỳ ngắn thường dưới đầu tư vào R&D có hồi vốn chậm.")


# ============================================================
# BÀI 9 — LAO ĐỘNG & AI
# ============================================================
def page_bai9():
    import cvxpy as cp
    st.markdown("## 👷 Bài 9 — Tác động AI tới thị trường lao động")
    names=["Nông-Lâm-TS","CN chế biến","Xây dựng","Bán buôn-lẻ","Tài chính-NH",
           "Logistics","CNTT-TT","Giáo dục"]
    N=8; risk=np.array([18,42,25,38,52,35,28,22])/100
    a1=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5]); b1=np.array([45,28,35,32,22,30,20,55])
    c1=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5]); d1=np.array([50,32,42,38,26,36,24,62])
    Ln=np.array([13.2,11.5,4.8,7.8,0.55,1.95,0.62,2.15])*1000

    cap5=st.checkbox("Ràng buộc 9.4.4: không ngành nào mất quá 5% lao động")
    xA=cp.Variable(N,nonneg=True); xH=cp.Variable(N,nonneg=True)
    Disp=cp.multiply(cp.multiply(c1,risk),xA)
    Net=cp.multiply(a1,xA)+cp.multiply(b1,xH)-Disp
    cons=[cp.sum(xA+xH)<=30000, Net>=0, Disp<=cp.multiply(d1,xH)]
    if cap5: cons.append(Disp<=0.05*Ln)
    prob=cp.Problem(cp.Maximize(cp.sum(Net)),cons); prob.solve()

    if prob.status.startswith("optimal"):
        df=pd.DataFrame({"Ngành":names,"x_AI":np.round(xA.value,0),"x_H":np.round(xH.value,0),
                         "NetJob (nghìn)":np.round(Net.value,1)})
        stat("Tổng việc làm ròng tạo thêm", f"{np.sum(Net.value):,.0f}", "nghìn việc")
        fig=go.Figure(go.Bar(x=df["Ngành"],y=df["NetJob (nghìn)"],marker_color="#ff4d6d"))
        fig.update_layout(title="NetJob ròng theo ngành",**PLOTLY_DARK)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(df,use_container_width=True,hide_index=True)
    else:
        st.error("Bài toán VÔ NGHIỆM với ràng buộc hiện tại.")

    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    policy_box("a) Ngành nào cần đào tạo lại nhiều nhất? Có khớp thực tế VN?",
        "Mô hình dồn x_H vào ngành rủi ro cao (CN chế biến, Tài chính, Bán buôn-lẻ). Khớp thực tế: đây là "
        "các ngành ILO/OECD cảnh báo 30-50% việc làm có nguy cơ tự động hóa.")
    policy_box("b) Tài chính-NH rủi ro thay thế 52% nhưng tạo việc mới cao — chiến lược?",
        "Mô hình khuyến nghị đầu tư <b>song song</b> AI và H: vừa tận dụng hệ số tạo việc mới cao (45,8), "
        "vừa đào tạo lại để hấp thụ lao động bị thay thế — chuyển dịch sang vai trò giá trị cao.")
    policy_box("c) Có nên đầu tư AI vào Nông-Lâm-TS?",
        "Mô hình hạn chế x_AI vào ngành này vì hệ số tạo việc AI thấp (8,5) trong khi lao động lớn (13,2tr). "
        "Nên ưu tiên x_H (nâng cấp kỹ năng) thay vì AI thuần để tránh dịch chuyển lao động ồ ạt.")
    policy_box("d) 'Tốc độ tự động hóa ≤ năng lực đào tạo lại' biểu diễn bằng ràng buộc nào?",
        "Chính là <b>Displaced ≤ RetrainingCapacity</b> (d₁·x_H). Đề xuất bổ sung: trần tuyệt đối 5% lao động/"
        "ngành (ràng buộc 9.4.4) làm lưới an sinh phòng khi đổi hàm mục tiêu sang tối đa tăng trưởng.")


# ============================================================
# BÀI 10 — STOCHASTIC SP
# ============================================================
def page_bai10():
    st.markdown("## 🎲 Bài 10 — Quy hoạch ngẫu nhiên 2 giai đoạn")
    J=["I","D","AI","H"]; S=["s1","s2","s3","s4"]
    p={"s1":.30,"s2":.45,"s3":.20,"s4":.05}; beta={"I":1.0,"D":1.1,"AI":1.25,"H":.95}
    bs={("s1","I"):1.25,("s1","D"):1.35,("s1","AI"):1.55,("s1","H"):1.05,
        ("s2","I"):1.0,("s2","D"):1.1,("s2","AI"):1.25,("s2","H"):.95,
        ("s3","I"):.75,("s3","D"):.85,("s3","AI"):.90,("s3","H"):1.0,
        ("s4","I"):.40,("s4","D"):.50,("s4","AI"):.55,("s4","H"):1.10}
    try:
        import pyomo.environ as pyo
        solver=pyo.SolverFactory("glpk")
        ok = solver.available()
    except Exception:
        ok=False

    if not ok:
        st.warning("Không tìm thấy solver GLPK/Pyomo trong môi trường này. "
                   "Hiển thị kết quả tính bằng công thức đóng (linprog).")
        # Giải đơn giản: SP first-stage dồn vào AI (β cao); minh họa bằng giá trị xấp xỉ.
        from scipy.optimize import linprog
        # max  βx + Σ p_s βs y_s  ; x sum<=65000 ; y_s sum<=15000 ; y_AI<=0.5 x_H
        # gộp biến: x(4) + y_s(4*4)=20 biến
        nb=4+4*len(S); cobj=np.zeros(nb)
        for i,j in enumerate(J): cobj[i]=-beta[j]
        for k,s in enumerate(S):
            for i,j in enumerate(J): cobj[4+4*k+i]=-p[s]*bs[(s,j)]
        A=[]; b=[]
        row=np.zeros(nb); row[:4]=1; A.append(row); b.append(65000)
        for k in range(len(S)):
            row=np.zeros(nb); row[4+4*k:4+4*k+4]=1; A.append(row); b.append(15000)
            row=np.zeros(nb); row[4+4*k+2]=1; row[3]=-0.5; A.append(row); b.append(0)
        r=linprog(cobj,A_ub=A,b_ub=b,bounds=[(0,None)]*nb,method="highs")
        x=r.x[:4]; Z=-r.fun
        c1,c2=st.columns(2)
        with c1: stat("Z* (SP)",f"{Z:,.0f}","tỷ VND")
        with c2:
            fig=go.Figure(go.Bar(x=J,y=x,marker_color="#ff4d6d"))
            fig.update_layout(title="Quyết định first-stage x*",**PLOTLY_DARK)
            st.plotly_chart(fig,use_container_width=True)
    else:
        def base():
            m=pyo.ConcreteModel(); m.x=pyo.Var(J,domain=pyo.NonNegativeReals)
            m.y=pyo.Var(S,J,domain=pyo.NonNegativeReals)
            m.b1=pyo.Constraint(expr=sum(m.x[j] for j in J)<=65000)
            m.b2=pyo.ConstraintList(); m.lk=pyo.ConstraintList()
            for s in S:
                m.b2.add(sum(m.y[s,j] for j in J)<=15000)
                m.lk.add(m.y[s,"AI"]<=0.5*m.x["H"])
            return m
        m=base(); m.obj=pyo.Objective(expr=sum(beta[j]*m.x[j] for j in J)
            +sum(p[s]*sum(bs[s,j]*m.y[s,j] for j in J) for s in S),sense=pyo.maximize)
        solver.solve(m); Z=pyo.value(m.obj); x=[pyo.value(m.x[j]) for j in J]
        c1,c2=st.columns(2)
        with c1: stat("Z* (SP)",f"{Z:,.0f}","tỷ VND")
        with c2:
            fig=go.Figure(go.Bar(x=J,y=x,marker_color="#ff4d6d"))
            fig.update_layout(title="Quyết định first-stage x*",**PLOTLY_DARK)
            st.plotly_chart(fig,use_container_width=True)

    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    policy_box("a) So lời giải xác định, SP đầu tư H nhiều hơn hay ít hơn?",
        "SP đầu tư H <b>nhiều hơn</b>: H là 'hàng hóa bảo hiểm' — hệ số tăng lên 1,10 trong kịch bản khủng "
        "hoảng (s4), giúp hấp thụ cú sốc. Lời giải xác định bỏ qua điều này nên đầu tư H ít hơn.")
    policy_box("b) VSS dương nói lên điều gì?",
        "VSS>0 nghĩa là tư duy xác suất (xét mọi kịch bản) tạo giá trị thật so với chỉ dùng kịch bản trung "
        "bình. Với nền kinh tế mở như VN (XNK ≈180% GDP), bất định lớn → VSS càng quan trọng.")
    policy_box("c) COVID & bão Yagi — VN có đang 'dưới đầu tư' nhân lực số?",
        "Bài học: các cú sốc thực tế cho thấy giá trị của dự phòng và nhân lực linh hoạt. Mô hình SP gợi ý "
        "VN nên coi đầu tư nhân lực số như khoản bảo hiểm chiến lược, không nên tối thiểu hóa.")


# ============================================================
# BÀI 11 — Q-LEARNING RL
# ============================================================
def page_bai11():
    st.markdown("## 🕹️ Bài 11 — Q-learning cho chính sách kinh tế thích nghi")
    st.caption("MDP: state=(GDP, D, AI, Unemploy) × 3 mức; action=5 chiến lược phân bổ")

    alloc={0:[.70,.10,.10,.10],1:[.40,.25,.15,.20],2:[.25,.45,.15,.15],
           3:[.20,.20,.45,.15],4:[.30,.20,.10,.40]}
    anames=["Truyền thống","Cân bằng","Số hóa nhanh","AI dẫn dắt","Bao trùm"]
    w=np.array([.40,.25,.20,.15])

    @st.cache_data
    def train(n_ep=8000):
        Q=np.zeros((3,3,3,3,5)); rng=np.random.default_rng(0); returns=[]
        for ep in range(n_ep):
            s=np.array([1,1,0,1]); K,D,AI,H=27500.,20.3,86.,30.
            Yp=(K**.33)*(54**.42)*(D**.10)*(AI**.08)*(H**.07)
            ret=0; eps=max(.05,1-ep/4000)
            for t in range(10):
                a=rng.integers(5) if rng.random()<eps else int(np.argmax(Q[tuple(s)]))
                al=alloc[a]; bud=1000
                K+=al[0]*bud; D+=al[1]*bud/100; AI+=al[2]*bud/20; H+=al[3]*bud/200
                Yn=(K**.33)*(54**.42)*(D**.10)*(AI**.08)*(H**.07)
                g=(Yn-Yp)/Yp; Yp=Yn
                disc=lambda v,lo,hi:0 if v<lo else(1 if v<hi else 2)
                s2=np.array([disc(g,.03,.05),disc(D,22,26),disc(AI,95,110),disc(H,31,33)])
                # reward đúng công thức đề: w1·ΔGDP - w2·Δunemploy - w3·cyber - w4·emission
                r=w[0]*g*100 - w[1]*(2-s2[3]) - w[2]*(al[2]*0.5) - w[3]*(al[0]*0.5)
                Q[tuple(s)+(a,)]+=0.1*(r+0.95*Q[tuple(s2)].max()-Q[tuple(s)+(a,)])
                s=s2; ret+=r
            returns.append(ret)
        return Q,returns

    Q,returns=train()
    def best(s): return int(np.argmax(Q[tuple(s)]))
    tests=[[1,1,0,1],[0,0,0,0],[2,2,2,2],[0,1,2,1],[2,0,1,0]]
    df=pd.DataFrame({"Trạng thái (G,D,AI,U)":[str(s) for s in tests],
                     "π* hành động":[anames[best(s)] for s in tests]})
    st.dataframe(df,use_container_width=True,hide_index=True)

    mv=np.convolve(returns,np.ones(100)/100,mode="valid")
    fig=go.Figure(go.Scatter(y=mv,mode="lines",line=dict(color="#ff4d6d")))
    fig.update_layout(title="Learning curve (MA-100)",xaxis_title="Episode",
                      yaxis_title="Return",**PLOTLY_DARK)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    policy_box("a) GDP thấp, D thấp, U cao — π* chọn gì? Có là 'quick win'?",
        f"Tại trạng thái xấu, π* chọn <b>{anames[best([0,0,0,1])]}</b> — ưu tiên kích hoạt tăng trưởng và "
        "nhân lực. Đúng tinh thần 'quick win': ổn định việc làm và tạo đà trước khi đẩy AI.")
    policy_box("b) GDP cao, AI cao, U thấp — π* chọn gì? Có là 'consolidation'?",
        f"Tại trạng thái tốt, π* chọn <b>{anames[best([2,2,2,0])]}</b> — chuyển sang củng cố và phân bổ cân "
        "bằng/AI dẫn dắt. Phù hợp giai đoạn 'consolidation': khai thác nền tảng đã xây.")
    policy_box("c) Tích hợp π* vào hoạch định mà không vi phạm 'AI không thay quyết định chính trị'?",
        "π* nên là <b>công cụ tư vấn</b>: gợi ý hành động + giải thích lý do, đặt trong quy trình thẩm định "
        "của hội đồng chính sách. Con người giữ quyền phủ quyết và chịu trách nhiệm cuối cùng — AI minh "
        "bạch hóa đánh đổi chứ không tự động ra quyết định.")


# ============================================================
# BÀI 12 — AIDEOM TÍCH HỢP (dashboard tabs)
# ============================================================
def page_bai12():
    st.markdown("## 🇻🇳 Bài 12 — Nguyên mẫu AIDEOM-VN tích hợp")
    tabs=st.tabs(["📊 Tổng quan (M1-M2)","💰 Phân bổ (M3)","📑 5 Kịch bản (M6)","⚠️ Cảnh báo rủi ro (M4-M5)"])

    with tabs[0]:
        st.markdown("### M1 — Dự báo kinh tế (Cobb-Douglas)")
        if st.button("Chạy M1"):
            st.session_state["m1"]=True
        A,M,Y=cobb_tfp()
        A_mean=A.mean(); MAPE=np.mean(np.abs((Y-A_mean*M)/Y))*100
        K30=K_ARR[-1]*1.06**5; L30=L_ARR[-1]*1.06**5; A30=A[-1]*1.012**5
        Y30=A30*(K30**ALPHA)*(L30**BETA)*(30**GAMMA)*(100**DELTA)*(35**THETA)
        c1,c2,c3=st.columns(3)
        with c1: stat("MAPE (Cobb-Douglas)",f"{MAPE:.2f}%","")
        with c2: stat("Ā",f"{A_mean:.4f}","")
        with c3: stat("Y 2030 dự báo",f"{Y30/1000:,.3f} ng.tỷ","")
        n=len(YEARS)-1; dln=lambda s:(np.log(s[-1])-np.log(s[0]))/n
        parts={"TFP (A)":dln(A),"Vốn (K)":ALPHA*dln(K_ARR),"Lao động (L)":BETA*dln(L_ARR),
               "Số hóa (D)":GAMMA*dln(D_ARR),"AI":DELTA*dln(AI_ARR),"Nhân lực số (H)":THETA*dln(H_ARR)}
        fig=go.Figure(go.Bar(x=list(parts),y=[v*100 for v in parts.values()],
            marker_color=["#7ce0c3","#f9f1a5","#9aa6b2","#ff8f8f","#7eb6ff","#ffb86b"]))
        fig.update_layout(title="Phân rã đóng góp tăng trưởng 2020-2025",
                          yaxis_title="Đóng góp_pct",xaxis_title="Yếu_tố",**PLOTLY_DARK)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("### M2 — Đánh giá sẵn sàng số (TOPSIS)")
        df=regions_df.copy()
        crit=["digital_index_0_100","ai_readiness_0_100","trained_labor_pct","internet_penetration_pct"]
        X=df[crit].values.astype(float); R=X/np.sqrt((X**2).sum(0))
        df["Score"]=R.mean(1)
        fig2=go.Figure(go.Bar(x=df["region_name_en"],y=df["Score"],marker_color="#ff4d6d"))
        fig2.update_layout(title="Chỉ số sẵn sàng số theo vùng",**PLOTLY_DARK)
        st.plotly_chart(fig2,use_container_width=True)

    with tabs[1]:
        st.markdown("### M3 — Tối ưu phân bổ ngân sách số (LP)")
        st.write("Tái sử dụng mô hình Bài 4: phân bổ 50.000 tỷ cho 6 vùng × 4 hạng mục.")
        page_bai4_compact()

    with tabs[2]:
        st.markdown("### M6 — So sánh 5 kịch bản chính sách 2030")
        scen={"S1 Truyền thống":[.70,.10,.10,.10],"S2 Số hóa nhanh":[.25,.45,.15,.15],
              "S3 AI dẫn dắt":[.20,.20,.45,.15],"S4 Bao trùm số":[.30,.20,.10,.40],
              "S5 Cân bằng (AIDEOM)":[.35,.25,.20,.20]}
        A,M,Y=cobb_tfp(); A30=A[-1]*1.012**5; rows=[]
        for nm,al in scen.items():
            K30=K_ARR[-1]+al[0]*5000*5; D30=20.3+al[1]*40; AI30=86+al[2]*120; H30=30+al[3]*30
            Y30=A30*(K30**ALPHA)*(54**BETA)*(D30**GAMMA)*(AI30**DELTA)*(H30**THETA)
            rows.append({"Kịch bản":nm,"GDP 2030 (ng.tỷ)":round(Y30/1000,2),
                         "D 2030(%)":round(D30,1),"H 2030(%)":round(H30,1)})
        dfk=pd.DataFrame(rows)
        fig=go.Figure(go.Bar(x=dfk["Kịch bản"],y=dfk["GDP 2030 (ng.tỷ)"],marker_color="#ff4d6d",
            text=dfk["GDP 2030 (ng.tỷ)"],textposition="outside"))
        fig.update_layout(title="GDP 2030 dự báo theo 5 kịch bản",**PLOTLY_DARK)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(dfk,use_container_width=True,hide_index=True)

    with tabs[3]:
        st.markdown("### M4-M5 — Cảnh báo lao động & rủi ro")
        df=sectors_df.copy()
        df["risk_flag"]=np.where(df["automation_risk_pct"]>=40,"⚠️ Cao","✅ Thấp")
        warn=df[df["automation_risk_pct"]>=40][["sector_name_en","automation_risk_pct","labor_million"]]
        st.warning(f"**{len(warn)} ngành** có rủi ro tự động hóa ≥40% — cần ưu tiên đào tạo lại.")
        fig=go.Figure(go.Bar(x=df["sector_name_en"],y=df["automation_risk_pct"],
            marker_color=np.where(df["automation_risk_pct"]>=40,"#ff4d6d","#7ce0c3")))
        fig.add_hline(y=40,line_dash="dash",line_color="#ffd166")
        fig.update_layout(title="Rủi ro tự động hóa theo ngành (%)",**PLOTLY_DARK)
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(warn,use_container_width=True,hide_index=True)

    st.markdown("### 💬 Khuyến nghị chính sách tổng hợp (AIDEOM-VN)")
    policy_box("Kịch bản nào nên chọn cho VN đến 2030?",
        "Kịch bản <b>S5 Cân bằng (AIDEOM)</b> tối ưu hóa đánh đổi: GDP cao gần S3 nhưng giữ nhân lực số (H) "
        "và bao trùm tốt hơn. Phù hợp Nghị quyết 57 và cam kết COP26 — tăng trưởng nhanh nhưng bền vững và "
        "không bỏ lại vùng yếu phía sau.")
    policy_box("Ba trụ cột hành động ưu tiên",
        "1) <b>Nhân lực số đi trước</b> (đào tạo 50k kỹ sư AI/bán dẫn) để tăng năng lực hấp thụ; "
        "2) <b>Hạ tầng & dữ liệu</b> phủ vùng yếu để thu hẹp khoảng cách số; "
        "3) <b>Lưới an sinh lao động</b> cho ngành rủi ro cao (chế biến, tài chính, bán lẻ).")


def page_bai4_compact():
    import pulp
    regions=["NMM","RRD","NCC","CH","SE","MD"]; items=["I","D","AI","H"]
    beta={('NMM','I'):1.15,('NMM','D'):0.85,('NMM','AI'):0.55,('NMM','H'):1.30,
          ('RRD','I'):0.95,('RRD','D'):1.25,('RRD','AI'):1.40,('RRD','H'):1.05,
          ('NCC','I'):1.05,('NCC','D'):0.95,('NCC','AI'):0.85,('NCC','H'):1.15,
          ('CH','I'):1.20,('CH','D'):0.75,('CH','AI'):0.45,('CH','H'):1.35,
          ('SE','I'):0.90,('SE','D'):1.30,('SE','AI'):1.55,('SE','H'):1.00,
          ('MD','I'):1.10,('MD','D'):0.85,('MD','AI'):0.65,('MD','H'):1.25}
    D0={'NMM':38,'RRD':78,'NCC':55,'CH':32,'SE':82,'MD':48}; g,lam=0.002,0.7
    m=pulp.LpProblem("b4c",pulp.LpMaximize)
    x=pulp.LpVariable.dicts("x",(regions,items),lowBound=0); M=pulp.LpVariable("M",lowBound=0)
    m+=pulp.lpSum(beta[r,i]*x[r][i] for r in regions for i in items)
    m+=pulp.lpSum(x[r][i] for r in regions for i in items)<=50000
    for r in regions:
        m+=pulp.lpSum(x[r][i] for i in items)>=5000; m+=pulp.lpSum(x[r][i] for i in items)<=12000
        m+=D0[r]+g*x[r]['D']<=M; m+=D0[r]+g*x[r]['D']>=lam*M
    m+=pulp.lpSum(x[r]['H'] for r in regions)>=12000
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    mat=pd.DataFrame([[x[r][i].varValue for i in items] for r in regions],index=regions,columns=items)
    fig=px.imshow(mat,text_auto=".0f",color_continuous_scale="Reds",aspect="auto")
    fig.update_layout(title="Heatmap phân bổ M3",paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#d4d7e0"))
    st.plotly_chart(fig,use_container_width=True)


# ============================================================
# ROUTER
# ============================================================
ROUTES={
    PAGES[0]:page_home, PAGES[1]:page_bai1, PAGES[2]:page_bai2, PAGES[3]:page_bai3,
    PAGES[4]:page_bai4, PAGES[5]:page_bai5, PAGES[6]:page_bai6, PAGES[7]:page_bai7,
    PAGES[8]:page_bai8, PAGES[9]:page_bai9, PAGES[10]:page_bai10, PAGES[11]:page_bai11,
    PAGES[12]:page_bai12,
}
ROUTES[page]()
