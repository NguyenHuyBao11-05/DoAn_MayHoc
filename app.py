"""
Steam Game Recommender — Bản kết nối DỮ LIỆU THẬT & KẾT QUẢ FINETUNE
"""
from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, List

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Cấu hình đường dẫn đến thư mục chứa các file bạn vừa tải về
MODEL_DIR = "models"

# ============================================================
# 📊 KẾT QUẢ FINETUNE THẬT CỦA NHÓM (khớp Bảng 3/4 trong báo cáo)
# ============================================================
MODEL_METRICS = pd.DataFrame([
    {"Mô hình": "NCF (After)",             "Accuracy": 0.6531, "Precision": 0.7536, "Recall": 0.4548, "F1-Score": 0.5673, "AUC-ROC": 0.7403},
    {"Mô hình": "NCF (Before)",            "Accuracy": 0.6813, "Precision": 0.6867, "Recall": 0.6669, "F1-Score": 0.6766, "AUC-ROC": 0.7483},
    {"Mô hình": "CNN 1D (After)",          "Accuracy": 0.5232, "Precision": 0.7673, "Recall": 0.0665, "F1-Score": 0.1223, "AUC-ROC": 0.6715},
    {"Mô hình": "CNN 1D (Before)",         "Accuracy": 0.6229, "Precision": 0.5994, "Recall": 0.7409, "F1-Score": 0.6627, "AUC-ROC": 0.6714},
    {"Mô hình": "MLP (After)",             "Accuracy": 0.5208, "Precision": 0.7739, "Recall": 0.0587, "F1-Score": 0.1092, "AUC-ROC": 0.6742},
    {"Mô hình": "MLP (Before)",            "Accuracy": 0.6237, "Precision": 0.6004, "Recall": 0.7393, "F1-Score": 0.6627, "AUC-ROC": 0.6735},
    {"Mô hình": "Random Forest",           "Accuracy": 0.6245, "Precision": 0.5987, "Recall": 0.7552, "F1-Score": 0.6679, "AUC-ROC": 0.6763},
    {"Mô hình": "Cosine Similarity (k-NN)", "Accuracy": 0.5690, "Precision": 0.5678, "Recall": 0.5746, "F1-Score": 0.5711, "AUC-ROC": 0.6013},
])

MODEL_EXPLANATIONS = {
    "NCF": (
        "**Neural Collaborative Filtering (NCF)** — Học một vector Embedding riêng cho từng User và từng Item, "
        "nối hai vector này rồi đưa qua mạng MLP để dự đoán xác suất tương tác. Đây là mô hình **duy nhất** "
        "không cần biết trước giờ chơi/giá/nền tảng — chỉ cần biết ai và game nào — nên là lựa chọn phù hợp "
        "nhất để gợi ý những game người dùng **chưa từng chơi**."
    ),
    "Random Forest": (
        "**Random Forest** — Tổ hợp 100 cây quyết định, huấn luyện trên 4 đặc trưng dạng bảng "
        "(`hours, price_original, mac, linux`). Vì đây là game người dùng **chưa chơi**, số giờ chơi thực tế "
        "chưa tồn tại nên được gán bằng 0 (giá trị đúng thực tế của một game chưa từng chơi) — mô hình vẫn "
        "chạy suy luận thật trên các trọng số đã huấn luyện, nhưng dự đoán có thể kém tin cậy hơn NCF vì phần "
        "lớn dữ liệu huấn luyện có giờ chơi > 0."
    ),
    "Cosine Similarity": (
        "**Cosine Similarity (k-NN, k=5)** — So khoảng cách cosine giữa đặc trưng của game ứng viên với 20.000 "
        "mẫu đã lưu lúc huấn luyện. Cùng hạn chế về giờ chơi = 0 như Random Forest, đồng thời để đảm bảo tốc "
        "độ phản hồi, ứng dụng chỉ lấy mẫu ngẫu nhiên một phần ứng viên trước khi tính khoảng cách."
    ),
    "MLP": (
        "**Multilayer Perceptron (MLP)** — Mạng nơ-ron 2 tầng ẩn trên cùng 4 đặc trưng dạng bảng như Random "
        "Forest (đã tinh chỉnh bằng KerasTuner). Cùng hạn chế giờ chơi = 0 khi gợi ý game chưa chơi — đây là lý "
        "do Recall của MLP sau tinh chỉnh rất thấp trên tập Test (Bảng 4), nên có thể ít game vượt ngưỡng hiển "
        "thị."
    ),
    "CNN 1D": (
        "**CNN 1D** — Áp một tầng Conv1D lên 4 đặc trưng dạng bảng để bắt tổ hợp cục bộ giữa các đặc trưng lân "
        "cận, cùng kiến trúc tuyến sau MLP. Chịu chung hạn chế giờ chơi = 0 khi gợi ý game chưa chơi như MLP."
    ),
}

# ============================================================
# ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU
# ============================================================
@dataclass
class Game:
    id: str
    title: str
    tags: List[str]
    price_usd: float
    rating: int
    win: bool = True
    mac: bool = False
    linux: bool = False
    app_idx: int = -1  # Chỉ số mã hóa để chống lag

@dataclass
class UserProfile:
    id: str
    played: List[Dict] = field(default_factory=list)
    bought: List[str] = field(default_factory=list)
    rated_count: int = 0

# ============================================================
# TẢI TÀI NGUYÊN THẬT TỪ THƯ MỤC MODELS
# ============================================================
@st.cache_resource
def load_keras_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    return load_model(path) if os.path.exists(path) else None

@st.cache_resource
def load_joblib_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_real_encoders():
    encoder_path = os.path.join(MODEL_DIR, 'encoders.pkl')
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_real_users():
    users_path = os.path.join(MODEL_DIR, 'real_users_sample.pkl')
    if os.path.exists(users_path):
        with open(users_path, 'rb') as f:
            return pickle.load(f)
    return {}

@st.cache_data
def load_real_games(_encoders):
    csv_path = os.path.join(MODEL_DIR, 'games_metadata.csv')
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    games_dict = {}
    app_encoder = _encoders['app_encoder'] if _encoders else None

    for _, row in df.iterrows():
        aid = str(int(row['app_id']))
        price = float(row['price_original']) if 'price_original' in row else 0.0
        rating = int(row['positive_ratio']) if 'positive_ratio' in row else 80
        win = bool(row['win']) if 'win' in row else True
        mac = bool(row['mac']) if 'mac' in row else False
        linux = bool(row['linux']) if 'linux' in row else False

        # Tiền mã hóa ID game sang số nguyên để truyền vào NCF siêu tốc
        app_idx = -1
        if app_encoder:
            try:
                app_idx = app_encoder.transform([int(aid)])[0]
            except Exception:
                pass

        games_dict[aid] = Game(
            id=aid, title=str(row['title']), tags=["Steam Game"],
            price_usd=price, rating=rating, win=win, mac=mac, linux=linux, app_idx=app_idx
        )
    return games_dict

# Khởi chạy nạp dữ liệu thật
st.set_page_config(page_title="Steam Recommender", page_icon="🎮", layout="wide")
encoders = load_real_encoders()
REAL_USERS = load_real_users()
GAMES = load_real_games(encoders)
ALL_GAMES = list(GAMES.values())

MODELS = {
    "NCF": load_keras_model('ncf_model_v1.keras'),
    "MLP": load_keras_model('mlp_model_v1.keras'),
    "CNN 1D": load_keras_model('cnn1d_model_v1.keras'),
    "Random Forest": load_joblib_model('rf_model.joblib'),
    "Cosine Similarity": load_joblib_model('knn_model.joblib'),
}
DEFAULT_THRESHOLDS = {"NCF": 0.63, "MLP": 0.70, "CNN 1D": 0.71, "Random Forest": 0.5, "Cosine Similarity": 0.5}
THRESHOLDS = (encoders or {}).get('thresholds', DEFAULT_THRESHOLDS)

def get_user_profile(user_id: str) -> UserProfile:
    if user_id in REAL_USERS:
        u_data = REAL_USERS[user_id]
        return UserProfile(id=user_id, played=u_data["played"], rated_count=u_data["rated_count"])

    # Nếu gõ ID lạ, tự sinh profile mô phỏng ngẫu nhiên để app không lỗi
    ids = list(GAMES.keys())
    if not ids: return UserProfile(user_id, [], [], 0)
    random.seed(user_id)
    played = [{"id": gid, "hours": random.randint(10, 150)} for gid in random.sample(ids, min(5, len(ids)))]
    return UserProfile(user_id, played, [], len(played))

# ============================================================
# ĐẶC TRƯNG DẠNG BẢNG CHO CÁC MÔ HÌNH KHÔNG DÙNG EMBEDDING
# ============================================================
def build_tab_features(games: List[Game]) -> np.ndarray:
    """Với game người dùng CHƯA chơi, hours=0 là giá trị đúng thực tế (chưa có giờ chơi nào)."""
    tabular_features = encoders['tabular_features']
    col_values = {
        'hours': lambda g: 0.0,
        'price_original': lambda g: g.price_usd,
        'mac': lambda g: int(g.mac),
        'linux': lambda g: int(g.linux),
        'win': lambda g: int(g.win),
        'steam_deck': lambda g: 0,
    }
    rows = [[col_values[f](g) for f in tabular_features] for g in games]
    X = np.array(rows, dtype=float)
    return encoders['scaler'].transform(X)

# ============================================================
# THUẬT TOÁN GỢI Ý VECTOR HÓA CHỐNG LAG
# ============================================================
def recommend(profile: UserProfile, model_name: str, os_filter: Dict[str, bool],
              max_price: float, free_only: bool, top_n: int = 12) -> List[Dict]:
    owned = {p["id"] for p in profile.played}

    def pass_filter(g: Game) -> bool:
        if free_only and g.price_usd > 0: return False
        if not free_only and g.price_usd > max_price: return False
        if any(os_filter.values()):
            if not ((os_filter["win"] and g.win) or (os_filter["mac"] and g.mac) or (os_filter["linux"] and g.linux)):
                return False
        return True

    candidates = [g for g in ALL_GAMES if g.id not in owned and pass_filter(g)]
    if not candidates:
        return []

    # k-NN là lazy learner (O(n*d) mỗi truy vấn) -> giới hạn số ứng viên để giữ tốc độ phản hồi (xem Mục 4.1)
    if model_name == "Cosine Similarity" and len(candidates) > 3000:
        random.seed(int(profile.id))
        candidates = random.sample(candidates, 3000)

    threshold = THRESHOLDS.get(model_name, 0.5)
    scored: List[tuple] = []

    if model_name == "NCF":
        ncf_model = MODELS.get("NCF")
        if ncf_model is None or encoders is None:
            return []
        try:
            user_idx = encoders['user_encoder'].transform([int(profile.id)])[0]
        except Exception:
            user_idx = 0
        valid_candidates = [g for g in candidates if g.app_idx != -1]
        if valid_candidates:
            user_input = np.full(len(valid_candidates), user_idx)
            item_input = np.array([g.app_idx for g in valid_candidates])
            probs = ncf_model.predict([user_input, item_input], verbose=0).flatten()
            scored = list(zip(valid_candidates, probs))

    else:
        model_obj = MODELS.get(model_name)
        if model_obj is None or encoders is None:
            return []
        X = build_tab_features(candidates)
        if model_name in ("Random Forest", "Cosine Similarity"):
            probs = model_obj.predict_proba(X)[:, 1]
        else:  # MLP, CNN 1D (Keras)
            probs = model_obj.predict(X, verbose=0).flatten()
        scored = list(zip(candidates, probs))

    # 1. Lưu tất cả các game qua ngưỡng Threshold
    passed_games = []
    for g, prob in scored:
        if prob >= threshold:
            passed_games.append({
                "id": g.id, "title": g.title, "tags": ", ".join(g.tags),
                "price": "Free" if g.price_usd == 0 else f"${g.price_usd:.2f}",
                "rating": g.rating, "match": round(float(prob) * 100, 1), "because": None,
                "platforms": " ".join([p for p, ok in [("Win", g.win), ("Mac", g.mac), ("Linux", g.linux)] if ok]),
            })

    # 2. Xử lý Popularity Bias bằng Re-ranking (Diversity)
    passed_games.sort(key=lambda x: x["match"], reverse=True)

    if len(passed_games) > top_n:
        # Trích xuất Top 60 game điểm cao nhất
        top_pool = passed_games[:60]
        # Dùng ID user làm seed để xáo trộn ngẫu nhiên (User khác nhau sẽ ra kết hợp game khác nhau)
        random.seed(int(profile.id))
        results = random.sample(top_pool, top_n)
        # Sắp xếp lại danh sách 12 game cuối cùng theo điểm Match
        results.sort(key=lambda x: x["match"], reverse=True)
    else:
        results = passed_games[:top_n]

    return results

def os_usage(profile: UserProfile) -> pd.DataFrame:
    win = mac = linux = 0
    for p in profile.played:
        g = GAMES.get(p["id"])
        if not g: continue
        if g.win:   win += p["hours"]
        if g.mac:   mac += p["hours"]
        if g.linux: linux += p["hours"]
    df = pd.DataFrame({"OS": ["Windows", "Mac", "Linux"], "Giờ chơi": [win, mac, linux]})
    return df[df["Giờ chơi"] > 0].set_index("OS")

# ============================================================
# GIAO DIỆN STREAMLIT PHẲNG MƯỢT
# ============================================================
st.title("🎮 Hệ thống Gợi ý Game trên Steam")
st.caption("Cả 5 mô hình (NCF, MLP, CNN 1D, Random Forest, Cosine Similarity) đều suy luận thật từ trọng số đã huấn luyện")

with st.sidebar:
    st.header("⚙️ Thanh công cụ")

    if "user_id" not in st.session_state:
        st.session_state.user_id = random.choice(list(REAL_USERS.keys())) if REAL_USERS else "76561198031234567"

    st.subheader("👤 Người dùng")
    user_id = st.text_input("User ID (SteamID64)", value=st.session_state.user_id)

    if st.button("🎲 Random User Thật", use_container_width=True):
        if REAL_USERS:
            st.session_state.user_id = random.choice(list(REAL_USERS.keys()))
            st.rerun()
    st.session_state.user_id = user_id

    st.divider()
    st.subheader("🤖 Mô hình dự đoán")
    model = st.selectbox(
        "Chọn mô hình",
        ["NCF", "MLP", "CNN 1D", "Random Forest", "Cosine Similarity"],
        index=0
    )
    if MODELS.get(model) is not None:
        st.success(f"✅ Đã tải mô hình {model} thật (ngưỡng phân loại: {THRESHOLDS.get(model, 0.5):.2f})")
        if model != "NCF":
            st.caption("⚠️ Game chưa chơi chưa có giờ chơi thật -> mô hình dùng hours=0 (giá trị đúng thực tế) "
                       "để suy luận, xem tab AI Explainability.")
    else:
        st.error(f"❌ Chưa tìm thấy file mô hình cho {model} trong thư mục models/.")

    st.divider()
    st.subheader("🖥️ Hệ điều hành")
    os_win, os_mac, os_linux = st.checkbox("Windows", True), st.checkbox("Mac"), st.checkbox("Linux")
    st.subheader("💰 Giá")
    free_only = st.checkbox("Chỉ game miễn phí")
    max_price = st.slider("Giá tối đa (USD)", 0, 60, 60, disabled=free_only)

profile = get_user_profile(st.session_state.user_id)
recs = recommend(profile, model, {"win": os_win, "mac": os_mac, "linux": os_linux}, float(max_price), free_only)

tab1, tab2, tab3 = st.tabs(["👤 Hồ sơ người dùng", "🎯 Kết quả gợi ý", "🧠 AI Explainability"])

with tab1:
    st.subheader(f"Hồ sơ: `{profile.id}`")
    if profile.id in REAL_USERS:
        st.success("✅ Trạng thái tài khoản: Tìm thấy trong tập dữ liệu huấn luyện (Data thật 100%)")
    else:
        st.warning("⚠️ ID lạ không nằm trong tập mẫu mẫu. Đang tự sinh hồ sơ mô phỏng an toàn.")

    total_hours = sum(p["hours"] for p in profile.played)
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng giờ chơi", f"{int(total_hours):,} giờ")
    c2.metric("Số game tương tác thật", len(profile.played))
    c3.metric("Độ tin cậy dữ liệu", "100% Real Data" if profile.id in REAL_USERS else "Demo Mode")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**🕹️ Danh sách game người này thực tế đã chơi:**")
        played_games_list = []
        for p in profile.played:
            if p["id"] in GAMES:
                played_games_list.append({"Tên Game (Dữ liệu gốc)": GAMES[p["id"]].title, "Thời gian chơi": f"{int(p['hours'])} giờ"})
        if played_games_list:
            st.dataframe(pd.DataFrame(played_games_list), hide_index=True, use_container_width=True)
        else:
            st.write("Không tìm thấy thông tin tên game tương ứng trong metadata.")
    with right:
        st.markdown("**💻 Phân bổ thời gian chơi theo Nền tảng**")
        if not os_usage(profile).empty:
            st.bar_chart(os_usage(profile))
        else:
            st.write("Chưa có dữ liệu hệ điều hành.")

with tab2:
    st.subheader(f"Gợi ý từ mô hình AI: {model}")
    if not recs:
        st.warning("⚠️ Không có game nào phù hợp bộ lọc (hoặc chưa tìm thấy file mô hình tương ứng).")
    else:
        cols = st.columns(3)
        for i, r in enumerate(recs):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"### {r['title']}")
                    a, b = st.columns(2)
                    a.metric("Độ phù hợp (Match)", f"{r['match']}%")
                    b.metric("Rating Thật", f"{r['rating']}% Tích cực")
                    st.write(f"💰 {r['price']}  ·  🖥️ {r['platforms']}")

with tab3:
    st.subheader("🧠 Cơ chế hoạt động của mô hình")
    st.markdown(MODEL_EXPLANATIONS.get(model, ""))
    st.divider()
    st.markdown("### 📊 So sánh các mô hình (Kết quả Finetune thật, khớp Bảng 3/4 báo cáo)")

    def _highlight_standouts(df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        color = "background-color: #c8e6c9"
        for col in ["Accuracy", "F1-Score", "Precision"]:
            styles.loc[df[col].idxmax(), col] = color
        # NCF vượt trội rõ rệt ở AUC-ROC so với các mô hình còn lại -> tô cả 2 dòng NCF (Before/After)
        for idx in df.index:
            if str(df.loc[idx, "Mô hình"]).startswith("NCF"):
                styles.loc[idx, "AUC-ROC"] = color
        return styles

    st.dataframe(
        MODEL_METRICS.style.apply(_highlight_standouts, axis=None),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Cả 5 mô hình đều được tải và chạy suy luận thật ở tab 'Kết quả gợi ý'. Với 4 mô hình còn lại ngoài "
        "NCF, do dùng chung đặc trưng giờ chơi (`hours`) vốn chỉ có ý nghĩa SAU khi đã chơi, ứng dụng gán "
        "hours=0 cho các game ứng viên (đúng với thực tế là chưa có giờ chơi nào) — đây là lý do NCF vẫn là "
        "mô hình được khuyến nghị triển khai chính thức cho bài toán gợi ý game chưa từng chơi."
    )
