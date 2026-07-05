"""
Steam Game Recommender — Streamlit demo
Chạy: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
import streamlit as st

# ============================================================
# DỮ LIỆU MẪU
# ============================================================

@dataclass
class Game:
    id: str
    title: str
    tags: List[str]
    price_usd: float
    rating: int  # 0..100
    win: bool = True
    mac: bool = False
    linux: bool = False


GAMES: Dict[str, Game] = {
    "elden":     Game("elden",     "Elden Ring",         ["Souls-like", "RPG", "Open World"], 59.99, 96, True, False, False),
    "cyberpunk": Game("cyberpunk", "Cyberpunk 2077",     ["RPG", "Sci-Fi", "Open World"],     29.99, 88, True, False, False),
    "witcher":   Game("witcher",   "The Witcher 3",      ["RPG", "Story-rich", "Fantasy"],     9.99, 97, True, True, True),
    "hades":     Game("hades",     "Hades",              ["Roguelike", "Action", "Indie"],    24.99, 93, True, True, False),
    "hollow":    Game("hollow",    "Hollow Knight",      ["Metroidvania", "Indie", "Souls-like"], 10.49, 95, True, True, True),
    "stardew":   Game("stardew",   "Stardew Valley",     ["Farming", "Cozy", "Indie"],        14.99, 94, True, True, True),
    "bg3":       Game("bg3",       "Baldur's Gate 3",    ["RPG", "Turn-based", "Story-rich"], 59.99, 96, True, True, False),
    "darksouls": Game("darksouls", "Dark Souls III",     ["Souls-like", "Dark Fantasy", "RPG"], 35.99, 89, True, False, False),
    "celeste":   Game("celeste",   "Celeste",            ["Platformer", "Indie", "Story-rich"], 19.99, 92, True, True, True),
    "factorio":  Game("factorio",  "Factorio",           ["Simulation", "Automation", "Sandbox"], 35.00, 96, True, True, True),
    "rimworld":  Game("rimworld",  "RimWorld",           ["Simulation", "Colony Sim", "Sandbox"], 34.99, 94, True, True, True),
    "terraria":  Game("terraria",  "Terraria",           ["Sandbox", "2D", "Adventure"],       9.99, 95, True, True, True),
    "dota2":     Game("dota2",     "Dota 2",             ["MOBA", "Free", "Multiplayer"],      0.00, 87, True, True, True),
    "csgo":      Game("csgo",      "Counter-Strike 2",   ["FPS", "Free", "Multiplayer"],       0.00, 90, True, False, True),
    "portal2":   Game("portal2",   "Portal 2",           ["Puzzle", "Story-rich", "Co-op"],    9.99, 98, True, True, True),
}
ALL_GAMES = list(GAMES.values())


@dataclass
class UserProfile:
    id: str
    played: List[Dict] = field(default_factory=list)  # {id, hours}
    bought: List[str] = field(default_factory=list)
    rated_count: int = 0


MOCK_USERS: Dict[str, UserProfile] = {
    "76561198000000001": UserProfile(
        "76561198000000001",
        [{"id": "elden", "hours": 187}, {"id": "darksouls", "hours": 142},
         {"id": "witcher", "hours": 96}, {"id": "hades", "hours": 54},
         {"id": "hollow", "hours": 38}],
        ["witcher", "bg3"], 12),
    "76561198000000002": UserProfile(
        "76561198000000002",
        [{"id": "factorio", "hours": 320}, {"id": "rimworld", "hours": 210},
         {"id": "stardew", "hours": 88}, {"id": "terraria", "hours": 64}],
        ["rimworld"], 8),
    "76561198000000003": UserProfile(
        "76561198000000003",
        [{"id": "dota2", "hours": 1240}, {"id": "csgo", "hours": 890},
         {"id": "portal2", "hours": 42}],
        ["portal2"], 15),
}


def _hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def get_user_profile(user_id: str) -> UserProfile:
    if user_id in MOCK_USERS:
        return MOCK_USERS[user_id]
    h = _hash(user_id or "guest")
    ids = list(GAMES.keys())
    played, seen = [], set()
    for i in range(4):
        gid = ids[(h * (i + 1) * 7) % len(ids)]
        if gid in seen:
            continue
        seen.add(gid)
        played.append({"id": gid, "hours": 20 + (h * (i + 3)) % 250})
    return UserProfile(user_id, played, [ids[(h * 11) % len(ids)]], 3 + h % 10)


# ============================================================
# MÔ HÌNH GỢI Ý (mô phỏng)
# ============================================================

MODEL_METRICS = pd.DataFrame([
    {"Mô hình": "NCF (Neural CF)",     "F1-Score": 0.89, "Precision": 0.87, "Recall": 0.91},
    {"Mô hình": "Random Forest",       "F1-Score": 0.82, "Precision": 0.85, "Recall": 0.79},
    {"Mô hình": "Cosine Similarity",   "F1-Score": 0.74, "Precision": 0.71, "Recall": 0.78},
    {"Mô hình": "Popularity Baseline", "F1-Score": 0.61, "Precision": 0.58, "Recall": 0.65},
])


def recommend(profile: UserProfile, model: str, os_filter: Dict[str, bool],
              max_price: float, free_only: bool, top_n: int = 12) -> List[Dict]:
    owned = {p["id"] for p in profile.played} | set(profile.bought)
    user_tags = set()
    for p in profile.played:
        g = GAMES.get(p["id"])
        if g:
            user_tags.update(g.tags)

    def pass_filter(g: Game) -> bool:
        if free_only and g.price_usd > 0:
            return False
        if not free_only and g.price_usd > max_price:
            return False
        any_os = any(os_filter.values())
        if any_os:
            ok = (os_filter["win"] and g.win) or (os_filter["mac"] and g.mac) or (os_filter["linux"] and g.linux)
            if not ok:
                return False
        return True

    candidates = [g for g in ALL_GAMES if g.id not in owned and pass_filter(g)]
    seed = _hash(profile.id + ":" + model)
    results = []

    for idx, g in enumerate(candidates):
        tag_overlap = len(set(g.tags) & user_tags)
        rating_norm = g.rating / 100
        because = None
        if model == "NCF":
            score = 0.55 + 0.25 * rating_norm + 0.15 * min(tag_overlap / 3, 1)
            score += ((seed + idx * 17) % 100) / 1000
        elif model == "Random Forest":
            score = 0.40 + 0.30 * rating_norm + 0.20 * min(tag_overlap / 3, 1)
            score += ((seed + idx * 29) % 200) / 1000
        else:  # Cosine
            best_sim, best_src = 0.0, None
            for p in profile.played:
                src = GAMES.get(p["id"])
                if not src:
                    continue
                overlap = len(set(g.tags) & set(src.tags))
                sim = overlap / max(len(src.tags) + len(g.tags) - overlap, 1)
                if sim > best_sim:
                    best_sim, best_src = sim, src
            score = 0.30 + 0.60 * best_sim + 0.10 * rating_norm
            because = best_src.title if best_src else None

        results.append({
            "id": g.id, "title": g.title, "tags": ", ".join(g.tags),
            "price": "Free" if g.price_usd == 0 else f"${g.price_usd:.2f}",
            "rating": g.rating, "match": round(min(0.99, score) * 100, 1),
            "because": because,
            "platforms": " ".join([p for p, ok in [("Win", g.win), ("Mac", g.mac), ("Linux", g.linux)] if ok]),
        })

    results.sort(key=lambda r: r["match"], reverse=True)
    return results[:top_n]


def feature_importance(profile: UserProfile) -> pd.DataFrame:
    h = _hash(profile.id)
    base = [
        ("hours_played", 0.34), ("price_original", 0.22), ("genre_overlap", 0.18),
        ("rating_avg", 0.13), ("platform_match", 0.08), ("release_year", 0.05),
    ]
    rows = [{"feature": f, "importance": max(0.02, v + (((h + i * 13) % 10) - 5) / 200)}
            for i, (f, v) in enumerate(base)]
    return pd.DataFrame(rows).set_index("feature")


def os_usage(profile: UserProfile) -> pd.DataFrame:
    win = mac = linux = 0
    for p in profile.played:
        g = GAMES.get(p["id"])
        if not g:
            continue
        if g.win:   win += p["hours"]
        if g.mac:   mac += p["hours"]
        if g.linux: linux += p["hours"]
    df = pd.DataFrame({"OS": ["Windows", "Mac", "Linux"], "Giờ chơi": [win, mac, linux]})
    return df[df["Giờ chơi"] > 0].set_index("OS")


# ============================================================
# GIAO DIỆN STREAMLIT
# ============================================================

st.set_page_config(page_title="Steam Recommender", page_icon="🎮", layout="wide")

st.title("🎮 Hệ thống Gợi ý Game trên Steam")
st.caption("Demo Machine Learning — NCF · Random Forest · Cosine Similarity")

# -------- SIDEBAR --------
with st.sidebar:
    st.header("⚙️ Thanh công cụ")

    if "user_id" not in st.session_state:
        st.session_state.user_id = "76561198000000001"

    st.subheader("👤 Người dùng")
    user_id = st.text_input("User ID (SteamID64)", value=st.session_state.user_id)
    if st.button("🎲 Random User", use_container_width=True):
        st.session_state.user_id = random.choice(list(MOCK_USERS.keys()))
        st.rerun()
    st.session_state.user_id = user_id

    st.divider()
    st.subheader("🤖 Mô hình dự đoán")
    model = st.selectbox(
        "Chọn mô hình",
        ["NCF", "Random Forest", "Cosine Similarity"],
        index=0,
        help="NCF là mô hình tối ưu nhất (F1-Score = 0.89)",
    )
    if model == "NCF":
        st.success("✅ Mô hình tối ưu (mặc định)")

    st.divider()
    st.subheader("🖥️ Hệ điều hành")
    os_win   = st.checkbox("Windows", value=True)
    os_mac   = st.checkbox("Mac", value=False)
    os_linux = st.checkbox("Linux", value=False)

    st.subheader("💰 Giá")
    free_only = st.checkbox("Chỉ game miễn phí")
    max_price = st.slider("Giá tối đa (USD)", 0, 60, 60, disabled=free_only)

profile = get_user_profile(st.session_state.user_id)
recs = recommend(
    profile, model,
    {"win": os_win, "mac": os_mac, "linux": os_linux},
    float(max_price), free_only,
)

# -------- TABS --------
tab1, tab2, tab3 = st.tabs(["👤 Hồ sơ người dùng", "🎯 Kết quả gợi ý", "🧠 AI Explainability"])

with tab1:
    st.subheader(f"Hồ sơ: `{profile.id}`")
    total_hours = sum(p["hours"] for p in profile.played)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng giờ chơi", f"{total_hours:,}")
    c2.metric("Số game sở hữu", len(profile.played) + len(profile.bought))
    c3.metric("Đã đánh giá", profile.rated_count)
    c4.metric("Đã mua gần đây", len(profile.bought))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**🕹️ Top game đã chơi**")
        top = sorted(profile.played, key=lambda p: p["hours"], reverse=True)
        df_top = pd.DataFrame([
            {"Game": GAMES[p["id"]].title, "Giờ chơi": p["hours"]}
            for p in top if p["id"] in GAMES
        ])
        st.dataframe(df_top, hide_index=True, use_container_width=True)

    with right:
        st.markdown("**💻 Phân bổ giờ chơi theo OS**")
        st.bar_chart(os_usage(profile))

with tab2:
    st.subheader(f"Gợi ý bằng mô hình: {model}")
    if not recs:
        st.warning("Không có game phù hợp với bộ lọc hiện tại.")
    else:
        first = recs[0]
        if model == "Cosine Similarity" and first["because"]:
            st.info(f"💡 Vì bạn đã chơi **{first['because']}**, chúng tôi đề xuất **{first['title']}**.")
        else:
            played_titles = [GAMES[p['id']].title for p in profile.played[:2] if p['id'] in GAMES]
            if played_titles:
                st.info(f"💡 Vì bạn đã chơi {', '.join(f'**{t}**' for t in played_titles)}, "
                        f"chúng tôi đề xuất **{first['title']}** cùng {len(recs)-1} game khác.")

        cols = st.columns(3)
        for i, r in enumerate(recs):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"### {r['title']}")
                    st.caption(r["tags"])
                    a, b = st.columns(2)
                    a.metric("Match", f"{r['match']}%")
                    b.metric("Rating", f"{r['rating']}/100")
                    st.write(f"💰 {r['price']}  ·  🖥️ {r['platforms']}")
                    if r["because"]:
                        st.caption(f"↳ vì bạn đã chơi *{r['because']}*")

with tab3:
    st.subheader("🧠 Vì sao mô hình đưa ra kết quả này?")

    if model == "Random Forest":
        st.markdown("**Feature Importance** — mức độ ảnh hưởng của từng đặc trưng cho user này:")
        fi = feature_importance(profile)
        st.bar_chart(fi)
        top_feat = fi["importance"].idxmax()
        st.success(f"➡️ Với user `{profile.id}`, đặc trưng **{top_feat}** ảnh hưởng lớn nhất tới đề xuất.")
    elif model == "Cosine Similarity":
        st.markdown("**Content-based** — so sánh vector tags giữa game đã chơi và game ứng viên.")
        st.latex(r"\text{sim}(A, B) = \frac{|tags_A \cap tags_B|}{|tags_A \cup tags_B|}")
    else:
        st.markdown("**Neural Collaborative Filtering (NCF)** — học embedding user & item, "
                    "kết hợp qua MLP để dự đoán xác suất tương tác.")

    st.divider()
    st.markdown("### 📊 So sánh các mô hình (Lab 10)")
    st.dataframe(
        MODEL_METRICS.style.highlight_max(subset=["F1-Score", "Precision", "Recall"], color="#c8e6c9"),
        hide_index=True, use_container_width=True,
    )
    st.success("✅ **NCF** được chọn làm mô hình mặc định vì có **F1-Score = 0.89** — cao nhất so với các mô hình còn lại.")
