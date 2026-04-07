"""
ChatBot tiếng Việt nâng cấp
────────────────────────────
Cải tiến so với phiên bản cũ:
  1. TF-IDF + Cosine similarity thay BoW thô → hiểu ngữ nghĩa tốt hơn
  2. Fuzzy matching (difflib) → chịu lỗi chính tả
  3. Chuẩn hoá tiếng Việt: bỏ dấu, viết thường, alias ("tr" → "triệu")
  4. Context tracking → nhớ chủ đề đang nói
  5. Adaptive threshold → ít false-positive hơn
  6. Top-3 suggestions khi không chắc chắn
  7. Intent file sạch (bỏ tag typo " bye", "iphonee", v.v.)
"""

import json
import random
import re
import string
from difflib import SequenceMatcher
from collections import deque
import numpy as np


# ─── Bảng chuẩn hoá alias tiếng Việt ──────────────────────────────────────
ALIAS_MAP = {
    r"\btr\b":      "triệu",
    r"\bk\b":       "nghìn",
    r"\bđt\b":      "điện thoại",
    r"\bsp\b":      "sản phẩm",
    r"\bkm\b":      "khuyến mãi",
    r"\biph\b":     "iphone",
    r"\bss\b":      "samsung",
    r"\bxl\b":      "xiaomi",
    r"\bop\b":      "oppo",
    r"\bpk\b":      "phụ kiện",
    r"\bsh\b":      "ship",
    r"\bmn\b":      "màn hình",
}

# Map bỏ dấu tiếng Việt (build động qua unicodedata, luôn đúng độ dài)
import unicodedata as _ud

def _build_viet_map() -> dict:
    _chars = (
        "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
        "òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨ"
        "ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ"
    )
    m = {}
    for c in _chars:
        if c in "đĐ":
            m[ord(c)] = "d"
        else:
            base = _ud.normalize("NFD", c)[0]
            m[ord(c)] = base.lower()
    return m

_VIET_MAP = _build_viet_map()


def remove_accents(text: str) -> str:
    return text.translate(_VIET_MAP)


def normalize(text: str) -> str:
    """Chuẩn hoá: lower → bỏ dấu → alias → bỏ dấu câu thừa."""
    text = text.lower().strip()
    # thay alias
    for pattern, replacement in ALIAS_MAP.items():
        text = re.sub(pattern, replacement, text)
    # bỏ dấu tiếng Việt
    text = remove_accents(text)
    # bỏ ký tự đặc biệt, giữ chữ số và khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


# ─── TF-IDF tự cài (nhẹ, không cần sklearn) ────────────────────────────────
class TFIDFVectorizer:
    def __init__(self, max_features: int = 2000):
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])

    def fit(self, documents: list[str]):
        from math import log
        tokenized = [tokenize(d) for d in documents]
        # đếm DF
        df: dict[str, int] = {}
        for tokens in tokenized:
            for w in set(tokens):
                df[w] = df.get(w, 0) + 1

        # chọn từ theo DF
        n = len(documents)
        sorted_words = sorted(df.keys(), key=lambda w: -df[w])
        selected = sorted_words[:self.max_features]
        self.vocab = {w: i for i, w in enumerate(selected)}
        self.idf = np.array([
            log((n + 1) / (df[w] + 1)) + 1.0
            for w in selected
        ])

    def transform(self, documents: list[str]) -> np.ndarray:
        V = len(self.vocab)
        matrix = np.zeros((len(documents), V), dtype=np.float32)
        for i, doc in enumerate(documents):
            tokens = tokenize(doc)
            tf: dict[str, float] = {}
            for w in tokens:
                tf[w] = tf.get(w, 0) + 1
            total = max(len(tokens), 1)
            for w, cnt in tf.items():
                if w in self.vocab:
                    j = self.vocab[w]
                    matrix[i, j] = (cnt / total) * self.idf[j]
        return matrix

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)

    def transform_one(self, text: str) -> np.ndarray:
        return self.transform([text])[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def fuzzy_sim(a: str, b: str) -> float:
    """SequenceMatcher fuzzy score 0-1."""
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


# ─── ChatBot chính ──────────────────────────────────────────────────────────
class ChatBot:
    # ngưỡng tin tưởng
    THRESHOLD_HIGH   = 0.55   # trả lời tự tin
    THRESHOLD_MEDIUM = 0.30   # trả lời nhưng báo không chắc
    FUZZY_BOOST      = 0.15   # cộng thêm nếu fuzzy match khớp tốt
    CONTEXT_BOOST    = 0.10   # cộng thêm nếu khớp context

    def __init__(self, intents_path: str = "intents.json"):
        with open(intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.intents: list[dict] = data["intents"]
        self.vectorizer = TFIDFVectorizer(max_features=1500)
        self.context: deque = deque(maxlen=3)   # 3 intent gần nhất
        self._build_index()

    def _build_index(self):
        """Xây TF-IDF index từ tất cả patterns."""
        self.pattern_docs: list[str] = []
        self.pattern_tags: list[str] = []

        for intent in self.intents:
            tag = intent["tag"].strip()
            for pat in intent["patterns"]:
                self.pattern_docs.append(pat)
                self.pattern_tags.append(tag)

        self.X = self.vectorizer.fit_transform(self.pattern_docs)
        print(f"[✓] Đã index {len(self.pattern_docs)} patterns | "
              f"{len(self.intents)} intents | "
              f"{len(self.vectorizer.vocab)} từ vựng")

    # ── Predict ─────────────────────────────────────────────────────────────
    def _score_intents(self, query: str) -> list[tuple[str, float]]:
        """Tính điểm cho từng intent, trả về list (tag, score) sorted."""
        q_vec = self.vectorizer.transform_one(query)

        # cosine với từng pattern
        sims = np.array([cosine_sim(q_vec, self.X[i]) for i in range(len(self.X))])

        # token overlap: query vs pattern (Jaccard-style specificity)
        q_tokens = set(normalize(query).split())

        # tổng hợp theo tag: cosine + specificity bonus
        tag_scores: dict[str, float] = {}
        for i, tag in enumerate(self.pattern_tags):
            sim = sims[i]
            if sim > 0.5:
                pat_tokens = set(normalize(self.pattern_docs[i]).split())
                # Jaccard overlap giữa query và pattern
                intersect = len(q_tokens & pat_tokens)
                union = len(q_tokens | pat_tokens)
                jaccard = intersect / max(union, 1)
                # pattern càng specific (nhiều tokens khớp query) càng được bonus
                score = min(sim * 0.85 + jaccard * 0.15, 1.0)
            else:
                score = sim
            tag_scores[tag] = max(tag_scores.get(tag, 0.0), score)

        # specific-beats-general: model cụ thể + ý định mua → boost
        _buy_words = {"mua", "dat", "lay", "order", "muon"}
        _model_boost = {
            "iphone 7": "mua_iphone7",
            "iphone 8": "mua_iphone8",
            "iphone 15": "mua_iphone15",
        }
        q_norm = normalize(query)
        q_words = set(q_norm.split())
        has_buy = bool(q_words & _buy_words)
        for model_kw, specific_tag in _model_boost.items():
            if remove_accents(model_kw) in q_norm and specific_tag in tag_scores and has_buy:
                tag_scores[specific_tag] = min(tag_scores[specific_tag] + 0.45, 1.0)

        # fuzzy boost: so với từng pattern gốc
        for i, pat in enumerate(self.pattern_docs):
            fz = fuzzy_sim(query, pat)
            if fz > 0.70:
                tag = self.pattern_tags[i]
                tag_scores[tag] = min(
                    tag_scores.get(tag, 0.0) + self.FUZZY_BOOST * fz, 1.0
                )

        # context boost
        for tag in self.context:
            if tag in tag_scores:
                tag_scores[tag] = min(tag_scores[tag] + self.CONTEXT_BOOST, 1.0)

        return sorted(tag_scores.items(), key=lambda x: -x[1])

    def predict(self, query: str) -> tuple[str, float, list[tuple[str, float]]]:
        """Trả về (best_tag, confidence, top3_alternatives)."""
        ranked = self._score_intents(query)
        if not ranked:
            return "unknown", 0.0, []

        best_tag, best_score = ranked[0]

        # Tie-break: nếu top-2 gần bằng nhau (< 0.02),
        # ưu tiên tag có pattern khớp DÀI HƠN so với query
        if len(ranked) > 1:
            second_tag, second_score = ranked[1]
            if abs(best_score - second_score) < 0.02:
                q_len = len(normalize(query).split())
                def best_pat_len_diff(tag):
                    diffs = []
                    for i, t in enumerate(self.pattern_tags):
                        if t == tag:
                            plen = len(normalize(self.pattern_docs[i]).split())
                            diffs.append(abs(plen - q_len))
                    return min(diffs) if diffs else 999
                if best_pat_len_diff(second_tag) < best_pat_len_diff(best_tag):
                    best_tag = second_tag

        top3 = ranked[:3]
        return best_tag, best_score, top3

    # ── Response ─────────────────────────────────────────────────────────────
    def get_response(self, tag: str) -> str:
        for intent in self.intents:
            if intent["tag"].strip() == tag:
                return random.choice(intent["responses"])
        return ""

    def reply(self, user_input: str) -> str:
        tag, conf, top3 = self.predict(user_input)

        if conf >= self.THRESHOLD_HIGH:
            self.context.append(tag)
            return self.get_response(tag)

        if conf >= self.THRESHOLD_MEDIUM:
            self.context.append(tag)
            resp = self.get_response(tag)
            return f"{resp}\n_(Bạn có thể nói rõ hơn nếu tôi hiểu chưa đúng nhé!)_"

        # fallback với gợi ý
        hints = []
        for t, s in top3:
            if s > 0.10:
                label = self._tag_to_label(t)
                hints.append(f"• {label}")

        if hints:
            hint_str = "\n".join(hints)
            return (
                "Xin lỗi, tôi chưa chắc chắn về câu hỏi của bạn.\n"
                "Bạn có đang hỏi về:\n"
                f"{hint_str}\n"
                "Hãy gõ rõ hơn hoặc chọn gợi ý trên nhé!"
            )
        return (
            "Xin lỗi, tôi chưa hiểu câu hỏi. "
            "Bạn có thể hỏi về: sản phẩm, giá, bảo hành, giao hàng, "
            "khuyến mãi hoặc hỗ trợ kỹ thuật."
        )

    def _tag_to_label(self, tag: str) -> str:
        """Chuyển tag kỹ thuật sang tên dễ đọc."""
        _map = {
            "greetings":          "Chào hỏi",
            "bye":                "Tạm biệt",
            "product_inquiry":    "Hỏi về sản phẩm",
            "iphone_general":     "iPhone nói chung",
            "iphone_duoi5tr":     "iPhone dưới 5 triệu",
            "iphone7_info":       "Thông tin iPhone 7",
            "mua_iphone7":        "Mua iPhone 7",
            "iphone_5den10tr":    "iPhone 5-10 triệu",
            "mua_iphone8":        "Mua iPhone 8",
            "iphone_tren10tr":    "iPhone trên 10 triệu / iPhone 15",
            "mua_iphone15":       "Mua iPhone 15",
            "discounts":          "Khuyến mãi & giảm giá",
            "stock_availability": "Tồn kho sản phẩm",
            "refund_policy":      "Chính sách hoàn tiền",
            "order_status":       "Trạng thái đơn hàng",
            "store_location":     "Địa chỉ cửa hàng",
            "dia_chi_hanoi":      "Chi nhánh Hà Nội",
            "operating_hours":    "Giờ mở cửa",
            "shipping_policy":    "Chính sách giao hàng",
            "product_warranty":   "Bảo hành sản phẩm",
            "technical_support":  "Hỗ trợ kỹ thuật",
            "battery_life":       "Thời lượng pin",
            "compare_products":   "So sánh sản phẩm",
            "new_arrivals":       "Sản phẩm mới",
            "payment_methods":    "Phương thức thanh toán",
            "exchange_policy":    "Chính sách đổi trả",
            "dung_luong":         "Dung lượng bộ nhớ",
            "samsung_dungluong":  "Samsung dung lượng lớn",
            "trade_in":           "Thu cũ đổi mới",
            "camera_features":    "Tính năng camera",
            "repair_services":    "Dịch vụ sửa chữa",
            "product_recommendations": "Tư vấn điện thoại",
            "gaming_phone":       "Điện thoại chơi game",
            "emergency_contact":  "Hotline liên hệ",
            "screen_features":    "Thông số màn hình",
            "waterproof":         "Chống nước",
            "security_features":  "Bảo mật vân tay / khuôn mặt",
            "accessories":        "Phụ kiện",
            "feedback":           "Góp ý & phản hồi",
        }
        return _map.get(tag, tag)

    # ── CLI ──────────────────────────────────────────────────────────────────
    def chat(self):
        print("\n" + "═" * 55)
        print("  🤖  ChatBot hỗ trợ khách hàng – Cửa hàng điện thoại")
        print("  Gõ 'exit' hoặc 'thoát' để kết thúc")
        print("═" * 55 + "\n")

        while True:
            try:
                user = input("Bạn : ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBot : Tạm biệt!")
                break

            if not user:
                continue

            if user.lower() in ("exit", "quit", "thoát", "bye", "tạm biệt"):
                print("Bot : Tạm biệt! Chúc bạn một ngày tốt lành. 👋")
                break

            # debug mode: gõ /debug <câu> xem điểm
            if user.startswith("/debug "):
                query = user[7:]
                ranked = self._score_intents(query)
                print("\n── DEBUG top-5 ──")
                for tag, sc in ranked[:5]:
                    print(f"  {sc:.3f}  {tag}")
                print("─────────────────\n")
                continue

            response = self.reply(user)
            print(f"Bot : {response}\n")


# ─── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    path = "intents.json"
    if not os.path.exists(path):
        print(f"[!] Không tìm thấy {path}")
        exit(1)

    bot = ChatBot(intents_path=path)
    bot.chat()
