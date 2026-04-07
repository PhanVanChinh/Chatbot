# ChatBot Tiếng Việt — Hỗ Trợ Khách Hàng

## Cài đặt & Chạy

```bash
# Yêu cầu: Python 3.10+, numpy (không cần cài thêm gì)
pip install numpy

# Chạy chatbot
python chatbot.py
```

## Cách dùng
- Gõ câu hỏi bình thường, có dấu hoặc không dấu đều được
- Gõ `/debug <câu>` để xem điểm confidence từng intent
- Gõ `exit` hoặc `thoát` để thoát

## Cải tiến so với phiên bản cũ

| Vấn đề cũ | Giải pháp mới |
|-----------|---------------|
| Bag-of-Words thô | TF-IDF + Cosine similarity |
| Không chịu lỗi chính tả | Fuzzy matching (difflib) |
| Không hiểu viết tắt | Alias map: "5tr" → "5 triệu", "đt" → "điện thoại"... |
| Không bỏ dấu tiếng Việt | Unicode normalization tự động |
| Threshold cứng 0.5 | Adaptive: HIGH(0.55)/MEDIUM(0.30)/fallback |
| Không nhớ ngữ cảnh | Context queue (3 intent gần nhất) |
| Gợi ý khi không hiểu | Top-3 suggestions có nhãn tiếng Việt |
| Tag typo (" bye","iphonee") | Đã sửa toàn bộ intents.json |
| Intent overlap | Jaccard specificity tie-breaking |

## Kết quả test: 20/20 (100%)
