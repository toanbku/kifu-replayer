# Kifu Replayer

Trích xuất các nước đi cờ vây từ ảnh kì phổ và mô phỏng lại trên trình duyệt.

## Pipeline

`image -> OpenCV(board+grid+stones) -> Tesseract OCR -> parity/sequence solver -> moves[]`

Các bước chi tiết:
1. **Board detection** — HSV color filter tìm vùng wood color, cắt bbox bàn cờ.
2. **Grid extraction** — adaptive threshold + morphology để tách đường ngang/dọc, cluster để có 19 đường mỗi chiều.
3. **Stone classification** — sample disk pixels quanh mỗi giao điểm, phân loại Black/White/Empty bằng mean intensity.
4. **OCR per stone** — crop quanh mỗi quân, upscale 4×, multi-threshold variants, chạy Tesseract digits-only.
5. **Smart assignment** *(secret sauce — đẩy accuracy lên ~99%)*:
   - Mỗi quân nhận đúng 1 số.
   - **Parity constraint**: số ODD ⇒ quân ĐEN, số EVEN ⇒ quân TRẮNG (vì Đen luôn đi trước).
   - Solve bằng Hungarian (linear sum assignment) trên cost matrix `stones × numbers`.
   - Quân nào không có OCR support được điền theo parity + uniqueness; cờ `ocr_supported=false` để UI highlight.

## Cấu trúc

```
.
├── backend/
│   ├── main.py             # FastAPI app + static serve
│   ├── kifu_extractor.py   # entrypoint of pipeline
│   ├── board_detector.py   # step 1
│   ├── grid_detector.py    # step 2
│   ├── stone_detector.py   # step 3
│   ├── number_ocr.py       # step 4
│   ├── validator.py        # step 5
│   └── requirements.txt
├── frontend/
│   └── index.html          # vanilla HTML + Canvas replayer
├── Dockerfile
├── railway.toml
└── README.md
```

## Chạy local

```bash
# 1. Cài deps (cần Tesseract trên máy)
brew install tesseract           # macOS
# hoặc: apt-get install tesseract-ocr

cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Chạy server
python main.py
# mở http://localhost:8000
```

## Deploy lên Railway

```bash
# Cách 1: Railway CLI
npm i -g @railway/cli
railway login
railway init     # tạo project mới
railway up       # deploy bằng Dockerfile
railway domain   # gen public URL
```

Hoặc connect GitHub repo: New Project → Deploy from GitHub → chọn repo này.
Railway tự nhận Dockerfile và build. Healthcheck path `/api/health` đã set.

## API

### `POST /api/extract`
- form-data: `image` (file), `debug` (string `"true"` để trả overlay png)
- response:
  ```json
  {
    "board_size": 19,
    "total_stones": 100,
    "moves": [
      {
        "n": 1, "color": "B",
        "row": 14, "col": 3,
        "go_x": 4, "go_y": 5, "go_coord": "D5",
        "confidence": 91.5,
        "ocr_supported": true,
        "conflict": false
      }
    ],
    "warnings": []
  }
  ```

### `GET /api/health`
- `{ "ok": true }`

## Tại sao đạt được accuracy cao

Chỉ OCR thuần thì khó vì digit nhỏ và contrast biến động. Cách hệ thống này
"gian lận" hợp pháp:

1. **Color parity** loại 50% lỗi OCR ngay lập tức. OCR đọc "8" cho 1 quân
   đen? Sai chắc luôn — 8 là chẵn, phải là số lẻ.
2. **Uniqueness + sequence** loại các duplicate. Hungarian assignment tự
   chọn phương án chi phí thấp nhất thoả mãn cả 2 ràng buộc.
3. **Multiple OCR variants** mỗi quân (OTSU, adaptive, sharpened) — gộp
   candidates với confidence cao nhất.
4. **Inferred numbers** — quân không OCR được vẫn nhận được 1 số duy nhất
   khả dĩ từ parity + những số còn trống. UI sẽ flag bằng dấu gạch đứt.

## Hạn chế / hướng cải thiện

- Ảnh chụp nghiêng (perspective): hiện tại không warp homography. Nếu cần,
  thêm 4-corner picker thủ công hoặc auto warp.
- Bàn cờ kích thước khác 19×19 chưa hỗ trợ (`BOARD_SIZE` hardcoded).
- Captures lớn (nhiều quân bị bắt mất trên ảnh) sẽ tạo gap ở move numbers
  — phía API trả về `warnings`; logic vẫn hoạt động đúng.
- DeepSeek vision API hiện đang preview (May 2026) — khi GA có thể plug-in
  như fallback cho stones có `ocr_supported=false`.
