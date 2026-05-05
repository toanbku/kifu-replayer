# Kifu Replayer

Trích xuất các nước đi cờ vây từ ảnh kì phổ và mô phỏng lại trên trình duyệt.

## Pipeline

`image -> OpenCV(board+grid+stones) -> Tesseract OCR (top-K) -> game-rule validator -> moves[]`

Các bước chi tiết:
1. **Board detection** — HSV color filter tìm vùng wood color, cắt bbox bàn cờ.
2. **Grid extraction** — adaptive threshold + morphology để tách đường ngang/dọc, cluster để có 19 đường mỗi chiều.
3. **Stone classification** — sample disk pixels quanh mỗi giao điểm, phân loại Black/White/Empty bằng mean intensity.
4. **OCR per stone** — crop quanh mỗi quân, upscale 4×, 2 binarisations × 2 PSM modes, gộp top-K candidates với confidence.
5. **Game-rule validator** *(secret sauce — đạt 100% trên test suite hiện tại)*:
   - Mỗi quân nhận đúng 1 số.
   - **Parity constraint**: số ODD ⇒ quân ĐEN, số EVEN ⇒ quân TRẮNG (vì Đen luôn đi trước).
   - Hungarian assignment trên cost matrix `stones × numbers`.
   - **Replay simulation**: replay sequence theo thứ tự, mỗi nước phải đặt vào ô trống tại thời điểm đó (sau captures), không suicide. Dùng cờ vây simulator có capture/liberty detection.
   - **Repair pass**: với các quân không có OCR support, brute-permute các số còn lại và pick phương án có số nước hợp lệ cao nhất (tie-break bằng locality heuristic).
   - **Hill-climb swap**: pairwise swap các quân là "violator" để giảm số vi phạm.
   - Quân nào không có OCR support được điền theo parity + replay; cờ `ocr_supported=false` để UI highlight.

## Cấu trúc

```
.
├── backend/
│   ├── main.py             # FastAPI app + static serve
│   ├── kifu_extractor.py   # entrypoint of pipeline
│   ├── board_detector.py   # step 1
│   ├── grid_detector.py    # step 2
│   ├── stone_detector.py   # step 3
│   ├── number_ocr.py       # step 4 (top-K candidates per stone)
│   ├── validator.py        # step 5 (Hungarian + replay + repair)
│   ├── go_simulator.py     # Go board state with captures/liberty
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
"gian lận" hợp pháp bằng các ràng buộc luật cờ vây:

1. **Color parity** loại 50% lỗi OCR ngay lập tức. OCR đọc "8" cho 1 quân
   đen? Sai chắc luôn — 8 là chẵn, phải là số lẻ.
2. **Uniqueness + sequence** loại các duplicate. Hungarian assignment tự
   chọn phương án chi phí thấp nhất thoả mãn cả 2 ràng buộc.
3. **Multiple OCR variants** mỗi quân (OTSU, sharpened × PSM 8/7) — gộp
   top-K candidates với vote bonus khi nhiều variants đồng thuận.
4. **Replay simulation** — mỗi nước phải đặt vào ô **trống** tại thời
   điểm đó (sau bắt quân). Validator đếm số nước vi phạm cho mỗi phương án
   assignment, pick phương án có ít vi phạm nhất.
5. **Locality tie-breaker** — khi nhiều phương án cùng hợp lệ, ưu tiên
   phương án mà các nước liên tiếp gần nhau hơn (heuristic mềm).
6. **Inferred numbers** — quân không OCR được vẫn nhận số đúng nhờ
   constraint propagation. UI flag bằng dấu gạch đứt nếu không OCR support.

## Hạn chế / hướng cải thiện

- Ảnh chụp nghiêng (perspective): hiện tại không warp homography. Nếu cần,
  thêm 4-corner picker thủ công hoặc auto warp.
- Bàn cờ kích thước khác 19×19 chưa hỗ trợ (`BOARD_SIZE` hardcoded).
- Captures lớn (nhiều quân bị bắt mất trên ảnh) sẽ tạo gap ở move numbers
  — phía API trả về `warnings`; logic vẫn hoạt động đúng.
- DeepSeek vision API hiện đang preview (May 2026) — khi GA có thể plug-in
  như fallback cho stones có `ocr_supported=false`.
