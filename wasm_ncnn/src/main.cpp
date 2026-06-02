// WASM entry point for the card identification browser demo.
//
// Mirrors mobile-license-plate-reader/wasm_ncnn/src/main.cpp but:
//   * blob names are in0/out0 (our pnnx export, not the reference's images/output0)
//   * the result buffer is a fixed-size struct array (see Detection below), so
//     later phases can fill OCR strings, symbol_id, border_id into the same
//     slot without changing the JS decoder.
//
// Currently wired: Phase 1 (YOLO) + Phase 2 (template classifier) +
// Phase 3 (OCR) + Phase 4 (symbol classifier + border color).
// Pending: Phase 5 (in-browser fuzzy match).

#include <emscripten.h>
#include <emscripten/bind.h>
#include <net.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "stb_image.h"
#include "stb_image_write.h"
#include "crop_regions.h"

constexpr int MAX_DETECTIONS = 8;
constexpr int MODEL_W = 320;
constexpr int MODEL_H = 320;
constexpr int TEMPLATE_INPUT = 320;
constexpr int N_TCGS = 3;

// (Baseline restored — no post-YOLO geometric mutations. The sleeve-aware
// YOLO outputs card-tight bboxes; we apply percent regions directly.)

// YOLO normalization (0..1).
const float YOLO_NORM[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};

// ImageNet normalization for template + symbol classifiers
// (both are MobileNetV3-Small pretrained). ncnn from_pixels yields 0..255, so
// mean = 255 * imnet_mean, norm = 1 / (255 * imnet_std).
const float IMAGENET_MEAN[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
const float IMAGENET_NORM[3] = {1.f / (0.229f * 255.f),
                                1.f / (0.224f * 255.f),
                                1.f / (0.225f * 255.f)};

static const char* TEMPLATE_PARAMS[N_TCGS] = {
    "assets/template_mtg.param",
    "assets/template_pokemon.param",
    "assets/template_yugioh.param",
};
static const char* TEMPLATE_BINS[N_TCGS] = {
    "assets/template_mtg.bin",
    "assets/template_pokemon.bin",
    "assets/template_yugioh.bin",
};

// Symbol classifier: same MobileNetV3-Small shape as template (320×320, ImageNet).
// Yugioh has no symbol classifier — that slot stays nullptr.
constexpr int SYMBOL_INPUT = 320;
static const char* SYMBOL_PARAMS[N_TCGS] = {
    "assets/symbol_mtg.param",
    "assets/symbol_pokemon.param",
    nullptr,
};
static const char* SYMBOL_BINS[N_TCGS] = {
    "assets/symbol_mtg.bin",
    "assets/symbol_pokemon.bin",
    nullptr,
};

// Border IDs: keep in sync with src/detector.js BORDER_NAMES.
constexpr int BORDER_UNKNOWN = -1;
constexpr int BORDER_BLACK = 0;
constexpr int BORDER_WHITE = 1;

// PaddleOCR rec model — height-48 input with one of two width buckets.
// Our PNNX export was traced with inputshape=[1,3,48,160] + inputshape2=[1,3,48,256].
constexpr int OCR_H = 48;
constexpr int OCR_W_NARROW = 160;
constexpr int OCR_W_WIDE = 256;
// PaddleOCR's standard preprocessing: (x - 127.5) / 127.5 → [-1, 1].
const float OCR_MEAN[3] = {127.5f, 127.5f, 127.5f};
const float OCR_NORM[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};

// Must match src/detector.js. 256 bytes/slot, MAX_DETECTIONS slots.
struct Detection {
    float conf;             // 4
    float bbox[4];          // 16  (x,y,w,h in source-image pixels)
    int32_t template_id;    // 4
    int32_t symbol_id;      // 4
    int32_t border_id;      // 4
    float template_conf;    // 4
    float symbol_conf;      // 4
    char name[64];          // 64
    char collector_id[32];  // 32
    char set_text[32];      // 32
    char _pad[88];          // 88
};
static_assert(sizeof(Detection) == 256, "Detection slot must be 256 bytes");

struct RectF { float x, y, w, h; };

static float iou(const RectF& a, const RectF& b) {
    float ax1 = a.x + a.w, ay1 = a.y + a.h;
    float bx1 = b.x + b.w, by1 = b.y + b.h;
    float ix0 = std::max(a.x, b.x), iy0 = std::max(a.y, b.y);
    float ix1 = std::min(ax1, bx1), iy1 = std::min(ay1, by1);
    float iw = std::max(0.f, ix1 - ix0), ih = std::max(0.f, iy1 - iy0);
    float inter = iw * ih;
    float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0 ? inter / uni : 0;
}

struct Box { float conf; RectF bbox; };

static void nms(std::vector<Box>& dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Box& a, const Box& b) { return a.conf > b.conf; });
    std::vector<Box> keep;
    for (auto& d : dets) {
        bool drop = false;
        for (auto& k : keep) {
            if (iou(d.bbox, k.bbox) > iou_thresh) { drop = true; break; }
        }
        if (!drop) keep.push_back(d);
    }
    dets.swap(keep);
}

// YOLOv8 single-class head: out shape [1, 5, N] → rows 0..3 = cx,cy,w,h, row 4 = conf.
static void decode_yolov8(const ncnn::Mat& out, float conf_thresh,
                          std::vector<Box>& dets) {
    dets.clear();
    int num = out.w;
    int rows = out.h;
    if (rows != 5) {
        fprintf(stderr, "Expected 5 rows in YOLO output, got %d\n", rows);
        return;
    }
    for (int i = 0; i < num; i++) {
        float cx = out.row(0)[i];
        float cy = out.row(1)[i];
        float w  = out.row(2)[i];
        float h  = out.row(3)[i];
        float c  = out.row(4)[i];
        if (c < conf_thresh) continue;
        dets.push_back({c, {cx - w * 0.5f, cy - h * 0.5f, w, h}});
    }
}

// Globals — single load, reused per frame.
ncnn::Net g_yolo;
static bool g_yolo_loaded = false;
ncnn::Net g_template[N_TCGS];
static bool g_template_loaded[N_TCGS] = {false, false, false};
ncnn::Net g_symbol[N_TCGS];
static bool g_symbol_loaded[N_TCGS] = {false, false, false};
ncnn::Net g_ocr;
static bool g_ocr_loaded = false;
static std::vector<std::string> g_ocr_keys;  // [0]="blank", [1..]=chars from dict.txt

static bool load_ocr_keys(const char* path) {
    g_ocr_keys.clear();
    g_ocr_keys.push_back("blank");  // CTC blank index 0
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) g_ocr_keys.push_back(line);
    }
    return true;
}

static std::string ctc_decode(const ncnn::Mat& out) {
    // out shape: (1, T, C). channel(0) gives the (T,C) slice.
    int T = out.h, C = out.w;
    const ncnn::Mat& mat = out.channel(0);
    std::string result;
    int last = 0;
    for (int t = 0; t < T; t++) {
        const float* row = mat.row(t);
        int best = 0;
        float best_v = row[0];
        for (int i = 1; i < C; i++) {
            if (row[i] > best_v) { best_v = row[i]; best = i; }
        }
        if (best != 0 && best != last && best < (int)g_ocr_keys.size()) {
            result += g_ocr_keys[best];
        }
        last = best;
    }
    return result;
}

static void softmax_argmax(const float* logits, int n, int& out_arg, float& out_conf);

static bool ensure_template(int tcg_id) {
    if (tcg_id < 0 || tcg_id >= N_TCGS) return false;
    if (g_template_loaded[tcg_id]) return true;
    g_template[tcg_id].opt.lightmode = true;
    g_template[tcg_id].opt.num_threads = 1;
    if (g_template[tcg_id].load_param(TEMPLATE_PARAMS[tcg_id]) ||
        g_template[tcg_id].load_model(TEMPLATE_BINS[tcg_id])) {
        fprintf(stderr, "Failed to load template classifier tcg=%d\n", tcg_id);
        return false;
    }
    g_template_loaded[tcg_id] = true;
    return true;
}

static bool ensure_symbol(int tcg_id) {
    if (tcg_id < 0 || tcg_id >= N_TCGS) return false;
    if (SYMBOL_PARAMS[tcg_id] == nullptr) return false;  // Yugioh has none.
    if (g_symbol_loaded[tcg_id]) return true;
    g_symbol[tcg_id].opt.lightmode = true;
    g_symbol[tcg_id].opt.num_threads = 1;
    if (g_symbol[tcg_id].load_param(SYMBOL_PARAMS[tcg_id]) ||
        g_symbol[tcg_id].load_model(SYMBOL_BINS[tcg_id])) {
        fprintf(stderr, "Failed to load symbol classifier tcg=%d\n", tcg_id);
        return false;
    }
    g_symbol_loaded[tcg_id] = true;
    return true;
}

// Sample a ~2% strip around each edge of the YOLO bbox in the source RGBA,
// compute mean Rec.601 luma, and bucket as black/white/unknown. Mirrors the
// `sample_border_color` helper in create_paddle_dataset/test_pipeline.py.
static int sample_border(uint8_t* rgba, int orig_w, int orig_h,
                         int x, int y, int w, int h) {
    int x0 = std::max(0, x);
    int y0 = std::max(0, y);
    int x1 = std::min(orig_w, x + w);
    int y1 = std::min(orig_h, y + h);
    if (x1 - x0 < 10 || y1 - y0 < 10) return BORDER_UNKNOWN;
    int strip = std::max(2, std::min(x1 - x0, y1 - y0) / 50);  // 2% of shorter side
    double sum = 0;
    int n = 0;
    auto add_rect = [&](int rx0, int ry0, int rx1, int ry1) {
        rx0 = std::max(0, rx0); ry0 = std::max(0, ry0);
        rx1 = std::min(orig_w, rx1); ry1 = std::min(orig_h, ry1);
        for (int py = ry0; py < ry1; py++) {
            const uint8_t* row = rgba + (py * orig_w + rx0) * 4;
            for (int px = rx0; px < rx1; px++) {
                float r = row[0], g = row[1], b = row[2];
                sum += 0.299f * r + 0.587f * g + 0.114f * b;
                n++;
                row += 4;
            }
        }
    };
    add_rect(x0, y0, x1, y0 + strip);          // top
    add_rect(x0, y1 - strip, x1, y1);          // bottom
    add_rect(x0, y0 + strip, x0 + strip, y1 - strip);  // left (no corners — already counted)
    add_rect(x1 - strip, y0 + strip, x1, y1 - strip);  // right
    if (n == 0) return BORDER_UNKNOWN;
    double mean_lum = sum / n;
    if (mean_lum < 60.0) return BORDER_BLACK;
    if (mean_lum > 180.0) return BORDER_WHITE;
    return BORDER_UNKNOWN;
}

// Crop a percent-region inside the YOLO bbox, resize to 320×320, run the
// symbol classifier, write argmax + softmax confidence into the slot.
static void run_symbol(int tcg_id, uint8_t* rgba, int orig_w, int orig_h,
                       int bx, int by, int bw, int bh,
                       const CropRegion& r, Detection& slot) {
    if (!ensure_symbol(tcg_id)) return;
    int rx = bx + (int)(r.x * bw);
    int ry = by + (int)(r.y * bh);
    int rw = (int)(r.w * bw);
    int rh = (int)(r.h * bh);
    rx = std::max(0, rx);
    ry = std::max(0, ry);
    if (rx + rw > orig_w) rw = orig_w - rx;
    if (ry + rh > orig_h) rh = orig_h - ry;
    if (rw < 4 || rh < 4) return;

    ncnn::Mat in = ncnn::Mat::from_pixels_roi_resize(
        rgba, ncnn::Mat::PIXEL_RGBA2RGB, orig_w, orig_h,
        rx, ry, rw, rh, SYMBOL_INPUT, SYMBOL_INPUT);
    in.substract_mean_normalize(IMAGENET_MEAN, IMAGENET_NORM);

    ncnn::Extractor ex = g_symbol[tcg_id].create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    int arg = -1;
    float prob = 0.f;
    int n = out.w * out.h * out.c;
    if (n > 0) softmax_argmax((const float*)out.data, n, arg, prob);
    slot.symbol_id = arg;
    slot.symbol_conf = prob;
}

// Softmax in-place over a flat vector. Returns the argmax and its softmax prob.
static void softmax_argmax(const float* logits, int n, int& out_arg, float& out_conf) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) sum += std::exp(logits[i] - maxv);
    int best = 0;
    float best_prob = 0;
    for (int i = 0; i < n; i++) {
        float p = std::exp(logits[i] - maxv) / sum;
        if (p > best_prob) { best_prob = p; best = i; }
    }
    out_arg = best;
    out_conf = best_prob;
}

extern "C" {

void models_init() {
    g_yolo.opt.lightmode = true;
    g_yolo.opt.num_threads = 1;
    if (g_yolo.load_param("assets/yolo_card.param") ||
        g_yolo.load_model("assets/yolo_card.bin")) {
        fprintf(stderr, "Failed to load YOLO card detector\n");
        return;
    }
    g_yolo_loaded = true;

    g_ocr.opt.lightmode = true;
    g_ocr.opt.num_threads = 1;
    if (g_ocr.load_param("assets/paddle_rec.param") ||
        g_ocr.load_model("assets/paddle_rec.bin")) {
        fprintf(stderr, "Failed to load PaddleOCR rec model\n");
        return;
    }
    if (!load_ocr_keys("assets/dict.txt")) {
        fprintf(stderr, "Failed to load OCR dict.txt\n");
        return;
    }
    g_ocr_loaded = true;
}

// Crop a percent-region inside the YOLO bbox, run OCR, write to `dest`.
// `dest_cap` is the slot capacity in bytes (including the null terminator).
static void run_ocr(uint8_t* rgba, int orig_w, int orig_h,
                    int bx, int by, int bw, int bh,
                    const CropRegion& r, char* dest, size_t dest_cap) {
    if (!g_ocr_loaded) return;
    // r is normalized 0..1 within the YOLO bbox.
    int rx = bx + (int)(r.x * bw);
    int ry = by + (int)(r.y * bh);
    int rw = (int)(r.w * bw);
    int rh = (int)(r.h * bh);
    rx = std::max(0, rx);
    ry = std::max(0, ry);
    if (rx + rw > orig_w) rw = orig_w - rx;
    if (ry + rh > orig_h) rh = orig_h - ry;
    if (rw < 4 || rh < 4) return;

    // Resize to height 48 with proportional width, bucketed to one of the two
    // shapes the PNNX export was traced with.
    float aspect = (float)rw / (float)rh;
    int target_w = (int)(OCR_H * aspect);
    int max_w = target_w > OCR_W_NARROW ? OCR_W_WIDE : OCR_W_NARROW;
    if (target_w > max_w) target_w = max_w;
    if (target_w < 8) target_w = 8;

    ncnn::Mat in = ncnn::Mat::from_pixels_roi_resize(
        rgba, ncnn::Mat::PIXEL_RGBA2RGB, orig_w, orig_h,
        rx, ry, rw, rh, target_w, OCR_H);
    in.substract_mean_normalize(OCR_MEAN, OCR_NORM);

    ncnn::Extractor ex = g_ocr.create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    std::string text = ctc_decode(out);
    if (text.empty()) return;
    std::strncpy(dest, text.c_str(), dest_cap - 1);
    dest[dest_cap - 1] = '\0';
}

// Run the per-TCG template classifier on a YOLO-cropped region and write the
// argmax label + softmax confidence into a Detection slot.
static void run_template(int tcg_id, uint8_t* rgba, int orig_w, int orig_h,
                         int x, int y, int w, int h, Detection& slot) {
    if (!ensure_template(tcg_id)) return;
    // Clamp ROI to source bounds — out-of-bounds reads at the edge cause GC.
    int x0 = std::max(0, x);
    int y0 = std::max(0, y);
    int x1 = std::min(orig_w, x + w);
    int y1 = std::min(orig_h, y + h);
    int rw = x1 - x0, rh = y1 - y0;
    if (rw < 4 || rh < 4) return;

    ncnn::Mat in = ncnn::Mat::from_pixels_roi_resize(
        rgba, ncnn::Mat::PIXEL_RGBA2RGB, orig_w, orig_h,
        x0, y0, rw, rh, TEMPLATE_INPUT, TEMPLATE_INPUT);
    in.substract_mean_normalize(IMAGENET_MEAN, IMAGENET_NORM);

    ncnn::Extractor ex = g_template[tcg_id].create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    int arg = -1;
    float prob = 0.f;
    int n = out.w * out.h * out.c;
    if (n > 0) softmax_argmax((const float*)out.data, n, arg, prob);
    slot.template_id = arg;
    slot.template_conf = prob;
}

// rgba: RGBA8 source image (orig_w * orig_h * 4 bytes).
// tcg_id: 0=mtg, 1=pokemon, 2=yugioh.
// result_buf: must be sizeof(Detection) * MAX_DETECTIONS = 2048 bytes.
void process_frame(uint8_t* rgba, int orig_w, int orig_h, int tcg_id,
                   float conf_thresh, uint8_t* result_buf) {
    memset(result_buf, 0, sizeof(Detection) * MAX_DETECTIONS);
    Detection* slots = reinterpret_cast<Detection*>(result_buf);
    for (int i = 0; i < MAX_DETECTIONS; i++) {
        slots[i].template_id = -1;
        slots[i].symbol_id = -1;
        slots[i].border_id = -1;
    }

    if (!g_yolo_loaded) return;

    // YOLO model was exported at fixed 320×320 input. Camera capture is
    // 640×640 for downstream-crop quality, so we resize down for YOLO here
    // and keep the full-res `rgba` for template / OCR / symbol crops below.
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
        rgba, ncnn::Mat::PIXEL_RGBA2BGR, orig_w, orig_h, MODEL_W, MODEL_H);
    input.substract_mean_normalize(0, YOLO_NORM);

    ncnn::Extractor ex = g_yolo.create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", input);
    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<Box> dets;
    decode_yolov8(out, conf_thresh, dets);
    nms(dets, 0.45f);

    float sx = (float)orig_w / (float)MODEL_W;
    float sy = (float)orig_h / (float)MODEL_H;
    int n = std::min((int)dets.size(), MAX_DETECTIONS);
    for (int i = 0; i < n; i++) {
        auto& b = dets[i].bbox;
        int x = (int)(b.x * sx);
        int y = (int)(b.y * sy);
        int w = (int)(b.w * sx);
        int h = (int)(b.h * sy);

        slots[i].conf = dets[i].conf;
        slots[i].bbox[0] = (float)x;
        slots[i].bbox[1] = (float)y;
        slots[i].bbox[2] = (float)w;
        slots[i].bbox[3] = (float)h;

        run_template(tcg_id, rgba, orig_w, orig_h, x, y, w, h, slots[i]);

        const TemplateCrops* tmpl = lookup_template(tcg_id, slots[i].template_id);
        if (tmpl) {
            if (tmpl->has_name)
                run_ocr(rgba, orig_w, orig_h, x, y, w, h, tmpl->name,
                        slots[i].name, sizeof(slots[i].name));
            if (tmpl->has_collector_id)
                run_ocr(rgba, orig_w, orig_h, x, y, w, h, tmpl->collector_id,
                        slots[i].collector_id, sizeof(slots[i].collector_id));
            if (tmpl->has_set_text)
                run_ocr(rgba, orig_w, orig_h, x, y, w, h, tmpl->set_text,
                        slots[i].set_text, sizeof(slots[i].set_text));
            if (tmpl->has_set_symbol)
                run_symbol(tcg_id, rgba, orig_w, orig_h, x, y, w, h,
                           tmpl->set_symbol, slots[i]);
        }
        slots[i].border_id = sample_border(rgba, orig_w, orig_h, x, y, w, h);
    }
}

// ---- Two-stage pipeline -------------------------------------------------
// Stage A runs YOLO + template + border on a small (downscaled) frame; the
// browser then crops the name/id/symbol regions from the FULL-resolution
// frame and calls the Stage-B helpers below on those small high-res crops.
// This keeps OCR/symbol input sharp without ever copying a full-res frame
// into the wasm heap every frame.

// Stage A: detect + classify only. Same as process_frame minus OCR/symbol.
void analyze_frame(uint8_t* rgba, int orig_w, int orig_h, int tcg_id,
                   float conf_thresh, uint8_t* result_buf) {
    memset(result_buf, 0, sizeof(Detection) * MAX_DETECTIONS);
    Detection* slots = reinterpret_cast<Detection*>(result_buf);
    for (int i = 0; i < MAX_DETECTIONS; i++) {
        slots[i].template_id = -1;
        slots[i].symbol_id = -1;
        slots[i].border_id = -1;
    }
    if (!g_yolo_loaded) return;

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
        rgba, ncnn::Mat::PIXEL_RGBA2BGR, orig_w, orig_h, MODEL_W, MODEL_H);
    input.substract_mean_normalize(0, YOLO_NORM);

    ncnn::Extractor ex = g_yolo.create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", input);
    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<Box> dets;
    decode_yolov8(out, conf_thresh, dets);
    nms(dets, 0.45f);

    float sx = (float)orig_w / (float)MODEL_W;
    float sy = (float)orig_h / (float)MODEL_H;
    int n = std::min((int)dets.size(), MAX_DETECTIONS);
    for (int i = 0; i < n; i++) {
        auto& b = dets[i].bbox;
        int x = (int)(b.x * sx);
        int y = (int)(b.y * sy);
        int w = (int)(b.w * sx);
        int h = (int)(b.h * sy);

        slots[i].conf = dets[i].conf;
        slots[i].bbox[0] = (float)x;
        slots[i].bbox[1] = (float)y;
        slots[i].bbox[2] = (float)w;
        slots[i].bbox[3] = (float)h;

        run_template(tcg_id, rgba, orig_w, orig_h, x, y, w, h, slots[i]);
        slots[i].border_id = sample_border(rgba, orig_w, orig_h, x, y, w, h);
    }
}

// Stage B: OCR a pre-cropped region. The whole rgba buffer IS the region
// (already cropped at high resolution browser-side). Writes UTF-8 into dest.
void ocr_region(uint8_t* rgba, int w, int h, char* dest, int dest_cap) {
    if (dest_cap > 0) dest[0] = '\0';
    if (!g_ocr_loaded || w < 4 || h < 4 || dest_cap < 2) return;

    float aspect = (float)w / (float)h;
    int target_w = (int)(OCR_H * aspect);
    int max_w = target_w > OCR_W_NARROW ? OCR_W_WIDE : OCR_W_NARROW;
    if (target_w > max_w) target_w = max_w;
    if (target_w < 8) target_w = 8;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        rgba, ncnn::Mat::PIXEL_RGBA2RGB, w, h, target_w, OCR_H);
    in.substract_mean_normalize(OCR_MEAN, OCR_NORM);

    ncnn::Extractor ex = g_ocr.create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    std::string text = ctc_decode(out);
    std::strncpy(dest, text.c_str(), dest_cap - 1);
    dest[dest_cap - 1] = '\0';
}

// Stage B: symbol-classify a pre-cropped region. out8 = [int32 id, float32 conf].
void symbol_region(int tcg_id, uint8_t* rgba, int w, int h, uint8_t* out8) {
    int32_t* id_out = reinterpret_cast<int32_t*>(out8);
    float* conf_out = reinterpret_cast<float*>(out8 + 4);
    *id_out = -1;
    *conf_out = 0.f;
    if (!ensure_symbol(tcg_id) || w < 4 || h < 4) return;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        rgba, ncnn::Mat::PIXEL_RGBA2RGB, w, h, SYMBOL_INPUT, SYMBOL_INPUT);
    in.substract_mean_normalize(IMAGENET_MEAN, IMAGENET_NORM);

    ncnn::Extractor ex = g_symbol[tcg_id].create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    int arg = -1;
    float prob = 0.f;
    int nn = out.w * out.h * out.c;
    if (nn > 0) softmax_argmax((const float*)out.data, nn, arg, prob);
    *id_out = arg;
    *conf_out = prob;
}

}  // extern "C"

EMSCRIPTEN_BINDINGS(card_browser) {
    emscripten::function("models_init", &models_init);
    emscripten::function("process_frame", &process_frame,
                         emscripten::allow_raw_pointer<uint8_t>());
}
