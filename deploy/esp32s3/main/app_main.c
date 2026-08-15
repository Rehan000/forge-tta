// ESP32-S3 firmware: forward-only int8 TTA (channel-recalib) on ResNet-20.
//
// Measures the on-device latency / energy / memory reported in the paper.
// Convolutions use the ESP-NN SIMD int8
// kernels (esp_nn_conv_s8); per-channel recalib / ReLU / residual stay in FPU float
// between convs, so adaptation is unchanged. The conv output is requantized to int8 at
// a per-conv out_scale (export_model.py), the host-validated esp_nn-scheme.
//
// GOLDEN CHECK: with adapt=0 the firmware must predict the same class as golden_logits
// (regenerate with `export_test_image.py --espnn`) before trusting any timing/energy.
//
// Build: cd deploy/esp32s3 && idf.py set-target esp32s3 && idf.py build flash monitor

#include <math.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "driver/gpio.h"

#include "esp_nn.h"         // Espressif SIMD int8 NN kernels (esp_nn_conv_s8 -> S3 assembly)
#include "esp_nn_ansi_headers.h"   // reference C kernels (for A/B correctness diagnostics)
// Toggle: ESPNN_CONV = esp_nn_conv_s8 (S3 SIMD) or esp_nn_conv_s8_ansi (reference C).
#define ESPNN_CONV esp_nn_conv_s8
#include "model_data.h"     // weights/scales/recalib targets (export_model.py)
#include "test_image.h"     // one sample + golden_logits (export_test_image.py)

#define TAG "tta"
#define PWR_TRIGGER_GPIO   4   // HIGH around the ADAPT updates -> power profiler
#define INFER_TRIGGER_GPIO 5   // HIGH around the full inference

#define MOM 0.01f              // recalib EMA momentum (must match the export)
#define EPS 1e-5f
#define MAXACT (16 * 32 * 32)  // largest activation tensor (layer1: 16ch x 32 x 32)

// ---- activation buffers (internal SRAM; 3*64KB float) ----
static float B0[MAXACT], B1[MAXACT], B2[MAXACT];

// ---- recalib online EMA state, persistent across the stream ----
#define N_RC 21
#define MAX_CH 64
static float g_rm[N_RC][MAX_CH], g_rv[N_RC][MAX_CH];

// recalib target tables, in forward order (site -> arrays + channel count)
static const float *const RC_MEAN[N_RC] = {
    m_bn1_target_mean,
    m_layer1_0_bn1_target_mean, m_layer1_0_bn2_target_mean,
    m_layer1_1_bn1_target_mean, m_layer1_1_bn2_target_mean,
    m_layer1_2_bn1_target_mean, m_layer1_2_bn2_target_mean,
    m_layer2_0_bn1_target_mean, m_layer2_0_bn2_target_mean, m_layer2_0_shortcut_1_target_mean,
    m_layer2_1_bn1_target_mean, m_layer2_1_bn2_target_mean,
    m_layer2_2_bn1_target_mean, m_layer2_2_bn2_target_mean,
    m_layer3_0_bn1_target_mean, m_layer3_0_bn2_target_mean, m_layer3_0_shortcut_1_target_mean,
    m_layer3_1_bn1_target_mean, m_layer3_1_bn2_target_mean,
    m_layer3_2_bn1_target_mean, m_layer3_2_bn2_target_mean,
};
static const float *const RC_STD[N_RC] = {
    m_bn1_target_std,
    m_layer1_0_bn1_target_std, m_layer1_0_bn2_target_std,
    m_layer1_1_bn1_target_std, m_layer1_1_bn2_target_std,
    m_layer1_2_bn1_target_std, m_layer1_2_bn2_target_std,
    m_layer2_0_bn1_target_std, m_layer2_0_bn2_target_std, m_layer2_0_shortcut_1_target_std,
    m_layer2_1_bn1_target_std, m_layer2_1_bn2_target_std,
    m_layer2_2_bn1_target_std, m_layer2_2_bn2_target_std,
    m_layer3_0_bn1_target_std, m_layer3_0_bn2_target_std, m_layer3_0_shortcut_1_target_std,
    m_layer3_1_bn1_target_std, m_layer3_1_bn2_target_std,
    m_layer3_2_bn1_target_std, m_layer3_2_bn2_target_std,
};
static const int RC_CH[N_RC] = {16, 16, 16, 16, 16, 16, 16,
                                32, 32, 32, 32, 32, 32, 32,
                                64, 64, 64, 64, 64, 64, 64};

static void recalib_reset(void) {
    for (int s = 0; s < N_RC; s++)
        for (int c = 0; c < RC_CH[s]; c++) {
            g_rm[s][c] = RC_MEAN[s][c];
            g_rv[s][c] = RC_STD[s][c] * RC_STD[s][c];
        }
}

static inline int8_t q8(float x, float scale) {
    float q = roundf(x / scale);
    if (q > 127.f) q = 127.f;
    if (q < -127.f) q = -127.f;
    return (int8_t)q;
}

// ESP-NN buffers: quantized conv input / output in HWC (channels-last), plus a
// grow-on-demand scratch buffer. The S3 SIMD kernels use aligned vector loads on these,
// so the buffers must be 16-byte aligned (scratch via heap_caps_aligned_alloc).
static int8_t g_in_hwc[MAXACT]  __attribute__((aligned(16)));
static int8_t g_out_hwc[MAXACT] __attribute__((aligned(16)));
static void  *g_scratch = NULL;
static int    g_scratch_sz = 0;

// effective scale -> esp_nn fixed-point (mult Q31, shift): value*(mult/2^31)*2^shift.
static void quant_mult(float eff, int32_t *mult, int32_t *shift) {
    if (eff <= 0.f) { *mult = 0; *shift = 0; return; }
    int e;
    double m = frexp((double)eff, &e);                 // eff = m * 2^e, m in [0.5,1)
    int64_t q = (int64_t)llround(m * 2147483648.0);    // m * 2^31
    if (q >= 2147483648LL) { q = 1073741824; e += 1; }
    *mult = (int32_t)q;
    *shift = e;
}

// int8 conv via ESP-NN SIMD kernels. float CHW in -> float CHW out: quantize the input at
// in_scale into HWC int8, run esp_nn_conv_s8 (per-output-channel requant to out_scale +
// int32 folded bias, NO activation clamp so ReLU/recalib stay in float afterwards), then
// dequant the HWC int8 output back to CHW float. Numerically this is the host-validated
// esp_nn-scheme: the conv output is snapped to the int8 grid at out_scale.
static void conv(const int8_t *wq_ohwi, const float *wscale, float in_scale, const float *bias,
                 const float *in, int Cin, int H, int W,
                 float *out, int Co, int k, int stride, int pad, float out_scale) {
    const int Ho = (H + 2 * pad - k) / stride + 1;
    const int Wo = (W + 2 * pad - k) / stride + 1;

    // quantize + transpose CHW float -> HWC int8 (esp_nn pads internally)
    for (int ci = 0; ci < Cin; ci++) {
        const float *src = in + ci * H * W;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                g_in_hwc[(y * W + x) * Cin + ci] = q8(src[y * W + x], in_scale);
    }

    // per-output-channel requant multiplier/shift + int32 folded bias
    static int32_t mult[MAX_CH], shift[MAX_CH], bi[MAX_CH];
    for (int co = 0; co < Co; co++) {
        const float ws = in_scale * wscale[co];
        quant_mult(ws / out_scale, &mult[co], &shift[co]);
        bi[co] = (int32_t)llroundf(bias[co] / ws);
    }

    data_dims_t id = {.width = W, .height = H, .channels = Cin, .extra = 0};
    data_dims_t fd = {.width = k, .height = k, .channels = Cin, .extra = 0};
    data_dims_t od = {.width = Wo, .height = Ho, .channels = Co, .extra = 0};
    conv_params_t cp = {.in_offset = 0, .out_offset = 0, .stride = {stride, stride},
                        .padding = {pad, pad}, .dilation = {1, 1}, .activation = {-128, 127}};
    quant_data_t qd = {.shift = shift, .mult = mult};

    int sz = esp_nn_get_conv_scratch_size(&id, &fd, &od, &cp);
    if (sz > g_scratch_sz) {
        free(g_scratch);
        g_scratch = heap_caps_aligned_alloc(16, sz, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        g_scratch_sz = sz;
    }
    esp_nn_set_conv_scratch_buf(g_scratch);
    ESPNN_CONV(&id, g_in_hwc, &fd, (int8_t *)wq_ohwi, bi, &od, g_out_hwc, &cp, &qd);

    // dequant + transpose HWC int8 -> CHW float
    for (int co = 0; co < Co; co++)
        for (int oy = 0; oy < Ho; oy++)
            for (int ox = 0; ox < Wo; ox++)
                out[(co * Ho + oy) * Wo + ox] = g_out_hwc[(oy * Wo + ox) * Co + co] * out_scale;
}

static void recalib(int site, float *x, int C, int HW, int adapt) {
    if (!adapt) return;
    const float *tm = RC_MEAN[site], *ts = RC_STD[site];
    float *rm = g_rm[site], *rv = g_rv[site];
    for (int c = 0; c < C; c++) {
        float *xc = x + c * HW;
        float mean = 0.f;
        for (int i = 0; i < HW; i++) mean += xc[i];
        mean /= HW;
        float var = 0.f;
        for (int i = 0; i < HW; i++) { float d = xc[i] - mean; var += d * d; }
        var /= HW;
        rm[c] = (1 - MOM) * rm[c] + MOM * mean;
        rv[c] = (1 - MOM) * rv[c] + MOM * var;
        float inv = ts[c] / sqrtf(rv[c] + EPS);
        for (int i = 0; i < HW; i++) xc[i] = (xc[i] - rm[c]) * inv + tm[c];
    }
}

static inline void relu(float *x, int n) { for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0; }

// one basic block. in/out/tmp are distinct buffers. cv* = conv1, ct* = conv2, sc* = shortcut.
typedef struct { const int8_t *w; const float *ws; float is; const float *b; int rc; float os; } Lyr;

static void block(const float *in, float *out, float *tmp,
                  int Cin, int Hin, int Cout, int Hout, int stride, int downsample,
                  Lyr c1, Lyr c2, Lyr sc, int adapt) {
    conv(c1.w, c1.ws, c1.is, c1.b, in, Cin, Hin, Hin, tmp, Cout, 3, stride, 1, c1.os);
    recalib(c1.rc, tmp, Cout, Hout * Hout, adapt);
    relu(tmp, Cout * Hout * Hout);
    conv(c2.w, c2.ws, c2.is, c2.b, tmp, Cout, Hout, Hout, out, Cout, 3, 1, 1, c2.os);
    recalib(c2.rc, out, Cout, Hout * Hout, adapt);
    if (downsample) {
        conv(sc.w, sc.ws, sc.is, sc.b, in, Cin, Hin, Hin, tmp, Cout, 1, stride, 0, sc.os);
        recalib(sc.rc, tmp, Cout, Hout * Hout, adapt);
        for (int i = 0; i < Cout * Hout * Hout; i++) out[i] += tmp[i];
    } else {
        for (int i = 0; i < Cout * Hout * Hout; i++) out[i] += in[i];
    }
    relu(out, Cout * Hout * Hout);
}

#define L(name, rc) (Lyr){ m_##name##_wq_ohwi, m_##name##_wscale, m_##name##_in_scale[0], \
                           m_##name##_bias, rc, m_##name##_out_scale[0] }
#define NL (Lyr){ 0, 0, 0, 0, 0, 0 }

static void forward(const float *image, float *logits10, int adapt) {
    // stem: conv1 (3->16, 3x3, s1, p1) -> B0
    conv(m_conv1_wq_ohwi, m_conv1_wscale, m_conv1_in_scale[0], m_conv1_bias, image, 3, 32, 32, B0, 16, 3, 1, 1, m_conv1_out_scale[0]);
    recalib(0, B0, 16, 32 * 32, adapt);
    relu(B0, 16 * 32 * 32);

    float *in = B0, *out = B1, *tmp = B2;
    // layer1: 3 blocks, 16ch 32x32, no downsample
    block(in, out, tmp, 16, 32, 16, 32, 1, 0, L(layer1_0_conv1, 1), L(layer1_0_conv2, 2), NL, adapt);
    in = B1; out = B0;
    block(in, out, tmp, 16, 32, 16, 32, 1, 0, L(layer1_1_conv1, 3), L(layer1_1_conv2, 4), NL, adapt);
    in = B0; out = B1;
    block(in, out, tmp, 16, 32, 16, 32, 1, 0, L(layer1_2_conv1, 5), L(layer1_2_conv2, 6), NL, adapt);
    // layer2: block0 downsample (16->32, s2), then 2 blocks 32ch 16x16
    in = B1; out = B0;
    block(in, out, tmp, 16, 32, 32, 16, 2, 1, L(layer2_0_conv1, 7), L(layer2_0_conv2, 8), L(layer2_0_shortcut_0, 9), adapt);
    in = B0; out = B1;
    block(in, out, tmp, 32, 16, 32, 16, 1, 0, L(layer2_1_conv1, 10), L(layer2_1_conv2, 11), NL, adapt);
    in = B1; out = B0;
    block(in, out, tmp, 32, 16, 32, 16, 1, 0, L(layer2_2_conv1, 12), L(layer2_2_conv2, 13), NL, adapt);
    // layer3: block0 downsample (32->64, s2), then 2 blocks 64ch 8x8
    in = B0; out = B1;
    block(in, out, tmp, 32, 16, 64, 8, 2, 1, L(layer3_0_conv1, 14), L(layer3_0_conv2, 15), L(layer3_0_shortcut_0, 16), adapt);
    in = B1; out = B0;
    block(in, out, tmp, 64, 8, 64, 8, 1, 0, L(layer3_1_conv1, 17), L(layer3_1_conv2, 18), NL, adapt);
    in = B0; out = B1;
    block(in, out, tmp, 64, 8, 64, 8, 1, 0, L(layer3_2_conv1, 19), L(layer3_2_conv2, 20), NL, adapt);

    // global avg pool over 8x8 -> vec[64]   (out == B1)
    float vec[64];
    for (int c = 0; c < 64; c++) {
        float s = 0.f;
        for (int i = 0; i < 64; i++) s += B1[c * 64 + i];
        vec[c] = s / 64.f;
    }
    // final linear (10x64), input quantized
    const float is = m_linear_in_scale[0];
    int8_t qv[64];
    for (int c = 0; c < 64; c++) qv[c] = q8(vec[c], is);
    for (int o = 0; o < 10; o++) {
        int32_t acc = 0;
        for (int c = 0; c < 64; c++) acc += (int32_t)qv[c] * (int32_t)m_linear_wq[o * 64 + c];
        logits10[o] = acc * (is * m_linear_wscale[o]) + m_linear_bias[o];
    }
}

// ---------------------------------------------------------------------------
static void trigger_init(void) {
    gpio_config_t io = { .pin_bit_mask = (1ULL << PWR_TRIGGER_GPIO) | (1ULL << INFER_TRIGGER_GPIO),
                         .mode = GPIO_MODE_OUTPUT };
    gpio_config(&io);
    gpio_set_level(PWR_TRIGGER_GPIO, 0);
    gpio_set_level(INFER_TRIGGER_GPIO, 0);
}

void app_main(void) {
    trigger_init();
    recalib_reset();
    size_t heap0 = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "free heap at start: %u", (unsigned)heap0);

    float logits[10];
    forward(test_image, logits, 0);                 // warm up + golden check

    // ESP-NN uses fixed-point requant, so logits differ sub-quantization-step from the
    // float golden (regenerate test_image.h with `export_test_image.py --espnn`). The
    // check is therefore: same predicted class as the golden, with a small logit margin.
    float maxerr = 0.f;
    int pred = 0, gpred = 0;
    for (int i = 0; i < 10; i++) {
        float e = fabsf(logits[i] - golden_logits[i]);
        if (e > maxerr) maxerr = e;
        if (logits[i] > logits[pred]) pred = i;
        if (golden_logits[i] > golden_logits[gpred]) gpred = i;
    }
    ESP_LOGI(TAG, "GOLDEN CHECK: pred=%d golden=%d (true=%d) max|logit-golden|=%.4f  [%s]",
             pred, gpred, test_label, maxerr,
             (pred == gpred && maxerr < 0.5f) ? "PASS" : "FAIL");

    const int ITERS = 5;   // naive conv is slow (~seconds/inference); 5 is enough to average
    int64_t t_inf = 0, t_adapt = 0;
    for (int i = 0; i < ITERS; i++) {
        gpio_set_level(INFER_TRIGGER_GPIO, 1);
        int64_t t0 = esp_timer_get_time();
        forward(test_image, logits, 0);
        t_inf += esp_timer_get_time() - t0;
        gpio_set_level(INFER_TRIGGER_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    recalib_reset();
    for (int i = 0; i < ITERS; i++) {
        gpio_set_level(PWR_TRIGGER_GPIO, 1);      // D1=1 labels this as an adapt window
        gpio_set_level(INFER_TRIGGER_GPIO, 1);
        int64_t t0 = esp_timer_get_time();
        forward(test_image, logits, 1);
        t_adapt += esp_timer_get_time() - t0;
        gpio_set_level(INFER_TRIGGER_GPIO, 0);
        gpio_set_level(PWR_TRIGGER_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    ESP_LOGI(TAG, "latency: inference %.2f ms | inference+adapt %.2f ms | adapt overhead %.2f ms",
             t_inf / 1000.0 / ITERS, t_adapt / 1000.0 / ITERS, (t_adapt - t_inf) / 1000.0 / ITERS);
    size_t heap1 = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "free heap at end: %u (used during run: %d)", (unsigned)heap1, (int)(heap0 - heap1));

    const double inf_ms = t_inf / 1000.0 / ITERS, adp_ms = t_adapt / 1000.0 / ITERS;
    ESP_LOGI(TAG, "RESULT golden_maxerr=%.4f inference=%.2fms adapt_overhead=%.2fms",
             maxerr, inf_ms, adp_ms - inf_ms);

    // ---- ENERGY MEASUREMENT LOOP (for the PPK2) ----
    // Alternate an inference-only window and an inference+adapt window forever. Each window
    // runs forward() EREPS times back-to-back so the two window DURATIONS differ by
    // EREPS*(adapt overhead) ~ 0.2 s on ~3 s windows -- cleanly separable by the duration
    // segmenter (with esp_nn the single-call gap is only ~22 ms, too small to resolve).
    // Per-call energy/time = window / EREPS. GPIO labels kept (D0=pin5 infer, D1=pin4 adapt).
    #define EREPS 10
    recalib_reset();
    for (;;) {
        gpio_set_level(PWR_TRIGGER_GPIO, 0);
        gpio_set_level(INFER_TRIGGER_GPIO, 1);
        for (int r = 0; r < EREPS; r++) forward(test_image, logits, 0);   // inference window
        gpio_set_level(INFER_TRIGGER_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(250));

        gpio_set_level(PWR_TRIGGER_GPIO, 1);
        gpio_set_level(INFER_TRIGGER_GPIO, 1);
        for (int r = 0; r < EREPS; r++) forward(test_image, logits, 1);   // inference+adapt window
        gpio_set_level(INFER_TRIGGER_GPIO, 0);
        gpio_set_level(PWR_TRIGGER_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(250));
    }
}
