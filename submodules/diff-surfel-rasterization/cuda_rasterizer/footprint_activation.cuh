// Experimental local-slab adaptation, NOT a reproduction of full GPIS transport.
// See README.md for derivation, conventions, approximations and integration.
#pragma once
#include <cmath>

#ifdef __CUDACC__
#define FP_HD __host__ __device__ __forceinline__
#else
#define FP_HD inline
#endif

// 0: linear; 1: Beer; 2: Gaussian height-field; 3: Gaussian gradient; 4: SGGX slab; 5: normalized Gaussian-CDF.
// Experimental default: SGGX slab. Compare modes 0/1/2/3 with identical settings.
#ifndef FOOTPRINT_MODE
#define FOOTPRINT_MODE 4
#endif
#ifndef FOOTPRINT_SCALE
#define FOOTPRINT_SCALE 1.0f
#endif
#ifndef FOOTPRINT_SLOPE_T
#define FOOTPRINT_SLOPE_T 0.35f
#endif
#ifndef FOOTPRINT_SLOPE_N
#define FOOTPRINT_SLOPE_N 0.15f
#endif
#ifndef FOOTPRINT_MU_MIN
#define FOOTPRINT_MU_MIN 0.05f
#endif
#ifndef FOOTPRINT_RATIO_MAX
#define FOOTPRINT_RATIO_MAX 4.0f
#endif
#ifndef FOOTPRINT_CDF_OFFSET
#define FOOTPRINT_CDF_OFFSET 3.0f
#endif
#ifndef FOOTPRINT_CUTOFF
#define FOOTPRINT_CUTOFF 4.5f
#endif

namespace footprint {
enum Mode { Linear = 0, Beer = 1, HeightField = 2, Generalized = 3, SGGX = 4, GaussianCDF = 5 };
struct Config {
    Mode mode;
    float scale, slope_t, slope_n, mu_min, ratio_max, alpha_max;
};
FP_HD Config defaults() {
    return {static_cast<Mode>(FOOTPRINT_MODE), FOOTPRINT_SCALE,
            FOOTPRINT_SLOPE_T, FOOTPRINT_SLOPE_N, FOOTPRINT_MU_MIN,
            FOOTPRINT_RATIO_MAX, 0.99f};
}
struct Result {
    float alpha;
    float d_opacity, d_G, d_mu;
    // These derivatives are available, but the supplied integration keeps
    // slopes global/fixed. Learning them needs extra tensor/optimizer plumbing.
    float d_slope_t, d_slope_n;
};
struct PositiveMoment { float value, d_mean, d_stddev; };

// E[max(X,0)] for X ~ N(mean, sd^2), and its exact partial derivatives.
// mean >= 0 in this adapter (two-sided facing convention).
FP_HD PositiveMoment positive_moment(float mean, float sd) {
    if (sd <= 0.0f) return {mean, 1.0f, 0.0f};
    const float z = mean / sd;
    const float phi = 0.3989422804014327f * expf(-0.5f * z * z);
    const float Phi = 0.5f * erfcf(-0.7071067811865475f * z);
    return {mean * Phi + sd * phi, Phi, phi};
}

// Inputs: opacity >= 0, 0 <= G <= 1, 0 <= mu=|n.d| <= 1.
// Config: scale/slopes >= 0, 0 < mu_min < 1, ratio_max >= 1,
//         0 < alpha_max < 1. No trainable clamps hidden in parameterization.
FP_HD Result activation(float opacity, float G, float mu,
                        const Config& cfg = defaults()) {
    Result out = {};
    if (cfg.mode == GaussianCDF) {
        // Geometry-field-inspired mapping. Not the full OaV/GFSGS renderer.
        // alpha=1-[Phi(c-x)/Phi(c)]^2, x=scale*opacity*G >= 0.
        // Normalization enforces alpha(0)=0 for finite c.
        const float c = FOOTPRINT_CDF_OFFSET;
        const float x = cfg.scale * opacity * G;
        const float inv_sqrt2 = 0.7071067811865475f;
        const float base_tail = erfcf(c * inv_sqrt2);
        const float Phi_c = 1.0f - 0.5f * base_tail;
        const float removed = 0.5f * (erfcf((c-x)*inv_sqrt2)-base_tail) / Phi_c;
        const float raw = removed * (2.0f-removed);
        out.alpha = fminf(cfg.alpha_max, raw);
        if (raw >= cfg.alpha_max) return out;
        const float phi = 0.3989422804014327f * expf(-0.5f*(c-x)*(c-x));
        const float derivative = 2.0f*(1.0f-removed)*phi/Phi_c;
        out.d_opacity = derivative * cfg.scale * G;
        out.d_G = derivative * cfg.scale * opacity;
        return out;
    }
    float ratio = 1.0f, r_mu = 0.0f, r_t = 0.0f, r_n = 0.0f;
    if (cfg.mode == HeightField || cfg.mode == Generalized) {
        const float m = fmaxf(mu, cfg.mu_min);
        const float st = cfg.slope_t;
        const float sn = cfg.mode == Generalized ? cfg.slope_n : 0.0f;
        const float tangent2 = fmaxf(0.0f, 1.0f - m * m);
        const float sd = sqrtf(st * st * tangent2 + sn * sn * m * m);
        const PositiveMoment A = positive_moment(m, sd);
        const PositiveMoment A0 = positive_moment(1.0f, sn);
        const float inv = 1.0f / (m * A0.value);
        ratio = A.value * inv;
        const float ds_dm = sd > 0.0f ? m * (sn * sn - st * st) / sd : 0.0f;
        r_mu = (A.d_mean + A.d_stddev * ds_dm) * inv - ratio / m;
        r_t = sd > 0.0f ? A.d_stddev * st * tangent2 / sd * inv : 0.0f;
        if (cfg.mode == Generalized) {
            r_n = (sd > 0.0f ? A.d_stddev * sn * m * m / sd * inv : 0.0f)
                  - ratio * A0.d_stddev / A0.value;
        }
        if (mu <= cfg.mu_min) r_mu = 0.0f;
        if (ratio >= cfg.ratio_max) {
            ratio = cfg.ratio_max;
            r_mu = r_t = r_n = 0.0f;
        }
    }
    if (cfg.mode == SGGX) {
        // S = r^2(I-nn^T) + nn^T; eigenvalues (r^2,r^2,1).
        // Projected area sqrt(d^T S d), divided by slab path cosine.
        const float m = fmaxf(mu, cfg.mu_min);
        const float r = cfg.slope_t; // Projected-area ratio, NOT Gaussian slope SD.
        const float tangent2 = fmaxf(0.0f, 1.0f - m * m);
        const float A = sqrtf(m * m + r * r * tangent2);
        ratio = A / m;
        r_mu = -r * r / (A * m * m);
        r_t = r * tangent2 / (A * m);
        if (mu <= cfg.mu_min) r_mu = 0.0f;
        if (ratio >= cfg.ratio_max) {
            ratio = cfg.ratio_max;
            r_mu = r_t = 0.0f;
        }
    }
    const float base = cfg.scale * opacity * G;
    const float tau = base * ratio;
    const float raw = cfg.mode == Linear ? tau : -expm1f(-tau);
    out.alpha = fminf(cfg.alpha_max, raw);
    // Exact derivative of the implemented hard cap, away from its kink.
    if (raw >= cfg.alpha_max) return out;
    const float d_alpha_d_tau = cfg.mode == Linear ? 1.0f : expf(-tau);
    out.d_opacity = d_alpha_d_tau * cfg.scale * G * ratio;
    out.d_G = d_alpha_d_tau * cfg.scale * opacity * ratio;
    out.d_mu = d_alpha_d_tau * base * r_mu;
    out.d_slope_t = d_alpha_d_tau * base * r_t;
    out.d_slope_n = d_alpha_d_tau * base * r_n;
    return out;
}

struct Vec3 { float x, y, z; };
struct Incidence { float mu; Vec3 d_normal; };
FP_HD Vec3 camera_ray(float px, float py, int width, int height,
                      float fx, float fy) {
    // Matches this repo's centered pinhole convention, cx=(W-1)/2.
    // For off-center projection matrices, pass the true camera ray instead.
    const float x = (px - 0.5f * (width - 1)) / fx;
    const float y = (py - 0.5f * (height - 1)) / fy;
    const float inv = 1.0f / sqrtf(x * x + y * y + 1.0f);
    return {x * inv, y * inv, inv};
}
FP_HD Incidence incidence(Vec3 normal, Vec3 unit_ray) {
    const float len2 = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
    if (len2 <= 1e-20f) return {1.0f, {0.0f, 0.0f, 0.0f}};
    const float inv = 1.0f / sqrtf(len2);
    const Vec3 n = {normal.x * inv, normal.y * inv, normal.z * inv};
    const float dot = n.x * unit_ray.x + n.y * unit_ray.y + n.z * unit_ray.z;
    const float sign = dot > 0 ? 1.0f : (dot < 0 ? -1.0f : 0.0f);
    if (fabsf(dot) >= 1.0f) return {1.0f, {0.0f, 0.0f, 0.0f}};
    return {fabsf(dot), {sign * inv * (unit_ray.x - dot * n.x),
                        sign * inv * (unit_ray.y - dot * n.y),
                        sign * inv * (unit_ray.z - dot * n.z)}};
}
// The integration uses a fixed cutoff so center derivatives stay unchanged.
// Check this bound on the host before launching the preprocessing kernel.
FP_HD float supported_opacity_max() {
    if (FOOTPRINT_MODE == 5) {
        // For c>=0: d(alpha)/dx <= 2/(sqrt(2*pi)*Phi(c)) <= 1.59577.
        return (1.0f/255.0f)/1.59577f * expf(0.5f*FOOTPRINT_CUTOFF*FOOTPRINT_CUTOFF)
               / FOOTPRINT_SCALE;
    }
    const float response_max = FOOTPRINT_MODE >= 2 ? FOOTPRINT_RATIO_MAX : 1.0f;
    const float threshold = FOOTPRINT_MODE == 0 ? (1.0f / 255.0f)
                                               : -log1pf(-1.0f / 255.0f);
    return threshold * expf(0.5f * FOOTPRINT_CUTOFF * FOOTPRINT_CUTOFF)
           / (FOOTPRINT_SCALE * response_max);
}
static_assert(FOOTPRINT_MODE >= 0 && FOOTPRINT_MODE <= 5, "Invalid FOOTPRINT_MODE");
static_assert(FOOTPRINT_CDF_OFFSET >= 0, "CDF offset must be nonnegative");
static_assert(FOOTPRINT_SCALE > 0, "Scale must be positive");
static_assert(FOOTPRINT_SLOPE_T >= 0 && FOOTPRINT_SLOPE_N >= 0, "Negative roughness");
static_assert(FOOTPRINT_MU_MIN > 0 && FOOTPRINT_MU_MIN < 1, "Invalid grazing clamp");
static_assert(FOOTPRINT_RATIO_MAX >= 1, "Ratio cap must be at least one");
static_assert(FOOTPRINT_CUTOFF >= 3, "Cutoff must contain original low-pass center support");
} // namespace footprint
#undef FP_HD
