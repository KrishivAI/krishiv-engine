// SPDX-License-Identifier: Apache-2.0
//! The fdlibm routines Java's `StrictMath` runs, for the functions where the
//! pure-Rust `libm` crate (musl's algorithms) differs from them in the last
//! bit: `cbrt`, `log10`, `hypot`, `acosh` and `tanh`. Spark calls these
//! through the JVM, so reproducing the same algorithm reproduces its bits.
//!
//! The algorithms are Sun Microsystems' fdlibm (1993): "Developed at
//! SunSoft, a Sun Microsystems, Inc. business. Permission to use, copy,
//! modify, and distribute this software is freely granted, provided that this
//! notice is preserved." The bit manipulation is explicit (`wrapping_*`)
//! because the crate denies implicit integer arithmetic.

/// The high 32 bits of `x` (as fdlibm's `__HI`).
fn hi(x: f64) -> i32 {
    let b = x.to_bits().to_be_bytes();
    i32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// The low 32 bits of `x` (`__LO`).
fn lo(x: f64) -> u32 {
    let b = x.to_bits().to_be_bytes();
    u32::from_be_bytes([b[4], b[5], b[6], b[7]])
}

/// `x` with its high word replaced (`SET_HIGH_WORD`).
fn with_hi(x: f64, high: i32) -> f64 {
    let h = high.to_be_bytes();
    let l = x.to_bits().to_be_bytes();
    f64::from_bits(u64::from_be_bytes([
        h[0], h[1], h[2], h[3], l[4], l[5], l[6], l[7],
    ]))
}

/// `cbrt(x)` (`s_cbrt.c`): a 23-bit estimate refined by one Newton step.
pub(super) fn cbrt(x: f64) -> f64 {
    const B1: i32 = 715_094_163;
    const B2: i32 = 696_219_795;
    const C: f64 = 5.428_571_428_571_428e-1;
    const D: f64 = -7.053_061_224_489_796e-1;
    const E: f64 = 1.414_285_714_285_714_4;
    const F: f64 = 1.607_142_857_142_857_2;
    const G: f64 = 3.571_428_571_428_571_5e-1;
    let high = hi(x);
    let negative = high < 0;
    let abs_high = high & 0x7fff_ffff;
    if abs_high >= 0x7ff0_0000 {
        return x + x;
    }
    if (abs_high.cast_unsigned() | lo(x)) == 0 {
        return x;
    }
    let x = with_hi(x, abs_high);
    let mut t = if abs_high < 0x0010_0000 {
        // Subnormal: scale by 2^54 first.
        let scaled = with_hi(0.0, 0x4350_0000) * x;
        with_hi(0.0, (hi(scaled) / 3).wrapping_add(B2))
    } else {
        with_hi(0.0, (abs_high / 3).wrapping_add(B1))
    };
    // A new cbrt to 23 bits.
    let r = t * t / x;
    let s = C + r * t;
    t *= G + F / (s + E + D / s);
    // Chop to 20 bits and make it larger than cbrt(x).
    t = f64::from_bits((t.to_bits() & 0xffff_ffff_0000_0000).wrapping_add(0x1_0000_0000));
    // One Newton step to 53 bits with error below 0.667 ulps.
    let s = t * t;
    let r = x / s;
    let w = t + t;
    let r = (r - t) / (w + r);
    t += t * r;
    if negative { -t } else { t }
}

/// `log10(x)` (`e_log10.c`): `log10(2^k * m) = k*log10(2) + log(m)/ln(10)`.
pub(super) fn log10(x: f64) -> f64 {
    const TWO54: f64 = 1.801_439_850_948_198_4e16;
    const IVLN10: f64 = std::f64::consts::LOG10_E;
    const LOG10_2HI: f64 = 3.010_299_956_636_117_7e-1;
    const LOG10_2LO: f64 = 3.694_239_077_158_931e-13;
    let mut x = x;
    let mut high = hi(x);
    let mut k: i32 = 0;
    if high < 0x0010_0000 {
        // x < 2^-1022, or not positive.
        if ((high & 0x7fff_ffff).cast_unsigned() | lo(x)) == 0 {
            return f64::NEG_INFINITY;
        }
        if high < 0 {
            return f64::NAN;
        }
        k = k.wrapping_sub(54);
        x *= TWO54;
        high = hi(x);
    }
    if high >= 0x7ff0_0000 {
        return x + x;
    }
    k = k.wrapping_add((high >> 20).wrapping_sub(1023));
    let i = i32::from(k < 0);
    high = (high & 0x000f_ffff) | (0x3ff_i32.wrapping_sub(i) << 20);
    let y = f64::from(k.wrapping_add(i));
    let x = with_hi(x, high);
    let z = y * LOG10_2LO + IVLN10 * libm::log(x);
    z + y * LOG10_2HI
}

/// `hypot(x, y)` (`e_hypot.c`): `sqrt(x*x + y*y)` without undue overflow
/// or underflow, with the squares expanded in pieces so the error stays
/// below one ulp.
pub(super) fn hypot(x: f64, y: f64) -> f64 {
    let (mut a, mut b) = (x, y);
    let mut ha = hi(a) & 0x7fff_ffff;
    let mut hb = hi(b) & 0x7fff_ffff;
    if hb > ha {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut ha, &mut hb);
    }
    a = with_hi(a, ha);
    b = with_hi(b, hb);
    if ha.wrapping_sub(hb) > 0x3c0_0000 {
        // x/y > 2^60.
        return a + b;
    }
    let mut k: i32 = 0;
    if ha > 0x5f30_0000 {
        // a > 2^500.
        if ha >= 0x7ff0_0000 {
            // Infinity or NaN.
            let mut w = a + b;
            if ((ha & 0xf_ffff).cast_unsigned() | lo(a)) == 0 {
                w = a;
            }
            if ((hb ^ 0x7ff0_0000).cast_unsigned() | lo(b)) == 0 {
                w = b;
            }
            return w;
        }
        // Scale a and b by 2^-600.
        ha = ha.wrapping_sub(0x2580_0000);
        hb = hb.wrapping_sub(0x2580_0000);
        k = k.wrapping_add(600);
        a = with_hi(a, ha);
        b = with_hi(b, hb);
    }
    if hb < 0x20b0_0000 {
        // b < 2^-500.
        if hb <= 0x000f_ffff {
            // Subnormal b or 0.
            if (hb.cast_unsigned() | lo(b)) == 0 {
                return a;
            }
            let t1 = with_hi(0.0, 0x7fd0_0000); // 2^1022
            b *= t1;
            a *= t1;
            k = k.wrapping_sub(1022);
        } else {
            // Scale a and b by 2^600.
            ha = ha.wrapping_add(0x2580_0000);
            hb = hb.wrapping_add(0x2580_0000);
            k = k.wrapping_sub(600);
            a = with_hi(a, ha);
            b = with_hi(b, hb);
        }
    }
    // Medium-size a and b.
    let mut w = a - b;
    if w > b {
        let t1 = with_hi(0.0, ha);
        let t2 = a - t1;
        w = (t1 * t1 - (b * (-b) - t2 * (a + t1))).sqrt();
    } else {
        a += a;
        let y1 = with_hi(0.0, hb);
        let y2 = b - y1;
        let t1 = with_hi(0.0, ha.wrapping_add(0x0010_0000));
        let t2 = a - t1;
        w = (t1 * y1 - (w * (-w) - (t1 * y2 + t2 * b))).sqrt();
    }
    if k == 0 {
        w
    } else {
        with_hi(1.0, hi(1.0).wrapping_add(k << 20)) * w
    }
}

/// `acosh(x)` (`e_acosh.c`).
pub(super) fn acosh(x: f64) -> f64 {
    const LN2: f64 = std::f64::consts::LN_2;
    if x.is_nan() || x < 1.0 {
        return f64::NAN;
    }
    if x >= 268_435_456.0 {
        // x >= 2^28.
        return if x.is_infinite() {
            x + x
        } else {
            libm::log(x) + LN2
        };
    }
    if x == 1.0 {
        return 0.0;
    }
    if x > 2.0 {
        let t = x * x;
        return libm::log(2.0 * x - 1.0 / (x + (t - 1.0).sqrt()));
    }
    let t = x - 1.0;
    libm::log1p(t + (2.0 * t + t * t).sqrt())
}

/// `tanh(x)` (`s_tanh.c`).
pub(super) fn tanh(x: f64) -> f64 {
    let high = hi(x);
    let abs_high = high & 0x7fff_ffff;
    if abs_high >= 0x7ff0_0000 {
        // Infinity or NaN.
        return if high >= 0 {
            1.0 / x + 1.0
        } else {
            1.0 / x - 1.0
        };
    }
    let z = if abs_high < 0x4036_0000 {
        // |x| < 22.
        if abs_high < 0x3c80_0000 {
            return x * (1.0 + x);
        }
        if abs_high >= 0x3ff0_0000 {
            let t = libm::expm1(2.0 * x.abs());
            1.0 - 2.0 / (t + 2.0)
        } else {
            let t = libm::expm1(-2.0 * x.abs());
            -t / (t + 2.0)
        }
    } else {
        1.0 - 1.0e-300
    };
    if high >= 0 { z } else { -z }
}
