//! Checks every `transpose_*` function against the scalar `transpose_arbitrary` /
//! `transpose_arbitrary_grouped` for all flip/flop modes over a sweep of sizes around the block
//! sizes the SIMD paths use (2, 4, 8 and 16).

use fast_transpose::*;

type Transpose<T> = fn(
    &[T],
    usize,
    &mut [T],
    usize,
    usize,
    usize,
    FlipMode,
    FlopMode,
) -> Result<(), TransposeError>;

const SIZES: [usize; 17] = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 64, 65];

const MODES: [(FlipMode, FlopMode); 4] = [
    (FlipMode::NoFlip, FlopMode::NoFlop),
    (FlipMode::NoFlip, FlopMode::Flop),
    (FlipMode::Flip, FlopMode::NoFlop),
    (FlipMode::Flip, FlopMode::Flop),
];

fn check<T: Copy + Default + PartialEq>(
    transpose: Transpose<T>,
    reference: Transpose<T>,
    cn: usize,
    value: fn(usize) -> T,
) {
    let mut failures = Vec::new();
    for width in SIZES {
        for height in SIZES {
            // Also check that padding at the end of each row is left alone.
            for pad in [0, 3] {
                let src_stride = width * cn + pad;
                let dst_stride = height * cn + pad;
                let src: Vec<T> = (0..src_stride * height).map(value).collect();
                for (flip, flop) in MODES {
                    let mut expected = vec![T::default(); dst_stride * width];
                    let mut actual = expected.clone();
                    reference(
                        &src,
                        src_stride,
                        &mut expected,
                        dst_stride,
                        width,
                        height,
                        flip,
                        flop,
                    )
                    .unwrap();
                    transpose(
                        &src,
                        src_stride,
                        &mut actual,
                        dst_stride,
                        width,
                        height,
                        flip,
                        flop,
                    )
                    .unwrap();
                    if actual != expected {
                        failures.push(format!("{width}x{height}+{pad} {flip:?} {flop:?}"));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches: {}",
        failures.len(),
        failures.join(", ")
    );
}

fn u8_value(i: usize) -> u8 {
    (i as u32).wrapping_mul(2654435761).to_le_bytes()[3]
}

fn u16_value(i: usize) -> u16 {
    i as u16
}

fn f32_value(i: usize) -> f32 {
    i as f32
}

#[test]
fn plane() {
    check(transpose_plane, transpose_arbitrary::<u8>, 1, u8_value);
}

#[test]
fn plane_with_alpha() {
    check(
        transpose_plane_with_alpha,
        transpose_arbitrary_grouped::<u8, 2>,
        2,
        u8_value,
    );
}

#[test]
fn rgb() {
    check(
        transpose_rgb,
        transpose_arbitrary_grouped::<u8, 3>,
        3,
        u8_value,
    );
}

#[test]
fn rgba() {
    check(
        transpose_rgba,
        transpose_arbitrary_grouped::<u8, 4>,
        4,
        u8_value,
    );
}

#[test]
fn plane16() {
    check(transpose_plane16, transpose_arbitrary::<u16>, 1, u16_value);
}

#[test]
fn plane16_with_alpha() {
    check(
        transpose_plane16_with_alpha,
        transpose_arbitrary_grouped::<u16, 2>,
        2,
        u16_value,
    );
}

#[test]
fn rgb16() {
    check(
        transpose_rgb16,
        transpose_arbitrary_grouped::<u16, 3>,
        3,
        u16_value,
    );
}

#[test]
fn rgba16() {
    check(
        transpose_rgba16,
        transpose_arbitrary_grouped::<u16, 4>,
        4,
        u16_value,
    );
}

#[test]
fn plane_f32() {
    check(
        transpose_plane_f32,
        transpose_arbitrary::<f32>,
        1,
        f32_value,
    );
}

#[test]
fn plane_f32_with_alpha() {
    check(
        transpose_plane_f32_with_alpha,
        transpose_arbitrary_grouped::<f32, 2>,
        2,
        f32_value,
    );
}

#[test]
fn rgb_f32() {
    check(
        transpose_rgb_f32,
        transpose_arbitrary_grouped::<f32, 3>,
        3,
        f32_value,
    );
}

#[test]
fn rgba_f32() {
    check(
        transpose_rgba_f32,
        transpose_arbitrary_grouped::<f32, 4>,
        4,
        f32_value,
    );
}
