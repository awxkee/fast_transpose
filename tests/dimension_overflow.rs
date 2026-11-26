use fast_transpose::*;

/// `HUGE * 4` wraps to 4.
const HUGE: usize = (1usize << 62) + 1;
/// `HUGE2 * 2` wraps to 4, `HUGE2 * 4` wraps to 8.
const HUGE2: usize = (1usize << 63) + 2;

fn assert_mismatch(r: Result<(), TransposeError>) {
    assert!(
        matches!(r, Err(TransposeError::MismatchDimensions)),
        "expected MismatchDimensions, got {:?}",
        r
    );
}

const MODES: [(FlipMode, FlopMode); 4] = [
    (FlipMode::NoFlip, FlopMode::NoFlop),
    (FlipMode::NoFlip, FlopMode::Flop),
    (FlipMode::Flip, FlopMode::NoFlop),
    (FlipMode::Flip, FlopMode::Flop),
];

#[test]
fn transpose_interleaved_rejects_overflowing_width() {
    for (flip, flop) in MODES {
        // stride 4, width HUGE, height 1: width * 4 and 4 * width wrap to 4.
        let i = vec![1u8; 4];
        let mut o = vec![0u8; 4];
        assert_mismatch(transpose_rgba(&i, 4, &mut o, 4, HUGE, 1, flip, flop));
        assert_mismatch(transpose_arbitrary_grouped::<u8, 4>(
            &i, 4, &mut o, 4, HUGE, 1, flip, flop,
        ));

        let i = vec![1u16; 4];
        let mut o = vec![0u16; 4];
        assert_mismatch(transpose_rgba16(&i, 4, &mut o, 4, HUGE, 1, flip, flop));

        let i = vec![1f32; 4];
        let mut o = vec![0f32; 4];
        assert_mismatch(transpose_rgba_f32(&i, 4, &mut o, 4, HUGE, 1, flip, flop));

        // Two channels: width * 2 wraps to 4, output_stride * width wraps to 8.
        let i = vec![1u8; 4];
        let mut o = vec![0u8; 8];
        assert_mismatch(transpose_plane_with_alpha(
            &i, 4, &mut o, 4, HUGE2, 1, flip, flop,
        ));
        let i = vec![1u16; 4];
        let mut o = vec![0u16; 8];
        assert_mismatch(transpose_plane16_with_alpha(
            &i, 4, &mut o, 4, HUGE2, 1, flip, flop,
        ));
        let i = vec![1f32; 4];
        let mut o = vec![0f32; 8];
        assert_mismatch(transpose_plane_f32_with_alpha(
            &i, 4, &mut o, 4, HUGE2, 1, flip, flop,
        ));
    }
}

#[test]
fn transpose_planar_rejects_overflowing_dimensions() {
    for (flip, flop) in MODES {
        // input_stride HUGE, height 4 -> 4; output_stride 4, width HUGE -> 4.
        let i = vec![1u8; 4];
        let mut o = vec![0u8; 4];
        assert_mismatch(transpose_plane(&i, HUGE, &mut o, 4, HUGE, 4, flip, flop));
        assert_mismatch(transpose_arbitrary(
            &i, HUGE, &mut o, 4, HUGE, 4, flip, flop,
        ));

        let i = vec![1u16; 4];
        let mut o = vec![0u16; 4];
        assert_mismatch(transpose_plane16(&i, HUGE, &mut o, 4, HUGE, 4, flip, flop));

        let i = vec![1f32; 4];
        let mut o = vec![0f32; 4];
        assert_mismatch(transpose_plane_f32(
            &i, HUGE, &mut o, 4, HUGE, 4, flip, flop,
        ));
    }
}

#[test]
fn flip_flop_rotate_reject_overflowing_dimensions() {
    let i = vec![1u8; 4];
    let mut o = vec![0u8; 4];
    // width * 4 wraps to 4.
    assert_mismatch(flip_rgba(&i, 4, &mut o, 4, HUGE, 1));
    assert_mismatch(flop_rgba(&i, 4, &mut o, 4, HUGE, 1));
    assert_mismatch(rotate180_rgba(&i, 4, &mut o, 4, HUGE, 1));
    // stride * height wraps to 4.
    assert_mismatch(flip_plane(&i, 4, &mut o, 4, 1, HUGE));
    assert_mismatch(flop_plane(&i, 4, &mut o, 4, 1, HUGE));
    assert_mismatch(rotate180_plane(&i, 4, &mut o, 4, 1, HUGE));
}
