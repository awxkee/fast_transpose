/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#[allow(unused_imports)]
use crate::rgba8::*;
use crate::{FlipMode, FlopMode, TransposeError};

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
struct TransposeBlockNeon2x2<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<u16> for TransposeBlockNeon2x2<FLOP, FLIP> {
    #[inline(always)]
    fn transpose_block(&self, src: &[u16], src_stride: usize, dst: &mut [u16], dst_stride: usize) {
        use crate::neon::neon_transpose_u16x4_2x2;
        neon_transpose_u16x4_2x2::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
struct TransposeBlockNeon4x4<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<u16> for TransposeBlockNeon4x4<FLOP, FLIP> {
    #[inline(always)]
    fn transpose_block(&self, src: &[u16], src_stride: usize, dst: &mut [u16], dst_stride: usize) {
        use crate::neon::neon_transpose_u16x4_4x4;
        neon_transpose_u16x4_4x4::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
struct TransposeBlockSSSE3_2x2<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<u16>
    for TransposeBlockSSSE3_2x2<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[u16], src_stride: usize, dst: &mut [u16], dst_stride: usize) {
        use crate::sse::ssse_transpose_u16x4_2x2;
        ssse_transpose_u16x4_2x2::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
struct TransposeBlockAvx2_4x4<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<u16>
    for TransposeBlockAvx2_4x4<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[u16], src_stride: usize, dst: &mut [u16], dst_stride: usize) {
        use crate::avx::avx2_transpose_u16x4_4x4;
        unsafe { avx2_transpose_u16x4_4x4::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
fn transpose_rgba16_impl_neon<const FLOP: bool, const FLIP: bool>(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) {
    const CN: usize = 4;

    let mut y = 0usize;

    y = transpose_executor::<u16, 4, 4, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
        TransposeBlockNeon4x4::<FLOP, FLIP> {},
    );

    y = transpose_executor::<u16, 2, 4, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
        TransposeBlockNeon2x2::<FLOP, FLIP> {},
    );

    transpose_section::<u16, CN, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
    )
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
#[target_feature(enable = "ssse3")]
unsafe fn transpose_rgba16_impl_ssse3<const FLOP: bool, const FLIP: bool>(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) {
    const CN: usize = 4;

    let mut y = 0usize;

    y = transpose_executor::<u16, 2, 4, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
        TransposeBlockSSSE3_2x2::<FLOP, FLIP> {},
    );

    transpose_section::<u16, CN, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
    )
}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
#[target_feature(enable = "avx2")]
unsafe fn transpose_rgba16_impl_avx2<const FLOP: bool, const FLIP: bool>(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) {
    const CN: usize = 4;

    let mut y = 0usize;

    y = transpose_executor::<u16, 4, 4, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
        TransposeBlockAvx2_4x4::<FLOP, FLIP> {},
    );

    y = transpose_executor::<u16, 2, 4, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
        TransposeBlockSSSE3_2x2::<FLOP, FLIP> {},
    );

    transpose_section::<u16, CN, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
    )
}

pub(crate) fn transpose_rgba16_chunked(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    if input_stride.checked_mul(height) != Some(input.len()) {
        return Err(TransposeError::MismatchDimensions);
    }
    if output_stride.checked_mul(width) != Some(output.len()) {
        return Err(TransposeError::MismatchDimensions);
    }
    if width.checked_mul(4).is_none_or(|v| input_stride < v) {
        return Err(TransposeError::MismatchDimensions);
    }
    if height.checked_mul(4).is_none_or(|v| output_stride < v) {
        return Err(TransposeError::MismatchDimensions);
    }

    #[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
    {
        let executor = match flip_mode {
            FlipMode::NoFlip => match flop_mode {
                FlopMode::NoFlop => transpose_rgba16_impl_neon::<false, false>,
                FlopMode::Flop => transpose_rgba16_impl_neon::<true, false>,
            },
            FlipMode::Flip => match flop_mode {
                FlopMode::NoFlop => transpose_rgba16_impl_neon::<false, true>,
                FlopMode::Flop => transpose_rgba16_impl_neon::<true, true>,
            },
        };
        executor(input, input_stride, output, output_stride, width, height);
        Ok(())
    }
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "unsafe",
        any(feature = "sse", feature = "avx")
    ))]
    {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            let executor = match flip_mode {
                FlipMode::NoFlip => match flop_mode {
                    FlopMode::NoFlop => transpose_rgba16_impl_avx2::<false, false>,
                    FlopMode::Flop => transpose_rgba16_impl_avx2::<true, false>,
                },
                FlipMode::Flip => match flop_mode {
                    FlopMode::NoFlop => transpose_rgba16_impl_avx2::<false, true>,
                    FlopMode::Flop => transpose_rgba16_impl_avx2::<true, true>,
                },
            };

            unsafe { executor(input, input_stride, output, output_stride, width, height) }
            return Ok(());
        }

        if std::arch::is_x86_feature_detected!("ssse3") {
            let executor: unsafe fn(&[u16], usize, &mut [u16], usize, usize, usize) =
                match flip_mode {
                    FlipMode::NoFlip => match flop_mode {
                        FlopMode::NoFlop => transpose_rgba16_impl_ssse3::<false, false>,
                        FlopMode::Flop => transpose_rgba16_impl_ssse3::<true, false>,
                    },
                    FlipMode::Flip => match flop_mode {
                        FlopMode::NoFlop => transpose_rgba16_impl_ssse3::<false, true>,
                        FlopMode::Flop => transpose_rgba16_impl_ssse3::<true, true>,
                    },
                };
            unsafe { executor(input, input_stride, output, output_stride, width, height) }
            return Ok(());
        }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "unsafe", feature = "neon")))]
    {
        use crate::transpose_arbitrary_group::transpose_arbitrary_grouped;
        transpose_arbitrary_grouped::<u16, 4>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            flip_mode,
            flop_mode,
        )
    }
}
