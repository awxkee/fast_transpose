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
struct TransposePlaneBlockNeon8x8<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<u8>
    for TransposePlaneBlockNeon8x8<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[u8], src_stride: usize, dst: &mut [u8], dst_stride: usize) {
        use crate::neon::neon_transpose_u8_8x8;
        neon_transpose_u8_8x8::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
struct TransposePlaneBlockSSSe3_8x8<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<u8>
    for TransposePlaneBlockSSSe3_8x8<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[u8], src_stride: usize, dst: &mut [u8], dst_stride: usize) {
        use crate::sse::sse_transpose_u8_8x8;
        sse_transpose_u8_8x8::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
struct TransposePlaneBlockNeon16x16<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<u8>
    for TransposePlaneBlockNeon16x16<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[u8], src_stride: usize, dst: &mut [u8], dst_stride: usize) {
        use crate::neon::neon_transpose_u8_16x16;
        neon_transpose_u8_16x16::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
fn transpose_plane8_impl_neon<const FLOP: bool, const FLIP: bool>(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) {
    const CN: usize = 1;

    let mut y = 0usize;

    y = transpose_executor::<u8, 16, CN, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
        TransposePlaneBlockNeon16x16::<FLOP, FLIP> {},
    );

    y = transpose_executor::<u8, 8, CN, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
        TransposePlaneBlockNeon8x8::<FLOP, FLIP> {},
    );

    transpose_section::<u8, CN, FLOP, FLIP>(
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
unsafe fn transpose_plane8_impl_ssse3<const FLOP: bool, const FLIP: bool>(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) {
    const CN: usize = 1;

    let mut y = 0usize;

    y = transpose_executor::<u8, 8, CN, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
        TransposePlaneBlockSSSe3_8x8::<FLOP, FLIP> {},
    );

    transpose_section::<u8, CN, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
    )
}

pub(crate) fn transpose_plane8_chunked(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
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
    if input_stride < width {
        return Err(TransposeError::MismatchDimensions);
    }
    if output_stride < height {
        return Err(TransposeError::MismatchDimensions);
    }

    #[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
    {
        let executor = match flip_mode {
            FlipMode::NoFlip => match flop_mode {
                FlopMode::NoFlop => transpose_plane8_impl_neon::<false, false>,
                FlopMode::Flop => transpose_plane8_impl_neon::<true, false>,
            },
            FlipMode::Flip => match flop_mode {
                FlopMode::NoFlop => transpose_plane8_impl_neon::<false, true>,
                FlopMode::Flop => transpose_plane8_impl_neon::<true, true>,
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
        if !std::arch::is_x86_feature_detected!("ssse3") {
            use crate::transpose_arbitrary::transpose_arbitrary;
            return transpose_arbitrary::<u8>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
                flip_mode,
                flop_mode,
            );
        }

        let executor: unsafe fn(&[u8], usize, &mut [u8], usize, usize, usize) = match flip_mode {
            FlipMode::NoFlip => match flop_mode {
                FlopMode::NoFlop => transpose_plane8_impl_ssse3::<false, false>,
                FlopMode::Flop => transpose_plane8_impl_ssse3::<true, false>,
            },
            FlipMode::Flip => match flop_mode {
                FlopMode::NoFlop => transpose_plane8_impl_ssse3::<false, true>,
                FlopMode::Flop => transpose_plane8_impl_ssse3::<true, true>,
            },
        };

        unsafe { executor(input, input_stride, output, output_stride, width, height) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "unsafe", feature = "neon")))]
    {
        use crate::transpose_arbitrary::transpose_arbitrary;
        transpose_arbitrary::<u8>(
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
