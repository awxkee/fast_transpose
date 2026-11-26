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
use crate::{transpose_arbitrary, FlipMode, FlopMode, TransposeError};
use std::marker::PhantomData;

pub(crate) trait TransposeExecutor<F> {
    fn execute(
        &self,
        input: &[F],
        input_stride: usize,
        output: &mut [F],
        output_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<(), TransposeError>;
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct DefaultExecutor<F> {
    flip_mode: FlipMode,
    flop_mode: FlopMode,
    _phantom: PhantomData<F>,
}

#[allow(dead_code)]
impl<F: Copy> TransposeExecutor<F> for DefaultExecutor<F> {
    fn execute(
        &self,
        input: &[F],
        input_stride: usize,
        output: &mut [F],
        output_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<(), TransposeError> {
        transpose_arbitrary(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            self.flip_mode,
            self.flop_mode,
        )
    }
}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
struct TransposeBlockNeon4x4F32<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<f32>
    for TransposeBlockNeon4x4F32<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[f32], src_stride: usize, dst: &mut [f32], dst_stride: usize) {
        use crate::neon::neon_transpose_4x4_f32;
        neon_transpose_4x4_f32::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
struct TransposeBlockNeon8x8F32<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<f32>
    for TransposeBlockNeon8x8F32<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[f32], src_stride: usize, dst: &mut [f32], dst_stride: usize) {
        use crate::neon::neon_transpose_8x8_f32;
        neon_transpose_8x8_f32::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
#[derive(Copy, Clone, Default)]
struct NeonDefaultExecutor<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
pub(crate) fn make_neon_default_executor(
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Box<dyn TransposeExecutor<f32>> {
    match flip_mode {
        FlipMode::NoFlip => match flop_mode {
            FlopMode::NoFlop => Box::new(NeonDefaultExecutor::<false, false>::default()),
            FlopMode::Flop => Box::new(NeonDefaultExecutor::<true, false>::default()),
        },
        FlipMode::Flip => match flop_mode {
            FlopMode::NoFlop => Box::new(NeonDefaultExecutor::<false, true>::default()),
            FlopMode::Flop => Box::new(NeonDefaultExecutor::<true, true>::default()),
        },
    }
}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
impl<const FLOP: bool, const FLIP: bool> TransposeExecutor<f32>
    for NeonDefaultExecutor<FLOP, FLIP>
{
    fn execute(
        &self,
        input: &[f32],
        input_stride: usize,
        output: &mut [f32],
        output_stride: usize,
        width: usize,
        height: usize,
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

        let mut y = 0usize;

        y = transpose_executor::<f32, 8, 1, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
            TransposeBlockNeon8x8F32::<FLOP, FLIP> {},
        );

        y = transpose_executor::<f32, 4, 1, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
            TransposeBlockNeon4x4F32::<FLOP, FLIP> {},
        );

        transpose_section::<f32, 1, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
        );

        Ok(())
    }
}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
struct TransposeBlockAvx28x8<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<f32> for TransposeBlockAvx28x8<FLOP, FLIP> {
    #[inline(always)]
    fn transpose_block(&self, src: &[f32], src_stride: usize, dst: &mut [f32], dst_stride: usize) {
        use crate::avx::avx_transpose_8x8_f32;
        unsafe { avx_transpose_8x8_f32::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
struct TransposeBlockSSSE38x8<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<f32>
    for TransposeBlockSSSE38x8<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[f32], src_stride: usize, dst: &mut [f32], dst_stride: usize) {
        use crate::sse::sse_transpose_8x8_f32;
        unsafe { sse_transpose_8x8_f32::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
struct TransposeBlockSSSE34x4<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<f32>
    for TransposeBlockSSSE34x4<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[f32], src_stride: usize, dst: &mut [f32], dst_stride: usize) {
        use crate::sse::sse_transpose_4x4_f32;
        unsafe { sse_transpose_4x4_f32::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
#[derive(Copy, Clone, Default)]
struct Ssse3DefaultExecutor<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
fn make_ssse3_default_executor(
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Option<Box<dyn TransposeExecutor<f32>>> {
    if std::arch::is_x86_feature_detected!("ssse3") {
        return Some(match flip_mode {
            FlipMode::NoFlip => match flop_mode {
                FlopMode::NoFlop => Box::new(Ssse3DefaultExecutor::<false, false>::default()),
                FlopMode::Flop => Box::new(Ssse3DefaultExecutor::<true, false>::default()),
            },
            FlipMode::Flip => match flop_mode {
                FlopMode::NoFlop => Box::new(Ssse3DefaultExecutor::<false, true>::default()),
                FlopMode::Flop => Box::new(Ssse3DefaultExecutor::<true, true>::default()),
            },
        });
    }
    None
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
impl<const FLOP: bool, const FLIP: bool> Ssse3DefaultExecutor<FLOP, FLIP> {
    #[target_feature(enable = "ssse3")]
    unsafe fn execute_impl(
        &self,
        input: &[f32],
        input_stride: usize,
        output: &mut [f32],
        output_stride: usize,
        width: usize,
        height: usize,
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

        let mut y = 0usize;

        y = transpose_executor::<f32, 8, 1, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
            TransposeBlockSSSE38x8::<FLOP, FLIP> {},
        );

        y = transpose_executor::<f32, 4, 1, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
            TransposeBlockSSSE34x4::<FLOP, FLIP> {},
        );

        transpose_section::<f32, 1, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
        );

        Ok(())
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
impl<const FLOP: bool, const FLIP: bool> TransposeExecutor<f32>
    for Ssse3DefaultExecutor<FLOP, FLIP>
{
    fn execute(
        &self,
        input: &[f32],
        input_stride: usize,
        output: &mut [f32],
        output_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<(), TransposeError> {
        unsafe { self.execute_impl(input, input_stride, output, output_stride, width, height) }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
#[derive(Copy, Clone, Default)]
struct Avx2DefaultExecutor<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
pub(crate) fn make_avx2_default_executor(
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Option<Box<dyn TransposeExecutor<f32>>> {
    if std::arch::is_x86_feature_detected!("avx2") {
        return Some(match flip_mode {
            FlipMode::NoFlip => match flop_mode {
                FlopMode::NoFlop => Box::new(Avx2DefaultExecutor::<false, false>::default()),
                FlopMode::Flop => Box::new(Avx2DefaultExecutor::<true, false>::default()),
            },
            FlipMode::Flip => match flop_mode {
                FlopMode::NoFlop => Box::new(Avx2DefaultExecutor::<false, true>::default()),
                FlopMode::Flop => Box::new(Avx2DefaultExecutor::<true, true>::default()),
            },
        });
    }
    None
}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
impl<const FLOP: bool, const FLIP: bool> Avx2DefaultExecutor<FLOP, FLIP> {
    #[target_feature(enable = "avx2")]
    unsafe fn execute_impl(
        &self,
        input: &[f32],
        input_stride: usize,
        output: &mut [f32],
        output_stride: usize,
        width: usize,
        height: usize,
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

        let mut y = 0usize;

        y = transpose_executor::<f32, 8, 1, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
            TransposeBlockAvx28x8::<FLOP, FLIP> {},
        );

        y = transpose_executor::<f32, 4, 1, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
            TransposeBlockSSSE34x4::<FLOP, FLIP> {},
        );

        transpose_section::<f32, 1, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
        );

        Ok(())
    }
}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
impl<const FLOP: bool, const FLIP: bool> TransposeExecutor<f32>
    for Avx2DefaultExecutor<FLOP, FLIP>
{
    fn execute(
        &self,
        input: &[f32],
        input_stride: usize,
        output: &mut [f32],
        output_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<(), TransposeError> {
        unsafe { self.execute_impl(input, input_stride, output, output_stride, width, height) }
    }
}

pub(crate) fn prepare_f32_plane_executor(
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Box<dyn TransposeExecutor<f32>> {
    #[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
    {
        if let Some(executor) = make_avx2_default_executor(flip_mode, flop_mode) {
            return executor;
        }
    }
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "unsafe",
        feature = "sse"
    ))]
    {
        if let Some(executor) = make_ssse3_default_executor(flip_mode, flop_mode) {
            return executor;
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
    {
        make_neon_default_executor(flip_mode, flop_mode)
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "unsafe", feature = "neon")))]
    {
        Box::new(DefaultExecutor {
            flip_mode,
            flop_mode,
            _phantom: PhantomData,
        })
    }
}
