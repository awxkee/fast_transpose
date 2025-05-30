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
use crate::float32_invoker::TransposeExecutor;
#[allow(unused_imports)]
use crate::rgba8::*;
use crate::transpose_arbitrary_group::transpose_arbitrary_grouped;
use crate::{FlipMode, FlopMode, TransposeError};
use std::marker::PhantomData;

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct DefaultCbCrExecutor<F> {
    flip_mode: FlipMode,
    flop_mode: FlopMode,
    _phantom: PhantomData<F>,
}

#[allow(dead_code)]
impl<F: Copy> TransposeExecutor<F> for DefaultCbCrExecutor<F> {
    fn execute(
        &self,
        input: &[F],
        input_stride: usize,
        output: &mut [F],
        output_stride: usize,
        width: usize,
        height: usize,
    ) -> Result<(), TransposeError> {
        transpose_arbitrary_grouped::<F, 2>(
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
struct TransposeBlockNeon4x4F32x2<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<f32>
    for TransposeBlockNeon4x4F32x2<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[f32], src_stride: usize, dst: &mut [f32], dst_stride: usize) {
        use crate::neon::neon_transpose_f32x2_4x4;
        neon_transpose_f32x2_4x4::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
struct TransposeBlockNeon2x2F32x2<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<f32>
    for TransposeBlockNeon2x2F32x2<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[f32], src_stride: usize, dst: &mut [f32], dst_stride: usize) {
        use crate::neon::neon_transpose_f32x2_2x2;
        neon_transpose_f32x2_2x2::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
struct TransposeBlockSse2x2F32x2<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "sse"
))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<f32>
    for TransposeBlockSse2x2F32x2<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[f32], src_stride: usize, dst: &mut [f32], dst_stride: usize) {
        use crate::sse::ssse_transpose_f32x2_2x2;
        ssse_transpose_f32x2_2x2::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
struct TransposeBlockAvx4x4F32x2<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
impl<const FLOP: bool, const FLIP: bool> TransposeBlock<f32>
    for TransposeBlockAvx4x4F32x2<FLOP, FLIP>
{
    #[inline(always)]
    fn transpose_block(&self, src: &[f32], src_stride: usize, dst: &mut [f32], dst_stride: usize) {
        use crate::avx::avx2_transpose_f32x2_4x4;
        unsafe { avx2_transpose_f32x2_4x4::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
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
        if input.len() != input_stride * height {
            return Err(TransposeError::MismatchDimensions);
        }
        if output.len() != output_stride * width {
            return Err(TransposeError::MismatchDimensions);
        }
        if input_stride < width * 2 {
            return Err(TransposeError::MismatchDimensions);
        }
        if output_stride < height * 2 {
            return Err(TransposeError::MismatchDimensions);
        }

        let mut y = 0usize;

        y = transpose_executor::<f32, 2, 2, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
            TransposeBlockSse2x2F32x2::<FLOP, FLIP> {},
        );

        transpose_section::<f32, 2, FLOP, FLIP>(
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

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "unsafe",
    feature = "avx"
))]
#[derive(Copy, Clone, Default)]
struct AvxDefaultExecutor<const FLOP: bool, const FLIP: bool> {}

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
impl<const FLOP: bool, const FLIP: bool> AvxDefaultExecutor<FLOP, FLIP> {
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
        if input.len() != input_stride * height {
            return Err(TransposeError::MismatchDimensions);
        }
        if output.len() != output_stride * width {
            return Err(TransposeError::MismatchDimensions);
        }
        if input_stride < width * 2 {
            return Err(TransposeError::MismatchDimensions);
        }
        if output_stride < height * 2 {
            return Err(TransposeError::MismatchDimensions);
        }

        let mut y = 0usize;

        y = transpose_executor::<f32, 4, 2, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
            TransposeBlockAvx4x4F32x2::<FLOP, FLIP> {},
        );

        y = transpose_executor::<f32, 2, 2, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
            TransposeBlockSse2x2F32x2::<FLOP, FLIP> {},
        );

        transpose_section::<f32, 2, FLOP, FLIP>(
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
impl<const FLOP: bool, const FLIP: bool> TransposeExecutor<f32> for AvxDefaultExecutor<FLOP, FLIP> {
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
        if input.len() != input_stride * height {
            return Err(TransposeError::MismatchDimensions);
        }
        if output.len() != output_stride * width {
            return Err(TransposeError::MismatchDimensions);
        }
        if input_stride < width * 2 {
            return Err(TransposeError::MismatchDimensions);
        }
        if output_stride < height * 2 {
            return Err(TransposeError::MismatchDimensions);
        }

        let mut y = 0usize;

        y = transpose_executor::<f32, 4, 2, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
            TransposeBlockNeon4x4F32x2::<FLOP, FLIP> {},
        );

        y = transpose_executor::<f32, 2, 2, FLOP, FLIP>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
            TransposeBlockNeon2x2F32x2::<FLOP, FLIP> {},
        );

        transpose_section::<f32, 2, FLOP, FLIP>(
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

#[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
fn make_avx2_default_executor(
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Option<Box<dyn TransposeExecutor<f32>>> {
    if std::arch::is_x86_feature_detected!("avx2") {
        return Some(match flip_mode {
            FlipMode::NoFlip => match flop_mode {
                FlopMode::NoFlop => Box::new(AvxDefaultExecutor::<false, false>::default()),
                FlopMode::Flop => Box::new(AvxDefaultExecutor::<true, false>::default()),
            },
            FlipMode::Flip => match flop_mode {
                FlopMode::NoFlop => Box::new(AvxDefaultExecutor::<false, true>::default()),
                FlopMode::Flop => Box::new(AvxDefaultExecutor::<true, true>::default()),
            },
        });
    }
    None
}

pub(crate) fn prepare_f32_cbcr_executor(
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Box<dyn TransposeExecutor<f32>> {
    #[cfg(all(target_arch = "x86_64", feature = "unsafe", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            if let Some(executor) = make_avx2_default_executor(flip_mode, flop_mode) {
                return executor;
            }
        }
    }
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "unsafe",
        feature = "sse"
    ))]
    {
        if std::arch::is_x86_feature_detected!("ssse3") {
            if let Some(executor) = make_ssse3_default_executor(flip_mode, flop_mode) {
                return executor;
            }
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "unsafe", feature = "neon"))]
    {
        make_neon_default_executor(flip_mode, flop_mode)
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "unsafe", feature = "neon")))]
    {
        Box::new(DefaultCbCrExecutor {
            flip_mode,
            flop_mode,
            _phantom: PhantomData,
        })
    }
}
