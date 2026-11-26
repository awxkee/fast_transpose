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
use crate::TransposeError;
use bytemuck::{AnyBitPattern, NoUninit, Pod};

trait Flopper<V: Copy> {
    fn flop(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    );
}

macro_rules! flop_grouped_copy {
    ($input:expr, $input_stride:expr,$output:expr, $output_stride:expr, $width:expr, $cn: expr) => {
        for (dst, src) in $output
            .chunks_exact_mut($output_stride)
            .rev()
            .zip($input.chunks_exact($input_stride))
        {
            let dst = &mut dst[0..$width * $cn];
            let src = &src[0..$width * $cn];
            let dst_casted: &mut [[V; $cn]] = bytemuck::cast_slice_mut(dst);
            let src_casted: &[[V; $cn]] = bytemuck::cast_slice(src);
            for (dst, src) in dst_casted.iter_mut().zip(src_casted.iter()) {
                *dst = *src;
            }
        }
    };
}

#[derive(Debug, Copy, Clone, Default)]
struct CommonGroupedFlopper<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize>
where
    [V; N]: Pod,
{
    _phantom: std::marker::PhantomData<V>,
}

impl<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> Flopper<V>
    for CommonGroupedFlopper<V, N>
where
    [V; N]: Pod,
{
    fn flop(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    ) {
        flop_grouped_copy!(input, input_stride, output, output_stride, width, N);
    }
}

#[derive(Debug, Copy, Clone, Default)]
struct CommonFlopper<V: Copy> {
    _phantom: std::marker::PhantomData<V>,
}

impl<V: Copy> Flopper<V> for CommonFlopper<V> {
    fn flop(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    ) {
        for (dst, src) in output
            .chunks_exact_mut(output_stride)
            .rev()
            .zip(input.chunks_exact(input_stride))
        {
            let dst = &mut dst[0..width];
            let src = &src[0..width];
            for (dst, src) in dst.iter_mut().zip(src.iter()) {
                *dst = *src;
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Default)]
struct FlopperGroupedFactory<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct FlopperAvx2GroupedFactory<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
#[derive(Debug, Copy, Clone, Default)]
struct FlopperAvx512GroupedFactory<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
impl<V: Copy + 'static + Copy + Pod + NoUninit + AnyBitPattern, const N: usize>
    FlopperAvx2GroupedFactory<V, N>
where
    V: Default,
    [V; N]: Pod,
{
    #[target_feature(enable = "avx2")]
    unsafe fn flop_impl(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    ) {
        flop_grouped_copy!(input, input_stride, output, output_stride, width, N);
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
impl<V: Copy + 'static + Copy + Pod + NoUninit + AnyBitPattern, const N: usize>
    FlopperAvx512GroupedFactory<V, N>
where
    V: Default,
    [V; N]: Pod,
{
    #[target_feature(enable = "avx512f")]
    unsafe fn flop_impl(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    ) {
        flop_grouped_copy!(input, input_stride, output, output_stride, width, N);
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
impl<V: Copy + 'static + Copy + Pod + NoUninit + AnyBitPattern, const N: usize> Flopper<V>
    for FlopperAvx512GroupedFactory<V, N>
where
    V: Default,
    [V; N]: Pod,
{
    fn flop(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    ) {
        unsafe { self.flop_impl(input, input_stride, output, output_stride, width) }
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
impl<V: Copy + 'static + Copy + Pod + NoUninit + AnyBitPattern, const N: usize> Flopper<V>
    for FlopperAvx2GroupedFactory<V, N>
where
    V: Default,
    [V; N]: Pod,
{
    fn flop(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    ) {
        unsafe { self.flop_impl(input, input_stride, output, output_stride, width) }
    }
}

impl<V: Copy + 'static + Copy + Pod + NoUninit + AnyBitPattern, const N: usize>
    FlopperGroupedFactory<V, N>
where
    V: Default,
    [V; N]: Pod,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
    fn make_flipper(&self) -> Box<dyn Flopper<V>> {
        #[cfg(feature = "nightly_avx512")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            return Box::new(FlopperAvx512GroupedFactory::<V, N>::default());
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return Box::new(FlopperAvx2GroupedFactory::<V, N>::default());
        }
        Box::new(CommonGroupedFlopper::<V, N>::default())
    }

    #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
    fn make_flipper(&self) -> Box<dyn Flopper<V>> {
        Box::new(CommonGroupedFlopper::<V, N>::default())
    }

    #[cfg(not(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"),
        all(target_arch = "aarch64", feature = "unsafe")
    )))]
    fn make_flipper(&self) -> Box<dyn Flopper<V>> {
        Box::new(CommonGroupedFlopper::<V, N>::default())
    }
}

/// Performs arbitrary flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_arbitrary<V: Copy + Default>(
    input: &[V],
    input_stride: usize,
    output: &mut [V],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    if input_stride.checked_mul(height) != Some(input.len()) {
        return Err(TransposeError::MismatchDimensions);
    }
    if output_stride.checked_mul(height) != Some(output.len()) {
        return Err(TransposeError::MismatchDimensions);
    }
    if input_stride < width {
        return Err(TransposeError::MismatchDimensions);
    }
    if output_stride < width {
        return Err(TransposeError::MismatchDimensions);
    }

    let common_flopper = CommonFlopper::default();
    common_flopper.flop(input, input_stride, output, output_stride, width);

    Ok(())
}

/// Performs arbitrary flopping for groups
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
fn flop_arbitrary_grouped<V: Copy + Default + Pod, const N: usize>(
    input: &[V],
    input_stride: usize,
    output: &mut [V],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError>
where
    [V; N]: Pod,
{
    if input_stride.checked_mul(height) != Some(input.len()) {
        return Err(TransposeError::MismatchDimensions);
    }
    if output_stride.checked_mul(height) != Some(output.len()) {
        return Err(TransposeError::MismatchDimensions);
    }
    if width.checked_mul(N).is_none_or(|v| input_stride < v) {
        return Err(TransposeError::MismatchDimensions);
    }
    if width.checked_mul(N).is_none_or(|v| output_stride < v) {
        return Err(TransposeError::MismatchDimensions);
    }

    let flopper = FlopperGroupedFactory::default().make_flipper();
    flopper.flop(input, input_stride, output, output_stride, width);

    Ok(())
}

/// Performs plane image flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_plane(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary(input, input_stride, output, output_stride, width, height)
}

/// Performs plane with alpha flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_plane_with_alpha(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary_grouped::<u8, 2>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGB image flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_rgb(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary_grouped::<u8, 3>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGBA image flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_rgba(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary_grouped::<u8, 4>(input, input_stride, output, output_stride, width, height)
}

/// Performs plane image flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_plane16(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary(input, input_stride, output, output_stride, width, height)
}

/// Performs plane with alpha image flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_plane16_with_alpha(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary_grouped::<u16, 2>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGB image flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_rgb16(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary_grouped::<u16, 3>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGBA image flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_rgba16(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary_grouped::<u16, 4>(input, input_stride, output, output_stride, width, height)
}

/// Performs plane image flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_plane_f32(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary(input, input_stride, output, output_stride, width, height)
}

/// Performs plane with alpha image flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_plane_f32_with_alpha(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary_grouped::<f32, 2>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGB image flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_rgb_f32(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary_grouped::<f32, 3>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGBA image flopping
///
/// # Arguments
///
/// * `input`: Input data
/// * `input_stride`: Input data stride
/// * `output`: Output data
/// * `output_stride`: Output data stride
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flop_rgba_f32(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flop_arbitrary_grouped::<f32, 4>(input, input_stride, output, output_stride, width, height)
}
