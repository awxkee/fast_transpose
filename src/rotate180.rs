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

trait Rotator<V: Copy> {
    fn rotate(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    );
}

macro_rules! rotate_flatten {
    ($input:expr, $input_stride:expr,$output:expr, $output_stride:expr, $width:expr) => {
        for (dst, src) in $output
            .chunks_exact_mut($output_stride)
            .rev()
            .zip($input.chunks_exact($input_stride))
        {
            let dst = &mut dst[0..$width];
            let src = &src[0..$width];
            for (dst, src) in dst.iter_mut().rev().zip(src.iter()) {
                *dst = *src;
            }
        }
    };
}

macro_rules! rotate_grouped_copy {
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
            for (dst, src) in dst_casted.iter_mut().rev().zip(src_casted.iter()) {
                *dst = *src;
            }
        }
    };
}

#[derive(Debug, Copy, Clone, Default)]
struct CommonGroupedFlipper<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

impl<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> Rotator<V>
    for CommonGroupedFlipper<V, N>
where
    [V; N]: Pod,
{
    #[inline(always)]
    fn rotate(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    ) {
        rotate_grouped_copy!(input, input_stride, output, output_stride, width, N);
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct SSSE3GroupedRotator<V: Copy, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

macro_rules! define_rotator_grouped_x86 {
    ($flipper_type:ident, $feature: literal) => {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
        impl<V: Copy + Pod, const N: usize> $flipper_type<V, N>
        where
            [V; N]: Pod,
        {
            #[target_feature(enable = $feature)]
            unsafe fn rotate_impl(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                rotate_grouped_copy!(input, input_stride, output, output_stride, width, N);
            }
        }

        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
        impl<V: Copy + Pod, const N: usize> Rotator<V> for $flipper_type<V, N>
        where
            [V; N]: Pod,
        {
            fn rotate(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                unsafe { self.rotate_impl(input, input_stride, output, output_stride, width) }
            }
        }
    };
}

macro_rules! define_rotator_grouped_aarch64 {
    ($flipper_type: ident, $feature: literal) => {
        #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
        impl<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> $flipper_type<V, N>
        where
            [V; N]: Pod,
        {
            #[target_feature(enable = $feature)]
            unsafe fn rotate_impl(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                rotate_grouped_copy!(input, input_stride, output, output_stride, width, N);
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
        impl<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> Rotator<V>
            for $flipper_type<V, N>
        where
            [V; N]: Pod,
        {
            fn rotate(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                unsafe { self.rotate_impl(input, input_stride, output, output_stride, width) }
            }
        }
    };
}

define_rotator_grouped_x86!(SSSE3GroupedRotator, "ssse3");

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct Sse41GroupedRotator<V: Copy, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

define_rotator_grouped_x86!(Sse41GroupedRotator, "sse4.1");

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct Avx2GroupedRotator<V: Copy, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

define_rotator_grouped_x86!(Avx2GroupedRotator, "avx2");

#[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct SveGroupedRotator<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

define_rotator_grouped_aarch64!(SveGroupedRotator, "sve2");

#[derive(Debug, Copy, Clone, Default)]
struct RotatorGroupedFactory<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

impl<V: Copy + 'static + Copy + Pod + NoUninit + AnyBitPattern, const N: usize>
    RotatorGroupedFactory<V, N>
where
    V: Default,
    [V; N]: Pod,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
    fn make_rotator(&self) -> Box<dyn Rotator<V>> {
        if std::arch::is_x86_feature_detected!("avx2") {
            return Box::new(Avx2GroupedRotator::<V, N>::default());
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return Box::new(Sse41GroupedRotator::<V, N>::default());
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            return Box::new(SSSE3GroupedRotator::<V, N>::default());
        }
        Box::new(CommonGroupedFlipper::<V, N>::default())
    }

    #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
    fn make_rotator(&self) -> Box<dyn Rotator<V>> {
        if std::arch::is_aarch64_feature_detected!("sve2") {
            return Box::new(SveGroupedRotator::<V, N>::default());
        }
        Box::new(CommonGroupedFlipper::<V, N>::default())
    }

    #[cfg(not(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"),
        all(target_arch = "aarch64", feature = "unsafe")
    )))]
    fn make_rotator(&self) -> Box<dyn Rotator<V>> {
        Box::new(CommonGroupedFlipper::<V, N>::default())
    }
}

macro_rules! define_rotator_aarch64 {
    ($flipper_type: ident, $feature: literal) => {
        #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
        impl<V: Copy + Default> $flipper_type<V> {
            #[target_feature(enable = $feature)]
            unsafe fn rotate_impl(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                rotate_flatten!(input, input_stride, output, output_stride, width);
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
        impl<V: Copy + Default> Rotator<V> for $flipper_type<V> {
            fn rotate(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                unsafe { self.rotate_impl(input, input_stride, output, output_stride, width) }
            }
        }
    };
}

macro_rules! define_rotator_x86 {
    ($flipper_type: ident, $feature: literal) => {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
        impl<V: Copy + Default> $flipper_type<V> {
            #[target_feature(enable = $feature)]
            unsafe fn rotate_impl(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                rotate_flatten!(input, input_stride, output, output_stride, width);
            }
        }

        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
        impl<V: Copy + Default> Rotator<V> for $flipper_type<V> {
            fn rotate(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                unsafe { self.rotate_impl(input, input_stride, output, output_stride, width) }
            }
        }
    };
}

#[derive(Debug, Copy, Clone, Default)]
struct CommonRotator<V: Copy + Default> {
    _phantom: std::marker::PhantomData<V>,
}

impl<V: Copy + Default> Rotator<V> for CommonRotator<V> {
    #[inline(always)]
    fn rotate(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    ) {
        rotate_flatten!(input, input_stride, output, output_stride, width);
    }
}

#[derive(Debug, Copy, Clone, Default)]
struct RotatorFactory<V: Copy + Default + 'static> {
    _phantom: std::marker::PhantomData<V>,
}

#[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct SveRotator<V: Copy + Default + 'static> {
    _phantom: std::marker::PhantomData<V>,
}

define_rotator_aarch64!(SveRotator, "sve2");

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct Avx2Rotator<V: Copy> {
    _phantom: std::marker::PhantomData<V>,
}

define_rotator_x86!(Avx2Rotator, "avx2");

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct Sse41Rotator<V: Copy> {
    _phantom: std::marker::PhantomData<V>,
}

define_rotator_x86!(Sse41Rotator, "sse4.1");

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct SSSE3Rotator<V: Copy> {
    _phantom: std::marker::PhantomData<V>,
}

define_rotator_x86!(SSSE3Rotator, "ssse3");

impl<V: Copy + Default + 'static> RotatorFactory<V> {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
    fn make_rotator(&self) -> Box<dyn Rotator<V>> {
        if std::arch::is_x86_feature_detected!("avx2") {
            return Box::new(Avx2Rotator::<V>::default());
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return Box::new(Sse41Rotator::<V>::default());
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            return Box::new(SSSE3Rotator::<V>::default());
        }
        Box::new(CommonRotator::<V>::default())
    }

    #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
    fn make_rotator(&self) -> Box<dyn Rotator<V>> {
        if std::arch::is_aarch64_feature_detected!("sve2") {
            return Box::new(SveRotator::<V>::default());
        }
        Box::new(CommonRotator::<V>::default())
    }

    #[cfg(not(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"),
        all(target_arch = "aarch64", feature = "unsafe")
    )))]
    fn make_rotator(&self) -> Box<dyn Rotator<V>> {
        Box::new(CommonRotator::<V>::default())
    }
}

/// Performs arbitrary rotating by 180
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
pub fn rotate180_arbitrary<V: Copy + Default + 'static>(
    input: &[V],
    input_stride: usize,
    output: &mut [V],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    if input.len() != output.len() {
        return Err(TransposeError::MismatchDimensions);
    }
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

    let flipper_factory = RotatorFactory::<V>::default();
    let flipper = flipper_factory.make_rotator();
    flipper.rotate(input, input_stride, output, output_stride, width);

    Ok(())
}

/// Performs arbitrary rotating by 180
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
fn rotate180_arbitrary_image<V: Copy + Default + 'static + Pod, const N: usize>(
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

    let flipper_factory = RotatorGroupedFactory::<V, N>::default();
    let flipper = flipper_factory.make_rotator();
    flipper.rotate(input, input_stride, output, output_stride, width);

    Ok(())
}

/// Performs plane image rotating by 180
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
pub fn rotate180_plane(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary(input, input_stride, output, output_stride, width, height)
}

/// Performs plane with alpha rotating by 180
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
pub fn rotate180_plane_with_alpha(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary_image::<u8, 2>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGB image rotating by 180
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
pub fn rotate180_rgb(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary_image::<u8, 3>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGBA image rotating by 180
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
pub fn rotate180_rgba(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary_image::<u8, 4>(input, input_stride, output, output_stride, width, height)
}

/// Performs plane image rotating by 180
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
pub fn rotate180_plane16(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary(input, input_stride, output, output_stride, width, height)
}

/// Performs plane with alpha image rotating by 180
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
pub fn rotate180_plane16_with_alpha(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary_image::<u16, 2>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGB image rotating by 180
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
pub fn rotate180_rgb16(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary_image::<u16, 3>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGBA image rotating by 180
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
pub fn rotate180_rgba16(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary_image::<u16, 4>(input, input_stride, output, output_stride, width, height)
}

/// Performs plane image rotating by 180
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
pub fn rotate180_plane_f32(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary(input, input_stride, output, output_stride, width, height)
}

/// Performs plane with alpha image rotating by 180
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
pub fn rotate180_plane_f32_with_alpha(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary_image::<f32, 2>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGB image rotating by 180
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
pub fn rotate180_rgb_f32(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary_image::<f32, 3>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGBA image rotating by 180
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
pub fn rotate180_rgba_f32(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    rotate180_arbitrary_image::<f32, 4>(input, input_stride, output, output_stride, width, height)
}
