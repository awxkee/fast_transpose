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

trait Flipper<V: Copy> {
    fn flip(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    );
}

macro_rules! reverse_copy_flatten {
    ($input:expr, $input_stride:expr,$output:expr, $output_stride:expr, $width:expr) => {
        for (dst, src) in $output
            .chunks_exact_mut($output_stride)
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

macro_rules! reverse_copy {
    ($input:expr, $input_stride:expr,$output:expr, $output_stride:expr, $width:expr, $cn: expr) => {
        for (dst, src) in $output
            .chunks_exact_mut($output_stride)
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

impl<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> Flipper<V>
    for CommonGroupedFlipper<V, N>
where
    [V; N]: Pod,
{
    #[inline(always)]
    fn flip(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    ) {
        reverse_copy!(input, input_stride, output, output_stride, width, N);
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct SSSE3GroupedFlipper<V: Copy, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

macro_rules! define_flipper_grouped_x86 {
    ($flipper_type:ident, $feature: literal) => {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
        impl<V: Copy + Pod, const N: usize> $flipper_type<V, N>
        where
            [V; N]: Pod,
        {
            #[target_feature(enable = $feature)]
            unsafe fn flip_impl(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                reverse_copy!(input, input_stride, output, output_stride, width, N);
            }
        }

        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
        impl<V: Copy + Pod, const N: usize> Flipper<V> for $flipper_type<V, N>
        where
            [V; N]: Pod,
        {
            fn flip(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                unsafe { self.flip_impl(input, input_stride, output, output_stride, width) }
            }
        }
    };
}

macro_rules! define_flipper_grouped_aarch64 {
    ($flipper_type: ident, $feature: literal) => {
        #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
        impl<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> $flipper_type<V, N>
        where
            [V; N]: Pod,
        {
            #[target_feature(enable = $feature)]
            unsafe fn flip_impl(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                reverse_copy!(input, input_stride, output, output_stride, width, N);
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
        impl<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> Flipper<V>
            for $flipper_type<V, N>
        where
            [V; N]: Pod,
        {
            fn flip(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                unsafe { self.flip_impl(input, input_stride, output, output_stride, width) }
            }
        }
    };
}

define_flipper_grouped_x86!(SSSE3GroupedFlipper, "ssse3");

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct Sse41GroupedFlipper<V: Copy, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

define_flipper_grouped_x86!(Sse41GroupedFlipper, "sse4.1");

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct Avx2GroupedFlipper<V: Copy, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

define_flipper_grouped_x86!(Avx2GroupedFlipper, "avx2");

#[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct SveGroupedFlipper<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

define_flipper_grouped_aarch64!(SveGroupedFlipper, "sve2");

#[derive(Debug, Copy, Clone, Default)]
struct FlipperGroupedFactory<V: Copy + Pod + NoUninit + AnyBitPattern, const N: usize> {
    _phantom: std::marker::PhantomData<V>,
}

impl<V: Copy + 'static + Copy + Pod + NoUninit + AnyBitPattern, const N: usize>
    FlipperGroupedFactory<V, N>
where
    V: Default,
    [V; N]: Pod,
{
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
    fn make_flipper(&self) -> Box<dyn Flipper<V>> {
        if std::arch::is_x86_feature_detected!("avx2") {
            return Box::new(Avx2GroupedFlipper::<V, N>::default());
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return Box::new(Sse41GroupedFlipper::<V, N>::default());
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            return Box::new(SSSE3GroupedFlipper::<V, N>::default());
        }
        Box::new(CommonGroupedFlipper::<V, N>::default())
    }

    #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
    fn make_flipper(&self) -> Box<dyn Flipper<V>> {
        if std::arch::is_aarch64_feature_detected!("sve2") {
            return Box::new(SveGroupedFlipper::<V, N>::default());
        }
        Box::new(CommonGroupedFlipper::<V, N>::default())
    }

    #[cfg(not(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"),
        all(target_arch = "aarch64", feature = "unsafe")
    )))]
    fn make_flipper(&self) -> Box<dyn Flipper<V>> {
        Box::new(CommonGroupedFlipper::<V, N>::default())
    }
}

macro_rules! define_flipper_aarch64 {
    ($flipper_type: ident, $feature: literal) => {
        #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
        impl<V: Copy + Default> $flipper_type<V> {
            #[target_feature(enable = $feature)]
            unsafe fn flip_impl(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                reverse_copy_flatten!(input, input_stride, output, output_stride, width);
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
        impl<V: Copy + Default> Flipper<V> for $flipper_type<V> {
            fn flip(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                unsafe { self.flip_impl(input, input_stride, output, output_stride, width) }
            }
        }
    };
}

macro_rules! define_flipper_x86 {
    ($flipper_type: ident, $feature: literal) => {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
        impl<V: Copy + Default> $flipper_type<V> {
            #[target_feature(enable = $feature)]
            unsafe fn flip_impl(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                reverse_copy_flatten!(input, input_stride, output, output_stride, width);
            }
        }

        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
        impl<V: Copy + Default> Flipper<V> for $flipper_type<V> {
            fn flip(
                &self,
                input: &[V],
                input_stride: usize,
                output: &mut [V],
                output_stride: usize,
                width: usize,
            ) {
                unsafe { self.flip_impl(input, input_stride, output, output_stride, width) }
            }
        }
    };
}

#[derive(Debug, Copy, Clone, Default)]
struct CommonFlipper<V: Copy + Default> {
    _phantom: std::marker::PhantomData<V>,
}

impl<V: Copy + Default> Flipper<V> for CommonFlipper<V> {
    #[inline(always)]
    fn flip(
        &self,
        input: &[V],
        input_stride: usize,
        output: &mut [V],
        output_stride: usize,
        width: usize,
    ) {
        reverse_copy_flatten!(input, input_stride, output, output_stride, width);
    }
}

#[derive(Debug, Copy, Clone, Default)]
struct FlipperFactory<V: Copy + Default + 'static> {
    _phantom: std::marker::PhantomData<V>,
}

#[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct SveFlipper<V: Copy + Default + 'static> {
    _phantom: std::marker::PhantomData<V>,
}

define_flipper_aarch64!(SveFlipper, "sve2");

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct Avx2Flipper<V: Copy> {
    _phantom: std::marker::PhantomData<V>,
}

define_flipper_x86!(Avx2Flipper, "avx2");

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct Sse41Flipper<V: Copy> {
    _phantom: std::marker::PhantomData<V>,
}

define_flipper_x86!(Sse41Flipper, "sse4.1");

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
#[derive(Debug, Copy, Clone, Default)]
struct SSSE3Flipper<V: Copy> {
    _phantom: std::marker::PhantomData<V>,
}

define_flipper_x86!(SSSE3Flipper, "ssse3");

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
#[derive(Debug, Copy, Clone, Default)]
struct Avx512Flipper<V: Copy> {
    _phantom: std::marker::PhantomData<V>,
}

#[cfg(feature = "nightly_avx512")]
define_flipper_x86!(Avx512Flipper, "avx512bw");

impl<V: Copy + Default + 'static> FlipperFactory<V> {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
    fn make_flipper(&self) -> Box<dyn Flipper<V>> {
        #[cfg(feature = "nightly_avx512")]
        if std::arch::is_x86_feature_detected!("avx512bw") {
            return Box::new(Avx512Flipper::<V>::default());
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return Box::new(Avx2Flipper::<V>::default());
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return Box::new(Sse41Flipper::<V>::default());
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            return Box::new(SSSE3Flipper::<V>::default());
        }
        Box::new(CommonFlipper::<V>::default())
    }

    #[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
    fn make_flipper(&self) -> Box<dyn Flipper<V>> {
        if std::arch::is_aarch64_feature_detected!("sve2") {
            return Box::new(SveFlipper::<V>::default());
        }
        Box::new(CommonFlipper::<V>::default())
    }

    #[cfg(not(any(
        all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"),
        all(target_arch = "aarch64", feature = "unsafe")
    )))]
    fn make_flipper(&self) -> Box<dyn Flipper<V>> {
        Box::new(CommonFlipper::<V>::default())
    }
}

/// Performs arbitrary flipping
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
pub fn flip_arbitrary<V: Copy + Default + 'static>(
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

    let flipper_factory = FlipperFactory::<V>::default();
    let flipper = flipper_factory.make_flipper();
    flipper.flip(input, input_stride, output, output_stride, width);

    Ok(())
}

/// Performs arbitrary flipping
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
fn flip_arbitrary_image<V: Copy + Default + 'static + Pod, const N: usize>(
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

    let flipper_factory = FlipperGroupedFactory::<V, N>::default();
    let flipper = flipper_factory.make_flipper();
    flipper.flip(input, input_stride, output, output_stride, width);

    Ok(())
}

/// Performs plane image flipping
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
pub fn flip_plane(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary(input, input_stride, output, output_stride, width, height)
}

/// Performs plane with alpha flipping
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
pub fn flip_plane_with_alpha(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary_image::<u8, 2>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGB image flipping
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
pub fn flip_rgb(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary_image::<u8, 3>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGBA image flipping
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
pub fn flip_rgba(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary_image::<u8, 4>(input, input_stride, output, output_stride, width, height)
}

/// Performs plane image flipping
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
pub fn flip_plane16(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary(input, input_stride, output, output_stride, width, height)
}

/// Performs plane with alpha image flipping
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
pub fn flip_plane16_with_alpha(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary_image::<u16, 2>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGB image flipping
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
pub fn flip_rgb16(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary_image::<u16, 3>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGBA image flipping
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
pub fn flip_rgba16(
    input: &[u16],
    input_stride: usize,
    output: &mut [u16],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary_image::<u16, 4>(input, input_stride, output, output_stride, width, height)
}

/// Performs plane image flipping
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
pub fn flip_plane_f32(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary(input, input_stride, output, output_stride, width, height)
}

/// Performs plane with alpha image flipping
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
pub fn flip_plane_f32_with_alpha(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary_image::<f32, 2>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGB image flipping
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
pub fn flip_rgb_f32(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary_image::<f32, 3>(input, input_stride, output, output_stride, width, height)
}

/// Performs RGBA image flipping
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
pub fn flip_rgba_f32(
    input: &[f32],
    input_stride: usize,
    output: &mut [f32],
    output_stride: usize,
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary_image::<f32, 4>(input, input_stride, output, output_stride, width, height)
}
