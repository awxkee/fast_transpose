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
#![forbid(unsafe_code)]
use crate::utils::FlopMode;
use crate::{transpose_arbitrary, FlipMode, TransposeError};

/// Performs plane image transposition
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
/// * `flip_mode`: see [FlipMode]
/// * `flop_mode`: see [FlopMode]
///
/// returns: Result<(), TransposeError>
///
pub fn transpose_plane(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    transpose_arbitrary(matrix, target, width, height, flip_mode, flop_mode)
}

/// Performs plane with alpha image transposition
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
/// * `flip_mode`: see [FlipMode]
/// * `flop_mode`: see [FlopMode]
///
/// returns: Result<(), TransposeError>
///
pub fn transpose_plane_with_alpha(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    let casted_source: &[[u8; 2]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[u8; 2]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    transpose_arbitrary(
        casted_source,
        casted_target,
        width,
        height,
        flip_mode,
        flop_mode,
    )
}

/// Performs RGB image transposition
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
/// * `flip_mode`: see [FlipMode]
/// * `flop_mode`: see [FlopMode]
///
/// returns: Result<(), TransposeError>
///
pub fn transpose_rgb(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    let casted_source: &[[u8; 3]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[u8; 3]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    transpose_arbitrary(
        casted_source,
        casted_target,
        width,
        height,
        flip_mode,
        flop_mode,
    )
}

/// Performs RGBA image transposition
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
/// * `flip_mode`: see [FlipMode]
/// * `flop_mode`: see [FlopMode]
///
/// returns: Result<(), TransposeError>
///
pub fn transpose_rgba(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    let casted_source: &[[u8; 4]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[u8; 4]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    transpose_arbitrary(
        casted_source,
        casted_target,
        width,
        height,
        flip_mode,
        flop_mode,
    )
}
