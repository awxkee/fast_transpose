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
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum TransposeError {
    MismatchDimensions,
    InvalidArraySize,
}

impl Display for TransposeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TransposeError::MismatchDimensions => {
                f.write_str("Source and target dimensions must match")
            }
            TransposeError::InvalidArraySize => {
                f.write_str("Vector size must match provided width and height")
            }
        }
    }
}

impl Error for TransposeError {}

/// Declares whether the image should be reflected across the vertical axis after
/// `transpose_*` function performs a 90 degree counter-clockwise rotation.
///
/// For example, passing the following image to a `transpose_*` function:
///
/// ```text
/// 0 1
/// 2 3
/// ```
///
/// will produce
///
/// - 90 degree counter-clockwise rotation with [`FlipMode::NoFlip`] and [`FlopMode::NoFlop`]:
///   ```text
///   1 3
///   0 2
///   ```
/// - reflection across major diagonal axis with [`FlipMode::NoFlip`] and [`FlopMode::Flop`]:
///   ```text
///   0 2
///   1 3
///   ```
/// - reflection across minor diagonal axis with [`FlipMode::Flip`] and [`FlopMode::NoFlop`]:
///   ```text
///   3 1
///   2 0
///   ```
/// - 270 degree counter-clockwise rotation with [`FlipMode::Flip`] and [`FlopMode::Flop`]:
///   ```text
///   2 0
///   3 1
///   ```
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum FlipMode {
    NoFlip,
    Flip,
}

/// Declares whether the image should be reflected across the horizontal axis after
/// `transpose_*` function performs a 90 degree counter-clockwise rotation.
///
/// For example, passing the following image to a `transpose_*` function:
///
/// ```text
/// 0 1
/// 2 3
/// ```
///
/// will produce
///
/// - 90 degree counter-clockwise rotation with [`FlipMode::NoFlip`] and [`FlopMode::NoFlop`]:
///   ```text
///   1 3
///   0 2
///   ```
/// - reflection across major diagonal axis with [`FlipMode::NoFlip`] and [`FlopMode::Flop`]:
///   ```text
///   0 2
///   1 3
///   ```
/// - reflection across minor diagonal axis with [`FlipMode::Flip`] and [`FlopMode::NoFlop`]:
///   ```text
///   3 1
///   2 0
///   ```
/// - 270 degree counter-clockwise rotation with [`FlipMode::Flip`] and [`FlopMode::Flop`]:
///   ```text
///   2 0
///   3 1
///   ```
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum FlopMode {
    NoFlop,
    Flop,
}
