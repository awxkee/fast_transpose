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

#![no_main]

use fast_transpose::{
    transpose_plane_f32_with_alpha, transpose_plane_with_alpha, FlipMode, FlopMode,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: (u16, u16)| {
    let width = data.0 as usize;
    let height = data.1 as usize;
    if width > 512 || height > 512 {
        return;
    }
    if width == 0 || height == 0 {
        return;
    }
    let src_data = vec![0.; width * height * 2];
    let mut dst_data = vec![0.; width * height * 2];
    transpose_plane_f32_with_alpha(
        &src_data,
        width * 2,
        &mut dst_data,
        height * 2,
        width,
        height,
        FlipMode::NoFlip,
        FlopMode::NoFlop,
    )
    .unwrap();

    transpose_plane_f32_with_alpha(
        &src_data,
        width * 2,
        &mut dst_data,
        height * 2,
        width,
        height,
        FlipMode::Flip,
        FlopMode::NoFlop,
    )
    .unwrap();

    transpose_plane_f32_with_alpha(
        &src_data,
        width * 2,
        &mut dst_data,
        height * 2,
        width,
        height,
        FlipMode::Flip,
        FlopMode::Flop,
    )
    .unwrap();
});
