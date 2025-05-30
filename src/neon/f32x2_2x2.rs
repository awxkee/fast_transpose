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

use crate::neon::utils::vrev128q_f64;
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn neon_transpose_f32x2_2x2_impl<const FLIP: bool>(
    v0: float32x4x2_t,
) -> float32x4x2_t {
    let l = vreinterpretq_f32_f64(vtrn1q_f64(
        vreinterpretq_f64_f32(v0.0),
        vreinterpretq_f64_f32(v0.1),
    ));
    let h = vreinterpretq_f32_f64(vtrn2q_f64(
        vreinterpretq_f64_f32(v0.0),
        vreinterpretq_f64_f32(v0.1),
    ));

    if FLIP {
        float32x4x2_t(
            vreinterpretq_f32_f64(vrev128q_f64(vreinterpretq_f64_f32(l))),
            vreinterpretq_f32_f64(vrev128q_f64(vreinterpretq_f64_f32(h))),
        )
    } else {
        float32x4x2_t(l, h)
    }
}

#[inline]
pub(crate) fn neon_transpose_f32x2_2x2<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld1q_f32(src.get_unchecked(0..).as_ptr());
        let row1 = vld1q_f32(src.get_unchecked(src_stride..).as_ptr());

        let v0 = neon_transpose_f32x2_2x2_impl::<FLIP>(float32x4x2_t(row0, row1));

        if FLOP {
            vst1q_f32(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.0);
            vst1q_f32(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.1);
        } else {
            vst1q_f32(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.0);
            vst1q_f32(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.1);
        }
    }
}
