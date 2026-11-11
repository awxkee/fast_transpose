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
use crate::neon::u16x4_2x2::neon_transpose_u64_2x2_impl;
use crate::neon::utils::{xvld1q_u16_u64, xvst1q_u16_u64};
use std::arch::aarch64::*;

#[inline]
pub(crate) fn neon_transpose_u16x4_4x4<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe {
        let r0 = xvld1q_u16_u64(src.get_unchecked(0..).as_ptr());
        let r1 = xvld1q_u16_u64(src.get_unchecked(8..).as_ptr());

        let r2 = xvld1q_u16_u64(src.get_unchecked(src_stride..).as_ptr());
        let r3 = xvld1q_u16_u64(src.get_unchecked(8 + src_stride..).as_ptr());

        let r4 = xvld1q_u16_u64(src.get_unchecked(2 * src_stride..).as_ptr());
        let r5 = xvld1q_u16_u64(src.get_unchecked(8 + 2 * src_stride..).as_ptr());

        let r6 = xvld1q_u16_u64(src.get_unchecked(3 * src_stride..).as_ptr());
        let r7 = xvld1q_u16_u64(src.get_unchecked(8 + 3 * src_stride..).as_ptr());

        // Perform an 4 x 4 matrix transpose by building on top of the existing 2 x 2
        // matrix transpose implementation:
        // [ A B ]^T => [ A^T C^T ]
        // [ C D ]      [ B^T D^T ]

        let mut q0 = neon_transpose_u64_2x2_impl::<FLIP>(uint64x2x2_t(r0, r2));
        let mut q1 = neon_transpose_u64_2x2_impl::<FLIP>(uint64x2x2_t(r1, r3));
        let mut q2 = neon_transpose_u64_2x2_impl::<FLIP>(uint64x2x2_t(r4, r6));
        let mut q3 = neon_transpose_u64_2x2_impl::<FLIP>(uint64x2x2_t(r5, r7));

        if FLIP {
            std::mem::swap(&mut q0, &mut q2);
            std::mem::swap(&mut q1, &mut q3);
        }

        if FLOP {
            xvst1q_u16_u64(dst.get_unchecked_mut(0..).as_mut_ptr(), q0.0);
            xvst1q_u16_u64(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), q0.1);

            xvst1q_u16_u64(dst.get_unchecked_mut(8..).as_mut_ptr(), q2.0);
            xvst1q_u16_u64(dst.get_unchecked_mut(8 + dst_stride..).as_mut_ptr(), q2.1);

            xvst1q_u16_u64(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), q1.0);
            xvst1q_u16_u64(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), q1.1);

            xvst1q_u16_u64(
                dst.get_unchecked_mut(8 + 2 * dst_stride..).as_mut_ptr(),
                q3.0,
            );
            xvst1q_u16_u64(
                dst.get_unchecked_mut(8 + 3 * dst_stride..).as_mut_ptr(),
                q3.1,
            );
        } else {
            xvst1q_u16_u64(
                dst.get_unchecked_mut(8 + 3 * dst_stride..).as_mut_ptr(),
                q2.0,
            );
            xvst1q_u16_u64(
                dst.get_unchecked_mut(8 + 2 * dst_stride..).as_mut_ptr(),
                q2.1,
            );

            xvst1q_u16_u64(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), q0.0);
            xvst1q_u16_u64(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), q0.1);

            xvst1q_u16_u64(dst.get_unchecked_mut(8 + dst_stride..).as_mut_ptr(), q3.0);
            xvst1q_u16_u64(dst.get_unchecked_mut(8..).as_mut_ptr(), q3.1);

            xvst1q_u16_u64(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), q1.0);
            xvst1q_u16_u64(dst.get_unchecked_mut(0..).as_mut_ptr(), q1.1);
        }
    }
}
