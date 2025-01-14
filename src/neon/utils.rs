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
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn vrev128_u32(a: uint32x4_t) -> uint32x4_t {
    let rev = vrev64q_u32(a);
    vcombine_u32(vget_high_u32(rev), vget_low_u32(rev))
}

#[inline(always)]
pub(crate) unsafe fn vtrnq_s64_to_u32(a0: uint32x4_t, a1: uint32x4_t) -> uint32x4x2_t {
    let b0 = vreinterpretq_u32_u64(vtrn1q_u64(
        vreinterpretq_u64_u32(a0),
        vreinterpretq_u64_u32(a1),
    ));
    let b1 = vreinterpretq_u32_u64(vtrn2q_u64(
        vreinterpretq_u64_u32(a0),
        vreinterpretq_u64_u32(a1),
    ));
    uint32x4x2_t(b0, b1)
}

#[inline(always)]
pub(crate) unsafe fn xvld1q_u8_u32(ptr: *const u8) -> uint32x4_t {
    vreinterpretq_u32_u8(vld1q_u8(ptr))
}

#[inline(always)]
pub(crate) unsafe fn xvst1q_u8_u32(ptr: *mut u8, a: uint32x4_t) {
    vst1q_u8(ptr, vreinterpretq_u8_u32(a))
}