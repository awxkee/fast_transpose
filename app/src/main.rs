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
use fast_transpose::{
    flip_arbitrary, flip_rgb, flop_arbitrary, flop_rgb, rotate180_rgb, transpose_arbitrary,
    transpose_plane, transpose_plane16, transpose_plane_f32, transpose_plane_f32_with_alpha,
    transpose_plane_with_alpha, transpose_rgb, transpose_rgba, transpose_rgba16, FlipMode,
    FlopMode,
};
use image::{ColorType, DynamicImage, GenericImageView, ImageReader};
use std::time::Instant;

fn main() {
    let img = ImageReader::open("assets/bench.jpg")
        .unwrap()
        .decode()
        .unwrap();

    let j: [u8; 4] = [3, 2, 1, 0];
    let mut arr = vec![];
    for i in 0..16 {
        let scale = i * 4;
        arr.push(j[0] + scale);
        arr.push(j[1] + scale);
        arr.push(j[2] + scale);
        arr.push(j[3] + scale);
    }
    println!("{:?}", arr);

    let img = DynamicImage::ImageRgba8(img.to_rgba8());

    let dimensions = img.dimensions();
    let components = if img.color() == ColorType::Rgb8 { 3 } else { 4 };
    // let img_bytes = img
    //     .to_luma8()
    //     .iter()
    //     .map(|&x| x as f32 / 255.0)
    //     .collect::<Vec<_>>();

    let mut transposed = vec![0.; dimensions.0 as usize * dimensions.1 as usize * 2];
    let mut transposed_rgb = vec![[0u8; 3]; dimensions.0 as usize * dimensions.1 as usize];

    let rgba_bytes = img.to_luma_alpha32f();

    let rgb_set = img
        .as_bytes()
        .chunks_exact(3)
        .map(|x| [x[0], x[1], x[2]])
        .collect::<Vec<_>>();

    let start = Instant::now();

    // transpose::transpose(
    //     &rgb_set,
    //     &mut transposed_rgb,
    //     dimensions.0 as usize,
    //     dimensions.1 as usize,
    // );

    println!("Transpose exec time {:?}", start.elapsed());

    let start = Instant::now();

    // rotate180_rgb(
    //     &rgb_bytes,
    //     img.dimensions().0 as usize * 3,
    //     &mut transposed,
    //     img.dimensions().0 as usize * 3,
    //     dimensions.0 as usize,
    //     dimensions.1 as usize,
    // )
    // .unwrap();

    // for i in 0..10 {
    let start = Instant::now();
    transpose_plane_f32_with_alpha(
        &rgba_bytes,
        dimensions.0 as usize * 2,
        &mut transposed,
        dimensions.1 as usize * 2,
        dimensions.0 as usize,
        dimensions.1 as usize,
        FlipMode::NoFlip,
        FlopMode::NoFlop,
    )
    .unwrap();
    println!("Exec time {:?}", start.elapsed());
    // }

    // transpose_rgba(
    //     &rgba_bytes,
    //     dimensions.0 as usize * 4,
    //     &mut transposed,
    //     dimensions.1 as usize * 4,
    //     dimensions.0 as usize,
    //     dimensions.1 as usize,
    //     FlipMode::NoFlip,
    //     FlopMode::Flop,
    // )
    // .unwrap();

    // transpose_rgba(
    //     &rgb_bytes,
    //     dimensions.0 as usize * 4,
    //     &mut transposed,
    //     dimensions.1 as usize * 4,
    //     dimensions.0 as usize,
    //     dimensions.1 as usize,
    //     FlipMode::Flip,
    //     FlopMode::NoFlop,
    // )
    // .unwrap();

    println!("Exec time {:?}", start.elapsed());

    // transposed = bytemuck::cast_vec(transposed_rgb);

    // let transposed = transposed
    //     .iter()
    //     .map(|&x| (x * 255.) as u8)
    //     .collect::<Vec<_>>();

    // image::save_buffer(
    //     "transposed.jpg",
    //     &transposed,
    //     dimensions.1,
    //     dimensions.0,
    //     image::ExtendedColorType::Rgb8,
    // )
    // .unwrap();

    // if components == 3 {
    //     image::save_buffer(
    //         "transposed.jpg",
    //         &transposed,
    //         dimensions.1,
    //         dimensions.0,
    //         image::ExtendedColorType::Rgb8,
    //     )
    //     .unwrap();
    // } else {
    image::save_buffer(
        "transposed.png",
        &transposed
            .iter()
            .map(|&x| (x * 255.0).round() as u8)
            .collect::<Vec<u8>>(),
        dimensions.1,
        dimensions.0,
        image::ExtendedColorType::La8,
    )
    .unwrap();
    // }
}
