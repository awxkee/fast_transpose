# Fast image transpose

Fast and simple image rotating in Rust with flipping and flopping in-place and rotating by 180 degrees.

## Supports:
- Reflection across vertical axis (`flip_*` functions)
- Reflection across horizontal axis (`flop_*` functions)
- Rotation by 90 degrees counter-clockwise (`transpose_*` functions with `FlipMode::NoFlip` and `FlopMode::NoFlop`)
- Rotation by 180 degrees (`rotate180_*` functions)
- Rotation by 270 degrees counter-clockwise (`transpose_*` functions with `FlipMode::Flip` and `FlopMode::Flop`)

### Adding to project

```bash
cargo add fast_transpose
```

### Rotate RGB image 90 degrees counter-clockwise

```rust
transpose_rgb(
    &img,
    &mut transposed,
    dimensions.0 as usize,
    dimensions.1 as usize,
    FlipMode::NoFlip,
    FlopMode::NoFlop,
)
.unwrap();
```

### Features

Turning off `unsafe` feature will activate `forbid unsafe` mode.

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
