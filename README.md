TensorFlow Rust Runtime Bindings provides idiomatic [Rust](https://www.rust-lang.org) language
bindings for [TensorFlow](https://www.tensorflow.org) that are linked at runtime. This allows building
without having TensorFlow installed on that machine. Instead it assumes the end device will have a copy of the TensorFlow C library installed in /usr/local/lib, which it will link to at runtime.

This code is a slightly reworked and simplified fork of the [Tensorflow crate](https://crates.io/crates/tensorflow). It is specifically designed for using runtime linking only. All other use
cases should use the original TensorFlow crate.

**Notice:** This project is still under active development and not guaranteed to have a
stable API.
