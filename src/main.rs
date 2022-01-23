#[cxx::bridge(namespace = "faiss")]
mod ffi {
    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("faiss-rust/include/IndexBinaryFlat.h");

        type IndexBinaryFlat;

        fn new_index_binary_flat(dims: i64) -> UniquePtr<IndexBinaryFlat>;
        fn display(&self);
    }
}

fn main() {
    let index = ffi::new_index_binary_flat(8);
    index.display();
}
