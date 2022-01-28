fn main() {
    println!("cargo:rustc-link-search=/usr/local/lib");
    println!("cargo:rustc-link-lib=faiss");
    
    cxx_build::bridge("src/lib.rs")
        .files(["src/IndexBinaryFlat.cpp", "src/IndexBinaryHash.cpp"])
        .flag_if_supported("-std=c++14")
        .compile("cxxbridge-demo");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/blobstore.cc");
    println!("cargo:rerun-if-changed=include/blobstore.h");
}
