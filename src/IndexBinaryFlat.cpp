#include "faiss-rust/include/IndexBinaryFlat.h"
#include <memory>

std::unique_ptr<faiss::IndexBinaryFlat> faiss::new_index_binary_flat(int64_t dims) {
  return std::unique_ptr<faiss::IndexBinaryFlat>(new faiss::IndexBinaryFlat(dims));
}
