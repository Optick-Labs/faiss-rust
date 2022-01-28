#pragma once
#include <faiss/IndexBinaryFlat.h>
#include "rust/cxx.h"
#include <memory>
#include <vector>

namespace faiss {
    // No constructor support
    std::unique_ptr<IndexBinaryFlat> new_index_binary_flat(int64_t dims);
    const std::vector<uint8_t>& index_binary_flat_extract_values(const std::unique_ptr<IndexBinaryFlat>& index);
    void index_binary_flat_range_search(const std::unique_ptr<faiss::IndexBinaryFlat>& index, Index::idx_t n, 
        const uint8_t* query, int radius, Index::idx_t k, int* distances, Index::idx_t* labels, Index::idx_t* sizes);
}
