#pragma once
#include <faiss/IndexBinaryHash.h>
#include "rust/cxx.h"
#include <memory>
#include <vector>

namespace faiss {
    // No constructor support
    std::unique_ptr<IndexBinaryMultiHash> new_index_binary_multi_hash(int64_t dims, int64_t nhash, int64_t b, int64_t nflip);
    const std::vector<uint8_t>& index_binary_multi_hash_extract_values(const std::unique_ptr<IndexBinaryMultiHash>& index);
    void index_binary_multi_hash_range_search(const std::unique_ptr<faiss::IndexBinaryMultiHash>& index, Index::idx_t n, 
        const uint8_t* query, int radius, Index::idx_t k, int* distances, Index::idx_t* labels, Index::idx_t* sizes);
}
