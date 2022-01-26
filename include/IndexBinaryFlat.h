#pragma once
#include <faiss/IndexBinaryFlat.h>
#include "rust/cxx.h"
#include <memory>
#include <vector>

namespace faiss {
    // No constructor support
    std::unique_ptr<IndexBinaryFlat> new_index_binary_flat(int64_t dims);
    const std::vector<uint8_t>& extract_values(const std::unique_ptr<IndexBinaryFlat>& index);
}
