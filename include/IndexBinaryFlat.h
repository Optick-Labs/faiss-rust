#pragma once
#include <faiss/IndexBinaryFlat.h>
#include "rust/cxx.h"

namespace faiss {
    // No constructor support
    std::unique_ptr<IndexBinaryFlat> new_index_binary_flat(int64_t dims);
}
