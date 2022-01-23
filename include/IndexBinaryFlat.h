#pragma once
#include <faiss/IndexBinaryFlat.h>
#include "rust/cxx.h"

namespace faiss {
    class Blah {
        int a;
    };
    // No constructor support
    std::unique_ptr<IndexBinaryFlat> new_index_binary_flat(int64_t dims);
}
