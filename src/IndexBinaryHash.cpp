#include "faiss-rust/include/IndexBinaryHash.h"
#include "faiss/impl/AuxIndexStructures.h"
#include <memory>
#include <algorithm>

std::unique_ptr<faiss::IndexBinaryMultiHash> faiss::new_index_binary_multi_hash(int64_t dims, int64_t nhash, int64_t b) {
  return std::unique_ptr<faiss::IndexBinaryMultiHash>(new faiss::IndexBinaryMultiHash(dims, nhash, b));
}

const std::vector<uint8_t>& faiss::index_binary_multi_hash_extract_values(const std::unique_ptr<faiss::IndexBinaryMultiHash>& index) {
  return index->storage->xb;
}

void index_binary_multi_hash_range_search(const std::unique_ptr<faiss::IndexBinaryMultiHash>& index, faiss::Index::idx_t n, 
  const uint8_t* query, int radius, faiss::Index::idx_t k, int* distances, faiss::Index::idx_t* labels, faiss::Index::idx_t* sizes) {

  faiss::RangeSearchResult result(n);
  index->range_search(n, query, radius, &result);

  for(size_t q=0; q<n; q++) {
    size_t begin = result.lims[q];
    size_t end = std::min(result.lims[q+1], (size_t)(begin+k));

    for(auto offset=begin; offset<end; offset++) {
      distances[q*k] = result.distances[offset]; // This is a float -> int
      labels[q*k] = result.labels[offset];
      sizes[q] = std::min((faiss::Index::idx_t)(end-begin), (faiss::Index::idx_t)k);
    }
  }
}