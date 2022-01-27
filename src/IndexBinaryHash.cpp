#include "faiss-rust/include/IndexBinaryHash.h"
#include <memory>

std::unique_ptr<faiss::IndexBinaryMultiHash> faiss::new_index_binary_multi_hash(int64_t dims, int64_t nhash, int64_t b) {
  return std::unique_ptr<faiss::IndexBinaryMultiHash>(new faiss::IndexBinaryMultiHash(dims, nhash, b));
}

const std::vector<uint8_t>& faiss::index_binary_multi_hash_extract_values(const std::unique_ptr<IndexBinaryMultiHash>& index) {
  return index->storage->xb;
}

void index_binary_multi_hash_range_search(const std::unique_ptr<IndexBinaryMultiHash>& index, Index::idx_t n, 
  const uint8_t* query, int radius, Index::idx_t k, int* distances, Index::idx_t* labels, Index::idx_t* sizes) {

  faiss::RangeSearchResult result;
  index->range_search(n, query, radius, &result);

  for(auto q=0; q<result.nq; q++) {
    size_t begin = lims[q];
    size_t end = std::min(limts[q+1], begin+k);

    for(auto offset=begin; offset<end; offset++) {
      *(distances+q*k) = result.distances[offset]; // This is a float -> int
      *(labels+q*k) = result.labels[offset]
      *(sizes+q) = std::min(end-begin, k);
    }
  }
}