#include "faiss-rust/include/IndexBinaryFlat.h"
#include "faiss/impl/AuxIndexStructures.h"
#include <memory>
#include <iostream>
#include <queue>

std::unique_ptr<faiss::IndexBinaryFlat> faiss::new_index_binary_flat(int64_t dims) {
  return std::unique_ptr<faiss::IndexBinaryFlat>(new faiss::IndexBinaryFlat(dims));
}

const std::vector<uint8_t>& faiss::index_binary_flat_extract_values(const std::unique_ptr<IndexBinaryFlat>& index) {
  return index->xb;
}

void faiss::index_binary_flat_range_search(const std::unique_ptr<faiss::IndexBinaryFlat>& index, faiss::Index::idx_t n, 
  const uint8_t* query, int radius, faiss::Index::idx_t k, int* distances, faiss::Index::idx_t* labels, faiss::Index::idx_t* sizes) {

  faiss::RangeSearchResult result(n);
  index->range_search(n, query, radius, &result);

  for(size_t q=0; q<n; q++) {
    std::priority_queue<std::pair<int, faiss::Index::idx_t> > pq;
    for(size_t r=result.lims[q];r<result.lims[q+1];r++) {
      pq.push(std::make_pair(-result.distances[r], r));
    }

    size_t size = std::min((size_t)k, pq.size());
    sizes[q] = size;
    size_t begin = result.lims[q];
    for(size_t x=0;x<size;x++) {
      auto index = pq.top().second;
      distances[q*k+x] = (int)result.distances[index];
      labels[q*k+x] = result.labels[index];
      pq.pop();
    }
  }
}