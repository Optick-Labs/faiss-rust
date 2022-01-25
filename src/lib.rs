#![allow(incomplete_features, dead_code)]

use std::sync::RwLock;

#[cxx::bridge(namespace = "faiss")]
mod ffi {
    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("faiss-rust/include/IndexBinaryFlat.h");

        type IndexBinaryFlat;

        fn new_index_binary_flat(dims: i64) -> UniquePtr<IndexBinaryFlat>;

        unsafe fn add(self: Pin<&mut Self>, n: i64, vals: *const u8);
        unsafe fn search(&self, n: i64, queries: *const u8, k: i64, distances: *mut i32, labels: *mut i64);
        fn display(&self);
    }
}

// Thread safety of FAISS is discussed here: 
// https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls#performance-of-search
unsafe impl Send for ffi::IndexBinaryFlat {}
unsafe impl Sync for ffi::IndexBinaryFlat {}

pub struct Assert<const COND: bool> {}
pub trait IsTrue {}
impl IsTrue for Assert<true> { }

#[derive(Clone)]
pub struct IndexBinarySearchQueryResult {
    pub distance: i32,
    pub index: i64,
    pub label: String
}
impl IndexBinarySearchQueryResult {
    fn new() -> Self {
        Self {
            distance: -1,
            index: -1,
            label: String::new()
        }
    }
}

pub struct IndexBinarySearchResult {
    pub queries: Vec<Vec<IndexBinarySearchQueryResult>>
}
impl IndexBinarySearchResult {
    fn new(k: usize, results: usize) -> Self {
        Self {
            queries: (0..results)
                .map(|_i| vec![IndexBinarySearchQueryResult::new(); k] as Vec<IndexBinarySearchQueryResult>)
                .collect()
        }
    }
}

// NOTE: We're handling Send/Sync via a RwLock where FAISS has guaranteed thread safety
// on read operations. This can't be guaranteed by Rust.
pub struct IndexBinaryFlat<const D: usize> 
{
    dims: i64,
    index: RwLock<cxx::UniquePtr<ffi::IndexBinaryFlat>>,
    ids: RwLock<Vec<String>>
}
impl<const D: usize> IndexBinaryFlat<D> 
// where Assert<{D % 8 == 0}>: IsTrue // TODO: Feature available in nightly
{
    pub fn new() -> Self {
        let index = ffi::new_index_binary_flat(D as i64);
        Self { 
            dims: D as i64,
            index: RwLock::new(index),
            ids: RwLock::new(Vec::new())
        }
    }

    pub fn add(&mut self, data: [u8; D], id: String) {
        unsafe {
            // TODO: This can panic if we fail in the write at some point. Crash more gracefully?
            self.index.write().unwrap().as_mut().map(|x| x.add(1 as i64, &data as *const u8));
        }
        self.ids.write().unwrap().push(id);
    }

    pub fn add_all(&mut self, data: &Vec<[u8; D]>, ids: &mut Vec<String>) {
        if data.len() == 0 {
            return;
        }
        
        unsafe {
            // TODO: This can panic if we fail in the write at some point. Crash more gracefully?
            self.index.write().unwrap().as_mut().map(|x| x.add(data.len() as i64, &data[0] as *const u8));
        }
        self.ids.write().unwrap().append(ids);
    }

    pub fn search(&self, queries: &Vec<[u8; D]>, k: usize) -> IndexBinarySearchResult {
        let mut result= IndexBinarySearchResult::new(k, queries.len());
        let mut distances: Vec<i32> = vec![-1; k*queries.len()];
        let mut indexes: Vec<i64> = vec![-1; k*queries.len()];

        unsafe {
            self.index.read().unwrap().search(queries.len() as i64, queries.as_ptr() as *const u8,
            k as i64, &mut distances[0] as *mut i32, &mut indexes[0] as *mut i64);
        }

        let ids = self.ids.read().unwrap();

        for r in 0..queries.len() {
            for rk in 0..k {
                result.queries[r][rk] = IndexBinarySearchQueryResult {
                    distance: distances[r*queries.len()+rk],
                    index: indexes[r*queries.len()+rk],
                    label: ids[r*queries.len()+rk].clone()
                }
            }
        }

        return result;
    }

    pub fn display(&self) {
        self.index.read().unwrap().display();
    }
}