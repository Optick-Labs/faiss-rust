#![feature(generic_const_exprs)]

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

pub struct Assert<const COND: bool> {}
pub trait IsTrue {}
impl IsTrue for Assert<true> { }


pub struct IndexBinarySearchResult<const N: usize> {
    pub distances: Vec<[i32; N]>,
    pub labels: Vec<[i64; N]>
}
impl<const N: usize> IndexBinarySearchResult<N> {
    fn new(results: usize) -> Self {
        let rval = Self {
            distances: vec![[0 as i32; N]; results],
            labels: vec![[0 as i64; N]; results]
        };
        return rval;
    }
}

pub struct IndexBinaryFlat<const D: usize> 
{
    index: cxx::UniquePtr<ffi::IndexBinaryFlat>,
    dims: i64
}
impl<const D: usize> IndexBinaryFlat<D> 
where Assert<{D % 8 == 0}>: IsTrue {
    pub fn new() -> Self {
        Self { index: ffi::new_index_binary_flat(D as i64), dims: D as i64}
    }

    pub fn add(&mut self, data: [u8; D/8]) {
        unsafe {
            self.index.as_mut().map(|x| x.add(1 as i64, &data as *const u8));
        }
    }

    pub fn add_all(&mut self, data: Vec<[u8; D/8]>) {
        unsafe {
            self.index.as_mut().map(|x| x.add(data.len() as i64, &data[0] as *const u8));
        }
    }

    pub fn search<const K: usize>(&self, queries: Vec<[u8; D/8]>) -> IndexBinarySearchResult<K> {
        let mut result: IndexBinarySearchResult<K> = IndexBinarySearchResult::new(queries.len());

        unsafe {
            self.index.search(queries.len() as i64, queries.as_ptr() as *const u8,
            K as i64, &mut result.distances[0] as *mut i32, &mut result.labels[0] as *mut i64);
        }

        return result;
    }

    pub fn display(&self) {
        self.index.display();
    }
}

fn main() {
    let data: [u8; 4] = [0, 255, 240, 1];

    let mut data_vec: Vec<[u8; 1]> = Vec::new();
    data_vec.push([data[0]]);
    data_vec.push([data[1]]);
    data_vec.push([data[2]]);
    data_vec.push([data[3]]);

    let index = ffi::new_index_binary_flat(8);


    let mut index: IndexBinaryFlat<8> = IndexBinaryFlat::new();
    index.add_all(data_vec);
    index.display();
    let result: IndexBinarySearchResult<2> = index.search(vec!([14]));
    println!("1st Result, distance: {} label: {}", result.distances[0][0], result.labels[0][0]);
    println!("2nd Result, distance: {} label: {}", result.distances[0][1], result.labels[0][1]);
}
