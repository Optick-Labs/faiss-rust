#![allow(incomplete_features, dead_code)]

use std::sync::RwLock;

use serde::{Serializer, ser::SerializeStruct, Deserializer, de::Visitor};
use serde_derive::{Serialize, Deserialize};

#[cxx::bridge(namespace = "faiss")]
mod ffi {
    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("faiss-rust/include/IndexBinaryFlat.h");

        type IndexBinaryFlat;

        fn new_index_binary_flat(dims: i64) -> UniquePtr<IndexBinaryFlat>;
        fn index_binary_flat_extract_values(index: &UniquePtr<IndexBinaryFlat>) -> &CxxVector<u8>;
        unsafe fn index_binary_flat_range_search(index: &UniquePtr<IndexBinaryFlat>, n: i64, queries: *const u8,
            radius: i32, k: i64, distances: *mut i32, labels: *mut i64, sizes: *mut i64);

        unsafe fn add(self: Pin<&mut Self>, n: i64, vals: *const u8);
        unsafe fn search(&self, n: i64, queries: *const u8, k: i64, distances: *mut i32, labels: *mut i64);

        fn display(&self);
    }
    unsafe extern "C++" {
        include!("faiss-rust/include/IndexBinaryHash.h");

        type IndexBinaryMultiHash;

        fn new_index_binary_multi_hash(dims: i64, nhash: i64, b: i64, nflip: i64) -> UniquePtr<IndexBinaryMultiHash>;
        fn index_binary_multi_hash_extract_values(index: &UniquePtr<IndexBinaryMultiHash>) -> &CxxVector<u8>;
        
        unsafe fn add(self: Pin<&mut Self>, n: i64, vals: *const u8);
        unsafe fn search(&self, n: i64, queries: *const u8, k: i64, distances: *mut i32, labels: *mut i64);
        unsafe fn index_binary_multi_hash_range_search(index: &UniquePtr<IndexBinaryMultiHash>, n: i64, queries: *const u8,
            radius: i32, k: i64, distances: *mut i32, labels: *mut i64, sizes: *mut i64);

        fn display(&self);
    }
}

// Thread safety of FAISS is discussed here: 
// https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls#performance-of-search
unsafe impl Send for ffi::IndexBinaryFlat {}
unsafe impl Sync for ffi::IndexBinaryFlat {}

unsafe impl Send for ffi::IndexBinaryMultiHash {}
unsafe impl Sync for ffi::IndexBinaryMultiHash {}

pub struct Assert<const COND: bool> {}
pub trait IsTrue {}
impl IsTrue for Assert<true> { }

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct IndexBinaryEntry<const D: usize> 
{
    #[serde(with = "serde_arrays")]
    pub hash: [u8; D],
    pub label: usize
}

#[derive(Clone)]
pub struct IndexBinarySearchQueryResult {
    pub distance: i32,
    pub index: usize
}
impl IndexBinarySearchQueryResult {
    fn new() -> Self {
        Self {
            distance: -1,
            index: 0
        }
    }
}

pub struct IndexBinarySearchResult {
    pub queries: Vec<Vec<IndexBinarySearchQueryResult>>
}
impl IndexBinarySearchResult {
    fn new(k: usize, results: usize) -> Self {
        Self {
            queries: vec![Vec::with_capacity(k); results]
        }
    }
}

impl<const D: usize> serde::Serialize for IndexBinaryFlat<D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        let mut state = serializer.serialize_struct("snapshot", 1)?;

        let i = self.index.read().unwrap();
        let values = ffi::index_binary_flat_extract_values(&(*i));
        unsafe {
            let addr = values.get_unchecked(0) as *const u8;
            // TODO: This makes a copy, do not like
            let slice = std::slice::from_raw_parts(addr, values.len()).to_vec();
            state.serialize_field("data", &slice)?;
        }
        state.end()
    }
}

impl<'de, const N: usize> serde::Deserialize<'de> for IndexBinaryFlat<N> {
    fn deserialize<D>(deserializer: D) -> Result<IndexBinaryFlat<N>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct IndexBinaryFlatVisitor<const N: usize>;

        impl<'de, const N: usize> Visitor<'de> for IndexBinaryFlatVisitor<N> {
            type Value = IndexBinaryFlat<N>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a binary file with index data")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                    A: serde::de::SeqAccess<'de>, {
                // TODO: This loads too much data, do not like
                let data = seq.next_element::<Vec<u8>>()?.unwrap();
                let mut index: IndexBinaryFlat<N> = IndexBinaryFlat::new();
                index.add_all_raw(&data);
                Ok(index)
            }
        }

        const FIELDS: &'static [&'static str] = &["data"];
        deserializer.deserialize_struct("snapshot", FIELDS, IndexBinaryFlatVisitor::<N>)
    }
}

// NOTE: We're handling Send/Sync via a RwLock where FAISS has guaranteed thread safety
// on read operations. This can't be guaranteed by Rust.
pub struct IndexBinaryFlat<const D: usize> 
{
    dims: i64,
    index: RwLock<cxx::UniquePtr<ffi::IndexBinaryFlat>>
}
impl<const D: usize> IndexBinaryFlat<D> 
// where Assert<{D % 8 == 0}>: IsTrue // TODO: Feature available in nightly
{
    pub fn new() -> Self {
        let index = ffi::new_index_binary_flat((D*8) as i64);
        Self { 
            dims: D as i64,
            index: RwLock::new(index)
        }
    }

    pub fn len(&self) -> usize {
        let i = self.index.read().unwrap();

        let values = ffi::index_binary_flat_extract_values(&(*i));
        values.len()/D
    }

    pub fn get_all(&self) -> Vec<IndexBinaryEntry<D>> {
        self.get_batch(0, std::usize::MAX)
    }

    pub fn get_batch(&self, offset: usize, len: usize) -> Vec<IndexBinaryEntry<D>> {
        let mut rval: Vec<IndexBinaryEntry<D>> = Vec::new();
        let i = self.index.read().unwrap();
        let values = ffi::index_binary_flat_extract_values(&(*i));

        let offset = std::cmp::min(offset, values.len()/D);
        let end = std::cmp::min(offset+len, values.len()/D);

        for x in offset..end {
            let mut hash: [u8; D] = [0; D];
            unsafe {
                let src_ptr = &(*values.get_unchecked(x*D)) as *const u8;
                let dst_ptr = &mut (hash[0]) as *mut u8;
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, D);
            }
            
            rval.push(IndexBinaryEntry {
                hash: hash,
                label: x
            });
        }
        rval
    }

    pub fn add(&mut self, data: [u8; D]) {
        unsafe {
            // TODO: This can panic if we fail in the write at some point. Crash more gracefully?
            self.index.write().unwrap().as_mut().map(|x| x.add(1 as i64, &data as *const u8));
        }
    }

    pub fn add_all(&mut self, data: &Vec<[u8; D]>) {
        if data.len() == 0 {
            return;
        }
        
        unsafe {
            // TODO: This can panic if we fail in the write at some point. Crash more gracefully?
            self.index.write().unwrap().as_mut().map(|x| x.add(data.len() as i64, &data[0] as *const u8));
        }
    }

    pub fn add_all_raw(&mut self, data: &Vec<u8>) {
        if data.len() == 0 {
            return;
        }
        
        unsafe {
            // TODO: This can panic if we fail in the write at some point. Crash more gracefully?
            self.index.write().unwrap().as_mut().map(|x| x.add((data.len()/D).try_into().unwrap(), &data[0] as *const u8));
        }
    }

    pub fn search(&self, queries: &Vec<[u8; D]>, k: usize) -> IndexBinarySearchResult {
        let mut result= IndexBinarySearchResult::new(k, queries.len());
        let mut distances: Vec<i32> = vec![-1; k*queries.len()];
        let mut indexes: Vec<i64> = vec![-1; k*queries.len()];

        unsafe {
            self.index.read().unwrap().search(queries.len() as i64, queries.as_ptr() as *const u8,
            k as i64, &mut distances[0] as *mut i32, &mut indexes[0] as *mut i64);
        }

        for r in 0..queries.len() {
            for rk in 0..k {
                result.queries[r][rk] = IndexBinarySearchQueryResult {
                    distance: distances[k*r+rk],
                    index: indexes[k*r+rk] as usize
                }
            }
        }

        return result;
    }

    pub fn range_search(&self, queries: &Vec<[u8; D]>, k: usize, radius: usize) -> IndexBinarySearchResult {
        let mut result= IndexBinarySearchResult::new(k, queries.len());
        let mut distances: Vec<i32> = vec![-1; k*queries.len()];
        let mut indexes: Vec<i64> = vec![-1; k*queries.len()];
        let mut sizes: Vec<i64> = vec![-1; queries.len()];

        unsafe {
            ffi::index_binary_flat_range_search(&self.index.read().unwrap(), 
                queries.len() as i64, queries.as_ptr() as *const u8, radius as i32, k as i64, 
                &mut distances[0] as *mut i32, &mut indexes[0] as *mut i64, &mut sizes[0] as *mut i64);
        }

        for r in 0..queries.len() {
            for rk in 0..std::cmp::min(sizes[r], k as i64) {
                let ix = indexes[k*r+(rk as usize)];
                if ix >= 0 {
                    result.queries[r].push(IndexBinarySearchQueryResult {
                        distance: distances[k*r+(rk as usize)],
                        index: indexes[k*r+(rk as usize)] as usize
                    });
                }
            }
        }
        result
    }

    pub fn display(&self) {
        self.index.read().unwrap().display();
    }
}

impl<const D: usize> serde::Serialize for IndexBinaryMultiHash<D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        let mut state = serializer.serialize_struct("snapshot", 2)?;
        state.serialize_field("dims", &self.dims)?;
        state.serialize_field("ids", &(*self.ids.read().unwrap()))?;

        let i = self.index.read().unwrap();
        let values = ffi::index_binary_multi_hash_extract_values(&(*i));
        unsafe {
            let addr = values.get_unchecked(0) as *const u8;
            // TODO: This makes a copy, do not like
            let slice = std::slice::from_raw_parts(addr, values.len()).to_vec();
            state.serialize_field("data", &slice)?;
        }
        state.end()
    }
}

impl<'de, const N: usize> serde::Deserialize<'de> for IndexBinaryMultiHash<N> {
    fn deserialize<D>(deserializer: D) -> Result<IndexBinaryMultiHash<N>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct IndexBinaryMultiHashVisitor<const N: usize>;

        impl<'de, const N: usize> Visitor<'de> for IndexBinaryMultiHashVisitor<N> {
            type Value = IndexBinaryMultiHash<N>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a binary file with index data")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                    A: serde::de::SeqAccess<'de>, {
                let mut ids = seq.next_element::<Vec<String>>()?.unwrap();
                // TODO: This loads too much data, do not like
                let data = seq.next_element::<Vec<u8>>()?.unwrap();
                let mut index: IndexBinaryMultiHash<N> = IndexBinaryMultiHash::new(32, 8, 2);
                index.add_all_raw(&data, &mut ids);
                Ok(index)
            }
        }

        const FIELDS: &'static [&'static str] = &["dims", "ids", "data"];
        deserializer.deserialize_struct("snapshot", FIELDS, IndexBinaryMultiHashVisitor::<N>)
    }
}

pub struct IndexBinaryMultiHash<const D: usize> 
{
    dims: i64,
    index: RwLock<cxx::UniquePtr<ffi::IndexBinaryMultiHash>>,
    ids: RwLock<Vec<String>>
}
impl<const D: usize> IndexBinaryMultiHash<D> 
// where Assert<{D % 8 == 0}>: IsTrue // TODO: Feature available in nightly
{
    pub fn new(nhash: i64, b: i64, nflip: i64) -> Self {
        let index = ffi::new_index_binary_multi_hash((D*8) as i64, nhash, b, nflip);
        Self { 
            dims: D as i64,
            index: RwLock::new(index),
            ids: RwLock::new(Vec::new())
        }
    }

    pub fn len(&self) -> usize {
        let i = self.index.read().unwrap();

        let values = ffi::index_binary_multi_hash_extract_values(&(*i));
        values.len()/D
    }

    pub fn get_all(&self) -> Vec<IndexBinaryEntry<D>> {
        self.get_batch(0, std::usize::MAX)
    }

    pub fn get_batch(&self, offset: usize, len: usize) -> Vec<IndexBinaryEntry<D>> {
        let mut rval: Vec<IndexBinaryEntry<D>> = Vec::new();
        let i = self.index.read().unwrap();
        let values = ffi::index_binary_multi_hash_extract_values(&(*i));

        let offset = std::cmp::min(offset, values.len()/D);
        let end = std::cmp::min(offset+len, values.len()/D);

        for x in offset..end {
            let mut hash: [u8; D] = [0; D];
            unsafe {
                let src_ptr = &(*values.get_unchecked(x*D)) as *const u8;
                let dst_ptr = &mut (hash[0]) as *mut u8;
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, D);
            }
            
            rval.push(IndexBinaryEntry {
                hash: hash,
                label: x
            });
        }
        rval
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

    pub fn add_all_raw(&mut self, data: &Vec<u8>, ids: &mut Vec<String>) {
        if data.len() == 0 {
            return;
        }
        
        unsafe {
            // TODO: This can panic if we fail in the write at some point. Crash more gracefully?
            self.index.write().unwrap().as_mut().map(|x| x.add(ids.len().try_into().unwrap(), &data[0] as *const u8));
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

        for r in 0..queries.len() {
            for rk in 0..k {
                result.queries[r][rk] = IndexBinarySearchQueryResult {
                    distance: distances[k*r+rk],
                    index: indexes[k*r+rk] as usize
                }
            }
        }

        return result;
    }

    pub fn range_search(&self, queries: &Vec<[u8; D]>, k: usize, radius: usize) -> IndexBinarySearchResult {
        let mut result= IndexBinarySearchResult::new(k, queries.len());
        let mut distances: Vec<i32> = vec![-1; k*queries.len()];
        let mut indexes: Vec<i64> = vec![-1; k*queries.len()];
        let mut sizes: Vec<i64> = vec![-1; queries.len()];

        unsafe {
            ffi::index_binary_multi_hash_range_search(&self.index.read().unwrap(), 
                queries.len() as i64, queries.as_ptr() as *const u8, radius as i32, k as i64, 
                &mut distances[0] as *mut i32, &mut indexes[0] as *mut i64, &mut sizes[0] as *mut i64);
        }

        for r in 0..queries.len() {
            for rk in 0..std::cmp::min(sizes[r], k as i64) {
                let ix = indexes[k*r+(rk as usize)];
                if ix >= 0 {
                    result.queries[r].push(IndexBinarySearchQueryResult {
                        distance: distances[k*r+(rk as usize)],
                        index: indexes[k*r+(rk as usize)] as usize
                    });
                }
            }
        }
        result
    }

    pub fn display(&self) {
        self.index.read().unwrap().display();
    }
}