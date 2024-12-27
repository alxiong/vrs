//! Merkle Tree implementation

use ark_crypto_primitives::{
    crh::{
        self,
        sha256::{digest::Digest, Sha256},
    },
    merkle_tree::{self as ark_mt, ByteDigestConverter},
};
use ark_ff::{Field, PrimeField};
pub use ark_mt::Config;
use ark_serialize::*;
use ark_std::{borrow::Borrow, iter::IntoIterator, marker::PhantomData, rand::Rng};

/// Parameter for a Merkle tree whose leaves are a symbol (arbitrary number of fields)
/// This param is for MT that uses byte-oriented hash/compression function
#[derive(Clone)]
pub struct SymbolMerkleTreeParams<F: Field>(PhantomData<F>);

impl<F: Field> ark_mt::Config for SymbolMerkleTreeParams<F> {
    type Leaf = [F];
    type LeafDigest = Vec<u8>; // always 32 bytes
    type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
    type InnerDigest = Vec<u8>; // always 32 bytes
    type LeafHash = Sha256OnFields<F>;
    type TwoToOneHash = crh::sha256::Sha256;
}

/// a CRH scheme that on inputs some field elements uses Sha256 to hash
pub struct Sha256OnFields<F>(PhantomData<F>);

impl<F: Field> crh::CRHScheme for Sha256OnFields<F> {
    type Input = [F];
    type Output = Vec<u8>; // always 32 bytes
    type Parameters = ();

    fn setup<R: Rng>(_r: &mut R) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }
    fn evaluate<T: Borrow<Self::Input>>(
        _pp: &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let fields = input.borrow();
        let mut bytes = vec![];
        fields
            .serialize_compressed(&mut bytes)
            .expect("fail to serialize");

        Ok(Sha256::digest(&bytes).to_vec())
    }
}
/// Merkle tree whose leaves are a symbol (consists of multiple field elements)
#[derive(Clone)]
pub struct SymbolMerkleTree<F: Field> {
    inner: ark_mt::MerkleTree<SymbolMerkleTreeParams<F>>,
}

impl<F: Field> SymbolMerkleTree<F> {
    /// Returns a new merkle tree. `leaves.len()` should be power of two.
    pub fn new<L: Borrow<[F]>>(leaves: impl IntoIterator<Item = L>) -> Self {
        let inner = ark_mt::MerkleTree::<SymbolMerkleTreeParams<F>>::new(&(), &(), leaves).unwrap();
        Self { inner }
    }

    /// Create an empty merkle tree such that all leaves are zero-filled.
    pub fn blank(height: usize) -> Self {
        let inner =
            ark_mt::MerkleTree::<SymbolMerkleTreeParams<F>>::blank(&(), &(), height).unwrap();
        Self { inner }
    }

    /// Returns the root of the Merkle tree.
    pub fn root(&self) -> Vec<u8> {
        self.inner.root()
    }

    /// Returns the root but as a field element
    pub fn root_as_field(&self) -> F {
        F::from_base_prime_field(<F::BasePrimeField as PrimeField>::from_le_bytes_mod_order(
            &self.inner.root(),
        ))
    }

    /// Returns the height of the Merkle tree.
    pub fn height(&self) -> usize {
        self.inner.height()
    }

    /// Returns the authentication path from leaf at `index` to root.
    pub fn generate_proof(&self, index: usize) -> Path<F> {
        Path {
            inner: self.inner.generate_proof(index).unwrap(),
        }
    }
}

/// A thin wrapper of Merkle Path/Proof
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct Path<F: Field> {
    inner: ark_mt::Path<SymbolMerkleTreeParams<F>>,
}

impl<F: Field> Path<F> {
    pub fn verify<L: Borrow<[F]>>(&self, root_hash: &Vec<u8>, leaf: L) -> bool {
        self.inner.verify(&(), &(), root_hash, leaf).unwrap()
    }
}
