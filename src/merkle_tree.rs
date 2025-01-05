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

    /// Convenience method to accept a slice of slices directly.
    pub fn from_slice<L: Borrow<[F]>>(leaves: &[L]) -> Self {
        Self::new(leaves.iter().map(|x| x.borrow()))
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

    /// Returns the byte size of the root value
    pub const fn root_byte_size() -> usize {
        32
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

    /// Returns the capacity of the tree
    pub fn capacity(&self) -> usize {
        2usize.pow(self.height() as u32 - 1)
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
    /// Verify the merkle path
    pub fn verify<L: Borrow<[F]>>(&self, root: &Vec<u8>, pos: usize, leaf: L) -> bool {
        self.inner.verify(&(), &(), root, leaf).unwrap() && pos == self.index()
    }
    /// returns the leaf index of this merkle proof is proving
    pub fn index(&self) -> usize {
        self.inner.leaf_index
    }
    /// return original tree height
    pub fn height(&self) -> usize {
        // auth_path didn't include the root nor the leaf level
        self.inner.auth_path.len() + 2
    }
    /// returns the capacity of the original merkle tree
    pub fn capacity(&self) -> usize {
        2usize.pow(self.height() as u32 - 1)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::test_rng;

    use super::*;
    use ark_bn254::Fr;
    use ark_std::UniformRand;

    // check our wrapper code didn't violate merkle tree correctness
    #[test]
    fn test_mt_wrapper_sanity() {
        let rng = &mut test_rng();

        for log_size in 1..10usize {
            let size = 2usize.pow(log_size as u32);
            let leaves: Vec<_> = (0..size).map(|_| [Fr::rand(rng)]).collect();
            let mt = SymbolMerkleTree::new(leaves.clone());
            let root = mt.root();

            assert_eq!(mt.height(), log_size + 1);
            assert_eq!(mt.capacity(), size);
            for _ in 0..10 {
                let idx = rng.gen_range(0..size) as usize;
                let proof = mt.generate_proof(idx);
                assert!(proof.verify(&root, idx, leaves[idx].as_slice()));
                assert_eq!(mt.height(), proof.height());
                assert_eq!(proof.capacity(), size);
            }
        }
    }
}
