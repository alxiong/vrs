use std::marker::PhantomData;

use ark_bn254::Fr;
use ark_serialize::CanonicalSerialize;
use rs_merkle::{Hasher, MerkleProof, MerkleTree};

#[derive(Debug, Clone)]
pub struct Blake16 {}

impl Hasher for Blake16 {
    type Hash = [u8; 16];

    fn hash(data: &[u8]) -> [u8; 16] {
        blake3::hash(data).as_bytes()[..16].try_into().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct Blake32 {}

impl Hasher for Blake32 {
    type Hash = [u8; 32];

    fn hash(data: &[u8]) -> [u8; 32] {
        blake3::hash(data).as_bytes().clone()
    }
}

#[derive(Clone)]
pub struct MerkleTreeProver<H: Hasher> {
    pub merkle_tree: MerkleTree<H>,
    leave_num: usize,
}

pub struct Serialize;

impl Serialize {
    pub fn serialize_fields(v: &[Fr]) -> Vec<u8> {
        let mut bytes = vec![];
        v.iter().for_each(|x| {
            <Fr as CanonicalSerialize>::serialize_compressed(&x, &mut bytes).unwrap()
        });
        bytes
    }
}

#[derive(Debug, Clone)]
pub struct MerkleTreeVerifier<H: Hasher> {
    pub merkle_root: H::Hash,
    pub leave_number: usize,
}

impl<H: Hasher> MerkleTreeProver<H> {
    pub fn new(leaf_values: &Vec<Vec<u8>>) -> Self {
        let leaves = leaf_values.iter().map(|x| H::hash(x)).collect::<Vec<_>>();
        let merkle_tree = MerkleTree::<H>::from_leaves(&leaves);
        Self {
            merkle_tree,
            leave_num: leaf_values.len(),
        }
    }

    pub fn leave_num(&self) -> usize {
        self.leave_num
    }

    pub fn commit(&self) -> H::Hash {
        self.merkle_tree.root().unwrap()
    }

    pub fn open(&self, leaf_indices: &[usize]) -> Vec<u8> {
        self.merkle_tree.proof(leaf_indices).to_bytes()
    }
}

impl<H: Hasher> MerkleTreeVerifier<H> {
    pub fn new(leave_number: usize, merkle_root: &H::Hash) -> Self {
        Self {
            leave_number,
            merkle_root: merkle_root.clone(),
        }
    }

    pub fn verify(
        &self,
        proof_bytes: Vec<u8>,
        indices: &Vec<usize>,
        leaves: &Vec<Vec<u8>>,
    ) -> bool {
        let proof = MerkleProof::<H>::try_from(proof_bytes).unwrap();
        let leaves_to_prove: Vec<H::Hash> = leaves.iter().map(|x| H::hash(x)).collect();
        proof.verify(
            self.merkle_root,
            indices,
            &leaves_to_prove,
            self.leave_number,
        )
    }
}

pub struct MerkleRoot<H: Hasher>(PhantomData<H>);
impl<H: Hasher> MerkleRoot<H> {
    pub fn get_root(
        proof_bytes: Vec<u8>,
        index: usize,
        leaf: Vec<u8>,
        leave_number: usize,
    ) -> H::Hash {
        let proof = MerkleProof::<H>::try_from(proof_bytes).unwrap();
        let leaf_hashes = vec![H::hash(&leaf)];
        proof
            .root(&vec![index], &leaf_hashes, leave_number)
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::{Field, UniformRand};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use rand::thread_rng;

    use super::*;

    #[test]
    fn commit_and_open() {
        let leaf_values = (0..8)
            .map(|x| Serialize::serialize_fields(&[Fr::from(x * 2), Fr::from(x * 2 + 1)]))
            .collect::<Vec<_>>();
        let leave_number = leaf_values.len();
        let prover = MerkleTreeProver::<Blake16>::new(&leaf_values);
        let root = prover.commit();
        let verifier = MerkleTreeVerifier::<Blake16>::new(leave_number, &root);
        let leaf_indices = vec![2, 3];
        let proof_bytes = prover.open(&leaf_indices);
        let open_values = vec![
            Serialize::serialize_fields(&[Fr::from(2 * 2), Fr::from(2 * 2 + 1)]),
            Serialize::serialize_fields(&[Fr::from(3 * 2), Fr::from(3 * 2 + 1)]),
        ];
        assert!(verifier.verify(proof_bytes, &leaf_indices, &open_values));
    }

    #[test]
    fn serialize() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let x = <Fr as UniformRand>::rand(&mut rng);
            let mut bytes = vec![];
            <Fr as CanonicalSerialize>::serialize_compressed(&x, &mut bytes).unwrap();
            let y = <Fr as CanonicalDeserialize>::deserialize_compressed(bytes.as_slice()).unwrap();
            assert_eq!(x, y);
        }

        for _ in 0..10000 {
            let bytes = (0..16).map(|_| rand::random::<u8>()).collect::<Vec<_>>();
            let _ = <Fr as Field>::from_random_bytes(&bytes).unwrap();
        }
    }
}
