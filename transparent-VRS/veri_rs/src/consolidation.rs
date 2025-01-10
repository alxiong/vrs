use std::{iter, mem::size_of};

use ark_bn254::Fr;
use ark_ff::Field;
use util::{
    merkle_tree::{Blake16, MerkleRoot, MerkleTreeProver},
    mul_group::Radix2Group,
};

use crate::poly::{MultilinearPoly, UniPolyEvals, UniVarPoly};

pub struct Proofs {
    poly: MultilinearPoly,
    replicas: Vec<UniVarPoly>,
    first_tree: MerkleTreeProver<Blake16>,
    merkle_trees: Vec<MerkleTreeProver<Blake16>>,
    proofs: Vec<Vec<UniPolyEvals>>,
    final_poly: UniVarPoly,
}

impl Proofs {
    pub fn n_th_replica(&self, n: usize) -> Symbol {
        let proofs = self
            .proofs
            .iter()
            .map(|polies| {
                let len = polies.len();
                polies[n & (len - 1)].clone()
            })
            .collect::<Vec<_>>();
        let merkle_paths = self
            .merkle_trees
            .iter()
            .map(|x| {
                let len = x.leave_num();
                x.open(&[n & (len - 1)])
            })
            .collect();
        Symbol {
            poly: self.poly.clone(),
            first_paths: self.first_tree.open(&[n]),
            replica: self.replicas[n].clone(),
            merkle_paths,
            proofs,
            final_poly: self.final_poly.clone(),
        }
    }
}

pub struct VeriRsProver {
    fft_group: Radix2Group,
    evaluate_groups: Vec<Radix2Group>,
    root_inv: Fr,
    code_rate: usize,
    log_row_length: usize,
    log_symbol_number: usize,
    layer_number: usize,
    step: usize,
}

pub struct Symbol {
    poly: MultilinearPoly,
    replica: UniVarPoly,
    proofs: Vec<UniPolyEvals>,
    first_paths: Vec<u8>,
    merkle_paths: Vec<Vec<u8>>,
    final_poly: UniVarPoly,
}

impl Symbol {
    pub fn proof_size(&self) -> usize {
        size_of::<Fr>()
            * (self.proofs.iter().map(|x| x.len()).sum::<usize>() + self.final_poly.len())
            + self.first_paths.len()
            + self.merkle_paths.iter().map(|x| x.len()).sum::<usize>()
    }
}

impl VeriRsProver {
    pub fn setup(
        log_blob_size: usize,
        log_layer_num: usize,
        code_rate: usize,
        step: usize,
    ) -> Self {
        let fft_group = Radix2Group::new(log_blob_size - log_layer_num + code_rate);
        let log_row_length = log_blob_size - log_layer_num;
        let evaluate_groups = iter::successors(Some(fft_group.clone()), |x| Some(x.exp(1 << step)))
            .take(log_row_length / step)
            .collect::<Vec<_>>();
        let root_inv = Radix2Group::new(step).element_inv_at(1);
        VeriRsProver {
            fft_group,
            root_inv,
            code_rate,
            evaluate_groups,
            log_row_length,
            log_symbol_number: log_blob_size - log_layer_num + code_rate,
            layer_number: 1 << log_layer_num,
            step,
        }
    }

    pub fn encode(&self, data: Vec<Fr>) -> Vec<Vec<Fr>> {
        data.chunks(1 << self.log_row_length)
            .map(|x| self.fft_group.fft(x.to_vec()))
            .collect::<Vec<_>>()
    }

    pub fn prove(&self, data: Vec<Fr>, codewords: Vec<Vec<Fr>>) -> Proofs {
        let replicas = (0..(1 << self.log_symbol_number))
            .map(|i| UniVarPoly::new((0..self.layer_number).map(|j| codewords[j][i]).collect()))
            .collect::<Vec<_>>();
        let (first_tree, challenge) = {
            let tree = MerkleTreeProver::new(&replicas.iter().map(|x| x.serialize()).collect());
            let bytes: [u8; 16] = tree.commit();
            let challenge = <Fr as Field>::from_random_bytes(&bytes).unwrap();
            (tree, challenge)
        };
        let mut merkle_trees = vec![];

        let mut claims = replicas
            .iter()
            .map(|x| x.eval(&challenge))
            .collect::<Vec<_>>();
        let inv_2 = <Fr as Field>::inverse(&2.into()).unwrap();
        let mut proofs = vec![];
        for group in &self.evaluate_groups {
            let log_replica = claims.len().ilog2() as usize - self.step;
            let this_proof = (0..(claims.len() >> self.step))
                .map(|i| {
                    UniPolyEvals::new(
                        (0..(1 << self.step))
                            .map(|j| claims[i + (j << log_replica)])
                            .collect(),
                        group.element_inv_at(i),
                    )
                })
                .collect::<Vec<_>>();
            let challenge = {
                let tree =
                    MerkleTreeProver::new(&this_proof.iter().map(|x| x.serialize()).collect());
                let root: [u8; 16] = tree.commit();
                merkle_trees.push(tree);
                <Fr as Field>::from_random_bytes(&root).unwrap()
            };
            claims = this_proof
                .iter()
                .map(|x| x.clone().eval(challenge, self.root_inv, inv_2))
                .collect();
            proofs.push(this_proof);
        }

        Proofs {
            poly: MultilinearPoly::new(data),
            replicas,
            proofs,
            first_tree,
            merkle_trees,
            final_poly: UniVarPoly::new({
                let log_order = claims.len().ilog2() as usize;
                let mut coeff = Radix2Group::new(log_order).ifft(claims);
                coeff.truncate(1 << (log_order - self.code_rate));
                coeff
            }),
        }
    }
}

pub struct VeriRsVerifier {
    index: usize,
    symbol_number: usize,
    inner_location: Vec<usize>,
    outer_location: Vec<usize>,
    leaves_number: Vec<usize>,
    step: usize,
    inv_2: Fr,
    omega_inv: Fr,
    final_point: Fr,
    final_length: usize,
}

impl VeriRsVerifier {
    pub fn setup(
        index: usize,
        step: usize,
        mut log_symbol_number: usize,
        code_rate: usize,
    ) -> VeriRsVerifier {
        let sn = 1 << log_symbol_number;
        let nv = log_symbol_number - code_rate;
        let round = nv / step;
        let mut inner_location = vec![];
        let mut outer_location = vec![];
        let mut leaves_number = vec![];
        for _ in 0..round {
            log_symbol_number -= step;
            inner_location.push((index >> log_symbol_number) & ((1 << step) - 1));
            outer_location.push(index & ((1 << log_symbol_number) - 1));
            leaves_number.push(1 << log_symbol_number);
        }
        let inv_2 = <Fr as Field>::inverse(&2.into()).unwrap();
        let omega_inv = Radix2Group::new(step).element_inv_at(1);
        let final_point =
            Radix2Group::new(log_symbol_number).element_at(index & ((1 << log_symbol_number) - 1));
        VeriRsVerifier {
            index,
            symbol_number: sn,
            inner_location,
            outer_location,
            leaves_number,
            step,
            inv_2,
            omega_inv,
            final_point,
            final_length: log_symbol_number - code_rate,
        }
    }

    pub fn verify(&self, symbol: Symbol) -> bool {
        let Symbol {
            poly,
            replica,
            first_paths,
            proofs,
            merkle_paths,
            final_poly,
        } = symbol;
        let root: [u8; 16] = MerkleRoot::<Blake16>::get_root(
            first_paths,
            self.index,
            replica.serialize(),
            self.symbol_number,
        );
        let first_challenge = <Fr as Field>::from_random_bytes(&root).unwrap();
        let mut x = replica.eval(&first_challenge);
        let mut eval_point = vec![];

        for ((poly, paths), ((&inner, &outer), &leave_number)) in
            proofs.into_iter().zip(merkle_paths.into_iter()).zip(
                self.inner_location
                    .iter()
                    .zip(self.outer_location.iter())
                    .zip(self.leaves_number.iter()),
            )
        {
            assert_eq!(x, poly.n_th_eval(inner));
            let root: [u8; 16] =
                MerkleRoot::<Blake16>::get_root(paths, outer, poly.serialize(), leave_number);
            let challenge = <Fr as Field>::from_random_bytes(&root).unwrap();
            x = poly.eval(challenge, self.omega_inv, self.inv_2);
            eval_point.append(
                &mut iter::successors(Some(challenge), |&x| Some(x * x))
                    .take(self.step)
                    .collect(),
            );
        }

        assert_eq!(x, final_poly.eval(&self.final_point));
        let x = final_poly.eval(&19260817.into());
        eval_point.append(
            &mut iter::successors(Some(Fr::from(19260817)), |&x| Some(x * x))
                .take(self.final_length)
                .collect(),
        );
        eval_point.append(
            &mut iter::successors(Some(first_challenge), |&x| Some(x * x))
                .take(replica.len().ilog2() as usize)
                .collect::<Vec<_>>(),
        );
        assert_eq!(x, poly.eval(&eval_point));
        true
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use rand::thread_rng;

    use super::{VeriRsProver, VeriRsVerifier};

    #[test]
    fn consolidation_test() {
        let mut rng = thread_rng();
        let log_blob_size = 13;
        let step = 4;
        let log_layer_num = 3;
        let code_rate = 1;
        let prover = VeriRsProver::setup(log_blob_size, log_layer_num, code_rate, step);
        let data = (0..(1 << log_blob_size))
            .map(|_| <Fr as UniformRand>::rand(&mut rng))
            .collect::<Vec<_>>();
        let codewords = prover.encode(data.clone());
        let proofs = prover.prove(data, codewords);
        let symbol = proofs.n_th_replica(103);
        let verifier = VeriRsVerifier::setup(
            103,
            step,
            log_blob_size - log_layer_num + code_rate,
            code_rate,
        );
        assert!(verifier.verify(symbol));
    }
}
