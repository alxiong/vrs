//! LightLigero placeholder for benchmark estimation only
//!
//! LightLigero can be seen as a new IOPP.
//! Due to the modular design, there's lots of wasteful re-compute that needs to be deducted during VRS benchmark
//! 1. similar to FRIDA, CONDA's interleaved encoding (and MT col-commit) should be shared with VRS's own encoding (as they are the same).
//! 2. CONDA's encode (and MT commit) should be further shared with LightLigero's encode during commit,
//! because RS on reshaped matrix can be derived from (with only cheap field ops)
//! think of a larger degree polynomial chunked into smaller polynomials, see unit test to demonstrate this idea `reshape_interleaved_rs()`, at least 90% of time is deductable
//! 3. PST batch open could further save us time, open time would be similar to commit time.
//!
//! Roughly the effect is similar to FRIDA + (1 PST commit in CONDA) * 1.3
//!
//! ```
//! Conda+lightligero
//! +--------+------+------+-----+------+----------+-------------+-------------------------+---------------+---------------+
//! | Scheme | N    | l    | k   | n    | |M| (MB) | prover (ms) | per-node overhead. (KB) | per-node (KB) | verifier (ms) |
//! +--------+------+------+-----+------+----------+-------------+-------------------------+---------------+---------------+
//! | Conda  | 1024 | 2048 | 256 | 1024 | 15       | 2629        | 86.5078125              | 150.5078125   | 9             |
//! +--------+------+------+-----+------+----------+-------------+-------------------------+---------------+---------------+
//!
//! Conda::computed TwinMLE: 37 ms
//! Conda::MLE commit: 786 ms, total: 824 ms
//! Conda::encode: 322 ms, total: 1146 ms (REMOVABLE 1)
//! Conda::MT commit: 263 ms, total: 1410 ms (REMOVABLE 1)
//! Conda::partial evals done, total: 1496 ms
//! Conda::Consolidate: 3 ms, total: 1500 ms
//!        height: 32, width: 16384, m: 65536
//!        LightLigero::PST commit: 116 ms, total: 136 ms
//!        LightLigero::encode: 532 ms, total: 669 ms (REMOVABLE 2)
//!        LightLigero::MT commit: 293 ms, total: 963 ms (REMOVABLE 2)
//!        mt: leaves:65536, height: 32
//!        LightLigero::MT query: 0 ms, total: 963 ms
//!        LightLigero::PST Open: 142 ms, total: 1106 ms (REDUCIBLE 3)
//!        total proof size: 85848
//!Conda::MLPC eval: 1126 ms, total: 2627 ms
//!Conda::prepare shares done, total: 2628 ms
//! ```

#![allow(unused_variables)]

use std::marker::PhantomData;
use std::time::Instant;

use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{
    DenseUVPolynomial, EvaluationDomain, MultilinearExtension, Polynomial, Radix2EvaluationDomain,
};
use ark_serialize::*;
use ark_std::borrow::Borrow;
use ark_std::rand::rngs::StdRng;
use ark_std::rand::{CryptoRng, Rng, RngCore, SeedableRng};
use derivative::Derivative;
use itertools::izip;
use jf_pcs::prelude::{Commitment, MultilinearKzgPCS, MultilinearKzgProof};
use jf_pcs::{prelude::MLE, PCSError, PolynomialCommitmentScheme, StructuredReferenceString};
use p3_maybe_rayon::prelude::*;
use std::sync::Arc;

use crate::gxz::niec::{self, ConsolidationConfig, ConsolidationProof};
use crate::matrix::Matrix;
use crate::merkle_tree::{Path, SymbolMerkleTree};

/// these are type alias for PST
type PstSrs<E> = <MultilinearKzgPCS<E> as PolynomialCommitmentScheme>::SRS;
type PstProverParam<E> = <PstSrs<E> as StructuredReferenceString>::ProverParam;
type PstVerifierParam<E> = <PstSrs<E> as StructuredReferenceString>::VerifierParam;
type Pst<E> = MultilinearKzgPCS<E>;
type PstCommitment<E> = Commitment<E>;
type PstProof<E> = MultilinearKzgProof<E>;

pub struct LightLigeroPCS<E: Pairing>(PhantomData<E>);

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug, PartialEq, Eq, Derivative)]
#[derivative(Hash(bound = "E: Pairing"))]
pub struct LightLigeroProof<E: Pairing> {
    pub pst_cm: PstCommitment<E>,
    pub query_answers: Vec<Vec<E::ScalarField>>,
    pub query_proofs: Vec<Path<E::ScalarField>>,
    pub rand_point: Vec<E::ScalarField>,
    pub sumcheck_proof: Vec<E::ScalarField>,
    pub pst_proof: PstProof<E>,
}

#[derive(Clone, Debug)]
pub struct LightLigeroSRS<E: Pairing> {
    pub pst_srs: PstSrs<E>,
}

impl<E: Pairing> PolynomialCommitmentScheme for LightLigeroPCS<E> {
    type SRS = LightLigeroSRS<E>;
    type Polynomial = MLE<E::ScalarField>;
    type Point = Vec<E::ScalarField>;
    type Evaluation = E::ScalarField;
    // just a merkle root
    type Commitment = Vec<u8>;
    // (pst_cm, query answer, merkle proofs, rand_point, consolidation proof, pst_proof)
    type Proof = LightLigeroProof<E>;
    type BatchCommitment = ();
    type BatchProof = ();

    fn commit(
        prover_param: impl Borrow<<Self::SRS as StructuredReferenceString>::ProverParam>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let pk = prover_param.borrow();
        let nv_x = pk.1;
        let nv_y = pk.0 .0.num_vars - nv_x;
        let height = 1 << nv_x;
        let m = (1 << nv_y) * 4; // rate is 1/4
        let domain = Radix2EvaluationDomain::<E::ScalarField>::new(m).unwrap();

        // assume passing in coeffs, instead of eval-form
        let coeffs = Matrix::new(poly.evaluations.clone(), m, height).unwrap();
        // row-wise FFT
        let encoded: Vec<E::ScalarField> =
            coeffs.par_row().flat_map(|row| domain.fft(row)).collect();
        let encoded_matrix = Matrix::new(encoded, m, height).unwrap();

        let mt = SymbolMerkleTree::new(encoded_matrix.col_iter());
        Ok(mt.root())
    }

    fn open(
        prover_param: impl Borrow<<Self::SRS as StructuredReferenceString>::ProverParam>,
        poly: &Self::Polynomial,
        point: &Self::Point,
    ) -> Result<(Self::Proof, Self::Evaluation), PCSError> {
        let pk = prover_param.borrow();
        let nv_x = pk.1;
        let nv_y = pk.0 .0.num_vars - nv_x;
        let height = 1 << nv_x;
        let m = (1 << nv_y) * 4; // rate is 1/4
        let rng = &mut StdRng::from_seed([42; 32]);
        let domain = Radix2EvaluationDomain::<E::ScalarField>::new(m).unwrap();
        #[cfg(feature = "print-conda")]
        println!("\t height: {height}, width: {}, m: {}", 1 << nv_y, m);

        let total_start = Instant::now();
        // see module doc
        let mut lightligero_deductible = 0;

        // evaluate the first nv_x variable to derive f'
        let f = Arc::new(poly.fix_variables(&point[..nv_x]));
        let start = Instant::now();
        let pst_cm = Pst::commit(&pk.0, &f).unwrap();
        #[cfg(feature = "print-conda")]
        println!(
            "\t LightLigero::PST commit: {} ms, total: {} ms",
            start.elapsed().as_millis(),
            total_start.elapsed().as_millis()
        );
        let commit_time = start.elapsed().as_millis();

        // TODO: change back to 50
        let s = 50;
        let indices = (0..s).map(|_| rng.gen_range(0..m)).collect::<Vec<_>>();
        // TODO: can only re-commit to get the merkle tree again
        // assume passing in coeffs, instead of eval-form
        let coeffs = Matrix::new(poly.evaluations.clone(), 1 << nv_y, height).unwrap();
        assert_eq!(poly.evaluations.len(), (1 << nv_y) * height);

        let start = Instant::now();
        // row-wise FFT
        let encoded: Vec<E::ScalarField> =
            coeffs.par_row().flat_map(|row| domain.fft(row)).collect();
        let encoded_matrix = Matrix::new(encoded.clone(), m, height).unwrap();
        #[cfg(feature = "print-conda")]
        println!(
            "\t LightLigero::encode: {} ms, total: {} ms",
            start.elapsed().as_millis(),
            total_start.elapsed().as_millis()
        );
        lightligero_deductible += start.elapsed().as_millis();

        let start = Instant::now();
        let mt = SymbolMerkleTree::new(encoded_matrix.col_iter());
        #[cfg(feature = "print-conda")]
        println!(
            "\t LightLigero::MT commit: {} ms, total: {} ms",
            start.elapsed().as_millis(),
            total_start.elapsed().as_millis()
        );
        lightligero_deductible += start.elapsed().as_millis();

        // prepare query proofs
        let start = Instant::now();
        let query_answers: Vec<Vec<E::ScalarField>> = indices
            .par_iter()
            .map(|idx| encoded_matrix.col(*idx).unwrap())
            .collect();
        let query_proofs: Vec<Path<E::ScalarField>> = indices
            .par_iter()
            .map(|idx| mt.generate_proof(*idx))
            .collect();
        #[cfg(feature = "print-conda")]
        println!(
            "\t LightLigero::MT query: {} ms, total: {} ms",
            start.elapsed().as_millis(),
            total_start.elapsed().as_millis()
        );

        // sumcheck is gonna be so fast, that we skip the prover time
        let rand_point = vec![E::ScalarField::from(42); nv_y];
        // simulate sumcheck proof size, 2 fields each round, nv_y rounds in total
        let sumcheck_proof = vec![E::ScalarField::from(42); nv_y * 2];

        // PST open
        let start = Instant::now();
        let (pst_proof, pst_eval) = Pst::open(&pk.0, &f, &rand_point).unwrap();
        #[cfg(feature = "print-conda")]
        println!(
            "\t LightLigero::PST Open: {} ms, total: {} ms",
            start.elapsed().as_millis(),
            total_start.elapsed().as_millis()
        );
        lightligero_deductible += start.elapsed().as_millis() - commit_time;

        // println!(
        //     "\t pst_cm size: {}, query_answer size: {}, query_proofs size: {}, pst_proof size: {}",
        //     pst_cm.serialized_size(Compress::No),
        //     query_answers.serialized_size(Compress::No),
        //     query_proofs.serialized_size(Compress::No),
        //     pst_proof.serialized_size(Compress::No),
        // );

        let proof = LightLigeroProof {
            pst_cm,
            query_answers,
            query_proofs,
            rand_point,
            sumcheck_proof,
            pst_proof,
        };
        #[cfg(feature = "print-conda")]
        println!(
            "\t total proof size: {}",
            proof.serialized_size(Compress::No)
        );
        println!(
            "⚠️ ⚠️ LightLigero:: deductiable: {} ms",
            lightligero_deductible
        );

        let eval = poly.evaluate(point);

        Ok((proof, eval))
    }

    fn verify(
        vk: &<Self::SRS as StructuredReferenceString>::VerifierParam,
        commitment: &Self::Commitment,
        point: &Self::Point,
        value: &Self::Evaluation,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        let mut verified = true;
        let nv_x = vk.1;
        let nv_y = vk.0 .0.num_vars - nv_x;
        let height = 1 << nv_x;
        let m = (1 << nv_y) * 4; // rate is: 1/4
        let rng = &mut StdRng::from_seed([42; 32]);

        let s = 50;
        let indices = (0..m).map(|_| rng.gen_range(0..m)).collect::<Vec<_>>();

        // verify all merkle proofs
        for (idx, val, proof) in izip!(
            indices.iter(),
            proof.query_answers.iter(),
            proof.query_proofs.iter()
        ) {
            // verified &= proof.verify(&commitment, *idx, val.clone());
            proof.verify(&commitment, *idx, val.clone());
        }

        // skip checking consolidation
        // just do PST verify
        Pst::verify(
            &vk.0,
            &proof.pst_cm,
            &proof.rand_point,
            &value,
            &proof.pst_proof,
        )
        .unwrap();
        Ok(verified)
    }

    // We don't need these, thus left unimplemented
    fn trim(
        srs: impl Borrow<Self::SRS>,
        supported_degree: usize,
        supported_num_vars: Option<usize>,
    ) -> Result<
        (
            <Self::SRS as StructuredReferenceString>::ProverParam,
            <Self::SRS as StructuredReferenceString>::VerifierParam,
        ),
        PCSError,
    > {
        unimplemented!()
    }
    fn batch_commit(
        prover_param: impl Borrow<<Self::SRS as StructuredReferenceString>::ProverParam>,
        polys: &[Self::Polynomial],
    ) -> Result<Self::BatchCommitment, PCSError> {
        unimplemented!()
    }
    fn batch_open(
        prover_param: impl Borrow<<Self::SRS as StructuredReferenceString>::ProverParam>,
        batch_commitment: &Self::BatchCommitment,
        polynomials: &[Self::Polynomial],
        points: &[Self::Point],
    ) -> Result<(Self::BatchProof, Vec<Self::Evaluation>), PCSError> {
        unimplemented!()
    }
    fn batch_verify<R: RngCore + CryptoRng>(
        verifier_param: &<Self::SRS as StructuredReferenceString>::VerifierParam,
        multi_commitment: &Self::BatchCommitment,
        points: &[Self::Point],
        values: &[Self::Evaluation],
        batch_proof: &Self::BatchProof,
        rng: &mut R,
    ) -> Result<bool, PCSError> {
        unimplemented!()
    }
}

impl<E: Pairing> StructuredReferenceString for LightLigeroSRS<E> {
    /// pst_pk, \mu' (nv_x)
    type ProverParam = (PstProverParam<E>, usize);
    /// pst_vk, \mu' (nv_x)
    type VerifierParam = (PstVerifierParam<E>, usize);

    fn gen_srs_for_testing<R: RngCore + CryptoRng>(
        rng: &mut R,
        supported_degree: usize,
    ) -> Result<Self, PCSError> {
        let pst_srs = PstSrs::gen_srs_for_testing(rng, supported_degree)?;
        Ok(Self { pst_srs })
    }

    fn trim(
        &self,
        supported_degree: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let (pst_pk, pst_vk) = PstSrs::trim(&self.pst_srs, supported_degree)?;
        // height is log(nv)
        let (nv_x, nv_y) = shape_heuristic(pst_pk.0.num_vars);
        let pk = (pst_pk, nv_x);
        let vk = (pst_vk, nv_x);

        Ok((pk, vk))
    }

    fn supported_degree(&self) -> usize {
        self.pst_srs.supported_degree()
    }
    fn extract_prover_param(&self, supported_degree: usize) -> Self::ProverParam {
        unimplemented!()
    }
    fn extract_verifier_param(&self, supported_degree: usize) -> Self::VerifierParam {
        unimplemented!()
    }
    fn trim_with_verifier_degree(
        &self,
        prover_supported_degree: usize,
        verifier_supported_degree: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        unimplemented!()
    }
    fn gen_srs_for_testing_with_verifier_degree<R: RngCore + CryptoRng>(
        rng: &mut R,
        prover_supported_degree: usize,
        verifier_supported_degree: usize,
    ) -> Result<Self, PCSError> {
        unimplemented!()
    }
}

/// given total nv, returns nv_x (height), nv_y
fn shape_heuristic(nv: usize) -> (usize, usize) {
    // let mut nv_x = nv.ilog2() as usize;
    // let mut nv_y = nv - nv_x;
    // if nv_y % 2 != 0 {
    //     nv_y -= 1;
    //     nv_x += 1;
    // }

    let mut nv_x = 4; // log(16), fixed height for now
    let mut nv_y = nv - nv_x;
    (nv_x, nv_y)
}

#[test]
fn test_lightligero_test() {
    use ark_bn254::{Bn254, Fr};
    use ark_poly::evaluations::multivariate::DenseMultilinearExtension;
    use ark_std::UniformRand;

    let rng = &mut StdRng::from_seed([42; 32]);
    let nv = 11;

    let pp = LightLigeroSRS::<Bn254>::gen_srs_for_testing(rng, nv).unwrap();
    let (pk, vk) = pp.trim(nv).unwrap();

    let poly = MLE::from(DenseMultilinearExtension::rand(nv, rng));
    let cm = LightLigeroPCS::<Bn254>::commit(&pk, &poly).unwrap();

    let point: Vec<_> = (0..nv).map(|_| Fr::rand(rng)).collect();
    let (proof, eval) = LightLigeroPCS::<Bn254>::open(&pk, &poly, &point).unwrap();

    assert!(LightLigeroPCS::<Bn254>::verify(&vk, &cm, &point, &eval, &proof).unwrap());
}

#[test]
fn reshape_interleaved_rs() {
    use ark_bn254::Fr;
    use ark_poly::{univariate::DensePolynomial, Radix2EvaluationDomain};
    let rng = &mut StdRng::from_seed([42; 32]);

    // we demonstrate that 1x16 RS encode is the same as 2x8 interleaved encode values
    let degree = 3;
    let width = degree + 1;
    let domain = Radix2EvaluationDomain::<Fr>::new(width * 2).unwrap();
    assert_eq!(domain.size as usize, width * 2);
    let poly = DensePolynomial::<Fr>::rand(degree, rng);

    let encoded = domain.fft(&poly.coeffs);

    let coeffs = Matrix::new(poly.coeffs.clone(), width / 2, 2).unwrap();
    let half_domain = Radix2EvaluationDomain::<Fr>::new(degree).unwrap();
    // row-wise FFT
    let interleaved_encoded: Vec<Fr> = coeffs
        .row_iter()
        .flat_map(|row| half_domain.fft(row))
        .collect();
    // TODO: directly derive values in interleaved encoded vector by taking a pair from encoded
    // and shifted by some \omega^i.
}
