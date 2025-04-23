//! LightLigero placeholder for benchmark estimation only

use std::marker::PhantomData;

use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::{EvaluationDomain, MultilinearExtension, Polynomial};
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
    pub consolidation_proof: ConsolidationProof<E::ScalarField>,
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
        let height = 2u32.pow(nv_x as u32) as usize;
        let m = pk.2.domain.size as usize;

        // assume passing in coeffs, instead of eval-form
        let coeffs = Matrix::new(poly.evaluations.clone(), m, height).unwrap();
        // row-wise FFT
        let encoded: Vec<E::ScalarField> = coeffs
            .par_row()
            .flat_map(|row| pk.2.domain.fft(row))
            .collect();
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
        let height = 2u32.pow(nv_x as u32) as usize;
        let m = pk.2.domain.size as usize;
        let rng = &mut StdRng::from_seed([42; 32]);

        // evaluate the first nv_x variable to derive f'
        let f = Arc::new(poly.fix_variables(&point[..nv_x]));
        let pst_cm = Pst::commit(&pk.0, &f).unwrap();

        let s = 50;
        let indices = (0..m).map(|_| rng.gen_range(0..m)).collect::<Vec<_>>();
        // TODO: can only re-commit to get the merkle tree again
        // assume passing in coeffs, instead of eval-form
        let coeffs = Matrix::new(poly.evaluations.clone(), m, height).unwrap();
        // row-wise FFT
        let encoded: Vec<E::ScalarField> = coeffs
            .par_row()
            .flat_map(|row| pk.2.domain.fft(row))
            .collect();
        let encoded_matrix = Matrix::new(encoded.clone(), m, height).unwrap();
        let mt = SymbolMerkleTree::new(encoded_matrix.col_iter());

        // prepare query proofs
        let query_answers: Vec<Vec<E::ScalarField>> = indices
            .par_iter()
            .map(|idx| encoded_matrix.col(*idx).unwrap())
            .collect();
        let query_proofs: Vec<Path<E::ScalarField>> = indices
            .par_iter()
            .map(|idx| mt.generate_proof(*idx))
            .collect();

        // consolidate all the multile opening, using consolidation to simulate sumcheck
        // skip the random linear combine steps, just take the first row codeword
        let (rand_point, consolidation_proofs) = niec::consolidate(&pk.2, &encoded[..m]);

        // PST open
        let (pst_proof, pst_eval) = Pst::open(&pk.0, &f, &rand_point).unwrap();

        let proof = LightLigeroProof {
            pst_cm,
            query_answers,
            query_proofs,
            rand_point,
            consolidation_proof: consolidation_proofs[0].clone(),
            pst_proof,
        };
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
        let height = 2u32.pow(nv_x as u32) as usize;
        let m = vk.2.domain.size as usize;
        let rng = &mut StdRng::from_seed([42; 32]);

        let s = 50;
        let indices = (0..m).map(|_| rng.gen_range(0..m)).collect::<Vec<_>>();

        // verify all merkle proofs
        for (idx, val, proof) in izip!(
            indices.iter(),
            proof.query_answers.iter(),
            proof.query_proofs.iter()
        ) {
            verified &= proof.verify(&commitment, *idx, val.clone());
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
    /// pst_pk, \mu' (nv_x), niec
    type ProverParam = (
        PstProverParam<E>,
        usize,
        ConsolidationConfig<E::ScalarField>,
    );
    /// pst_vk, \mu' (nv_x), niec
    type VerifierParam = (
        PstVerifierParam<E>,
        usize,
        ConsolidationConfig<E::ScalarField>,
    );

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
        let m = 2u32.pow(nv_y as u32) as usize * 4; // rate is 1/4

        let config = ConsolidationConfig::new(nv_y, m, 2);
        let pk = (pst_pk, nv_x, config.clone());
        let vk = (pst_vk, nv_x, config);

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
    let mut nv_x = nv.ilog2() as usize;
    let mut nv_y = nv - nv_x;
    if nv_y % 2 != 0 {
        nv_y -= 1;
        nv_x += 1;
    }
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
