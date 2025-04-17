//! Zero-overhead Data Availability (ZODA) [EMA24]
//!
//! NOTE: our conventional orientation is the transposed of that in the ZODA paper, see doc of [`ZodaConfig`] for details.
//!
//! # References
//! - https://eprint.iacr.org/2025/034
//! - parameters and bound: https://github.com/bcc-research/zoda-numerics

use crate::{
    matrix::Matrix,
    merkle_tree::{self, Path, SymbolMerkleTree, SymbolMerkleTreeParams},
    VerifiableReedSolomon, VrsError, VrsShare,
};
use ark_ff::{FftField, Field, PrimeField};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_serialize::CanonicalSerialize;
use ark_std::{
    collections::{BTreeSet, HashMap},
    end_timer,
    fmt::Debug,
    marker::PhantomData,
    rand::{CryptoRng, Rng, RngCore, SeedableRng},
    start_timer,
};
use itertools::{iproduct, izip};
use nimue::{
    plugins::ark::{FieldChallenges, FieldIOPattern},
    ByteWriter, DefaultHash, IOPattern,
};
use rand_chacha::ChaCha20Rng;

type MerkleRoot<F> = <SymbolMerkleTreeParams<F> as merkle_tree::Config>::InnerDigest;

/// A ZODA-based VRS
#[derive(Debug, Clone)]
pub struct ZodaVRS<F> {
    _field: PhantomData<F>,
}

/// Configuration/Parameters for ZODA scheme
///
/// # Notation
/// It's important to note that we use different notation than the ZODA paper to be consistent with the rest of the implementations in this repo.
/// Particularly, in ZODA, the nxn' data block B (\tilde{X} in paper) is 2D-tensor-coded to Z of size mxm',
/// by first "col-wise encode" into X (mxn'), think of n' as the interleave ratio, rho = n/m;
/// then B is col-wise scaled before "row-wise encode" into Y (nxm'), rho' = n'/m'.
///
/// Similarly, in ZODA paper, sampler samples |S| rows and |S'| cols of Z; vice versa in our notation.
///
/// By our convention (also that of Ligero), the block is first row-wise interleaved encoded (namely expanding horizontally) from Lxk to Lxn;
/// then col-wise from Lxn to mxn. Our row_blowup factor is n/k which is ZODA paper's rho'.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZodaConfig {
    /// field size, possibly extension field
    pub field_bit_size: u32,
    /// blowup factor when row-wise interleaved encoding into X
    pub row_blowup: usize,
    /// blowup factor when col-wise interleaved encoding into Y
    pub col_blowup: usize,
    /// soundness error for correct encoding verification (affect sampling size |S| and |S'|)
    pub log_soundness_err: u32,
    /// recovery error for having enough sampling nodes to fully recover the original B
    pub log_recovery_err: u32,
}

impl ZodaConfig {
    /// ref: https://github.com/bcc-research/zoda-numerics/blob/a3f166f4296150897ed20643d5afa066670c7b42/total-comm.jl#L60
    /// Returns (|S|, |S'|) to sample per node to get log_soundness_err
    /// |S|: number of cols of X
    /// |S'|: number of rows of Y
    #[inline]
    pub fn num_samples_per_node(&self) -> (usize, usize) {
        assert_eq!(
            self,
            &Self::default(),
            "only supports the default config for now"
        );

        let num_cols = (-(self.log_soundness_err as f64)
            / ((1.0 + (self.row_blowup as f64).recip()) / 2.0).log2())
        .ceil() as usize;
        let num_rows = num_cols;
        (num_cols, num_rows)
    }
    /// ref: https://github.com/bcc-research/zoda-numerics/blob/a3f166f4296150897ed20643d5afa066670c7b42/total-comm.jl#L3-L10
    /// `msg_len`: = k = L, since we only deals with square-shape data for now
    /// Returns the number of minimum nodes required to achieve log_recovery_err
    #[inline]
    pub fn num_nodes(&self, msg_len: usize) -> usize {
        assert_eq!(
            self,
            &Self::default(),
            "only supports the default config for now"
        );
        let codeword_len = msg_len * self.row_blowup;
        (self.log_recovery_err as usize + codeword_len)
            / (self.row_blowup.ilog2() as usize * self.num_samples_per_node().0)
    }
}

impl Default for ZodaConfig {
    /// ref: https://github.com/bcc-research/zoda-numerics/blob/main/total-comm.jl
    fn default() -> Self {
        Self {
            field_bit_size: ark_bn254::Fr::MODULUS_BIT_SIZE,
            row_blowup: 2,
            col_blowup: 2,
            log_soundness_err: 80,
            log_recovery_err: 40,
        }
    }
}

/// Proof of correct encoding for ZODA, only contains merkle proofs against three merkle roots
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, Default)]
pub struct ZodaProof<F: Field> {
    /// merkle proofs for sampled rows of X
    pub sampled_cols_proofs: Vec<Path<F>>,
    /// merkle proofs for sampled rows of Y
    pub sampled_rows_proofs: Vec<Path<F>>,
    /// merkle proofs for cells at the intersection of sampled rows and cols
    pub cell_proofs: Vec<Path<F>>,
}

impl<F> VerifiableReedSolomon<F> for ZodaVRS<F>
where
    F: FftField,
{
    type PublicParams = ZodaConfig;
    // config, data_width, data_height, transcript
    type ProverKey = (ZodaConfig, usize, usize, IOPattern);
    // config, data_width, data_height, transcript
    type VerifierKey = (ZodaConfig, usize, usize, IOPattern);
    // merkle roots for X, Y, Z
    type Commitment = (MerkleRoot<F>, MerkleRoot<F>, MerkleRoot<F>);
    type Proof = ZodaProof<F>;

    fn setup<R>(
        _max_y_degree: usize,
        _max_x_degree: usize,
        _rng: &mut R,
    ) -> Result<Self::PublicParams, VrsError>
    where
        R: RngCore + CryptoRng,
    {
        Ok(ZodaConfig::default())
    }

    fn preprocess(
        pp: &Self::PublicParams,
        y_degree: usize,
        x_degree: usize,
        _eval_domain: &Radix2EvaluationDomain<F>,
    ) -> Result<(Self::ProverKey, Self::VerifierKey), VrsError> {
        // TODO: (alex) extend to non-square shape
        if y_degree != x_degree {
            return Err(VrsError::InvalidParam(
                "only square data blob for now".to_string(),
            ));
        }
        let mut io = IOPattern::<DefaultHash>::new("ZodaVRS").absorb(32, "col_commit_root");
        io = FieldIOPattern::<F>::challenge_scalars(io, x_degree + 1, "rand_scales");

        Ok((
            (*pp, y_degree + 1, x_degree + 1, io.clone()),
            (*pp, y_degree + 1, x_degree + 1, io),
        ))
    }

    fn compute_shares(
        pk: &Self::ProverKey,
        data: &Matrix<F>,
    ) -> Result<(Self::Commitment, Vec<VrsShare<F, Self>>), VrsError> {
        let config = pk.0;
        assert_eq!(data.width(), pk.1);
        assert_eq!(data.height(), pk.2);
        let encoded_width = data.width() * config.row_blowup;
        let encoded_height = data.height() * config.col_blowup;
        let row_domain = Radix2EvaluationDomain::new(encoded_width).unwrap();
        let col_domain = Radix2EvaluationDomain::new(encoded_height).unwrap();
        let mut merlin = pk.3.to_merlin();

        let (num_col_samples, num_row_samples) = config.num_samples_per_node();
        let num_nodes = config.num_nodes(data.width());

        let total_time = start_timer!(|| ark_std::format!(
            "ZODA::compute shares (Data size: k={}, L={}; 2D-encoded size: n={}, m={})",
            data.width(),
            data.height(),
            encoded_width,
            encoded_height,
        ));

        // 1. encode Lxk into Lxn matrix (row-wise FFT)
        let encode_time = start_timer!(|| "row-wise encode data to X");
        let encoded = Self::interleaved_rs_encode(data, &row_domain)?;
        assert_eq!(encoded.width(), encoded_width);
        end_timer!(encode_time);

        // 2. commit to cols of X, and get the random linear scaling vectors (random diagonal matrix)
        let col_commit_mt = SymbolMerkleTree::new(encoded.col_iter());
        merlin.add_bytes(&col_commit_mt.root()).unwrap();
        let mut scaling_factors = vec![F::default(); data.height()];
        merlin.fill_challenge_scalars(&mut scaling_factors).unwrap();

        // 3. randomize/scale rows and encode Lxk into mxk matrix (col-wise FFT)
        let encode_time = start_timer!(|| "col-wise encode data to Y");
        let mut scaled = data.clone();
        scaled.scale_rows(&scaling_factors).unwrap();
        scaled.transpose_in_place();
        let mut col_encoded = Self::interleaved_rs_encode(&scaled, &col_domain)?;
        assert_eq!(col_encoded.width(), encoded_height);
        end_timer!(encode_time);

        // 4. commit to rows of Y
        let row_commit_mt = SymbolMerkleTree::new(col_encoded.col_iter());

        // 5. 2D encode into Z, and commit every cells
        let encode_time = start_timer!(|| "row-wise encode Y to Z");
        col_encoded.transpose_in_place();
        assert_eq!(col_encoded.width(), data.width());
        assert_eq!(col_encoded.height(), encoded_height);

        let final_encoded = Self::interleaved_rs_encode(&col_encoded, &row_domain)?;
        assert_eq!(final_encoded.width(), encoded_width);
        assert_eq!(final_encoded.height(), encoded_height);
        end_timer!(encode_time);

        let cell_commit_mt = SymbolMerkleTree::from_single_elem_leaves(&final_encoded.all_cells());
        let commits = (
            col_commit_mt.root(),
            row_commit_mt.root(),
            cell_commit_mt.root(),
        );

        // 6. for all nodes (those are the minimum number of nodes to guarantee recovery),
        // each randomly sample a few samples of cols of X, rows of Y
        let mut shares = vec![];
        // we further cache queried cols and rows in memory for faster merkle proof assembly
        let mut col_proofs = HashMap::<usize, Path<F>>::new(); // col_idx, merkle_proof against X
        let mut row_proofs = HashMap::<usize, Path<F>>::new(); // row_idx, merkle_proof against Y
        let mut cell_proofs = HashMap::<(usize, usize), Path<F>>::new(); // (row_idx, col_idx), merkle_proof against Z

        let open_time = start_timer!(|| "Prepare queried cols and rows");
        for node_idx in 0..num_nodes {
            let rng = &mut indexed_rng(node_idx);

            let mut zoda_proof = ZodaProof::default();
            // TODO: (alex) should we change VrsShare for custom associated type for data instead of Vec<F>?
            let mut zoda_data = vec![];
            let mut sampled_cols = BTreeSet::new();
            let mut sampled_rows = BTreeSet::new();

            while sampled_cols.len() < num_col_samples {
                sampled_cols.insert(rng.gen_range(0..encoded_width));
            }
            while sampled_rows.len() < num_row_samples {
                sampled_rows.insert(rng.gen_range(0..encoded_height));
            }

            for &col in sampled_cols.iter() {
                let col_proof = col_proofs
                    .entry(col)
                    .or_insert(col_commit_mt.generate_proof(col));
                zoda_proof.sampled_cols_proofs.push(col_proof.clone());
                zoda_data.extend_from_slice(&encoded.col(col)?);
            }
            for &row in sampled_rows.iter() {
                let row_proof = row_proofs
                    .entry(row)
                    .or_insert(row_commit_mt.generate_proof(row));
                zoda_proof.sampled_rows_proofs.push(row_proof.clone());
                zoda_data.extend_from_slice(&col_encoded.row(row)?);
            }
            for (&row, &col) in iproduct!(sampled_rows.iter(), sampled_cols.iter()) {
                let cell_proof = cell_proofs
                    .entry((row, col))
                    .or_insert(cell_commit_mt.generate_proof(row * encoded_width + col));
                zoda_proof.cell_proofs.push(cell_proof.clone());
            }

            shares.push(VrsShare {
                data: zoda_data,
                proof: zoda_proof,
            });
        }
        end_timer!(open_time);

        end_timer!(total_time);
        Ok((commits, shares))
    }

    fn verify_share(
        vk: &Self::VerifierKey,
        comm: &Self::Commitment,
        idx: usize,
        share: &VrsShare<F, Self>,
    ) -> Result<bool, VrsError> {
        let mut rng = indexed_rng(idx);
        let mut verified = true;
        let config = vk.0;
        let data_width = vk.1;
        let data_height = vk.2;
        let encoded_width = data_width * config.row_blowup;
        let encoded_height = data_height * config.col_blowup;

        // derive the random scaling factors
        // NOTE: I don't want to put the transcript repetitively in the Commitment, so just use merlin (instead of arthur)
        let mut merlin = vk.3.to_merlin();
        merlin.add_bytes(&comm.0).unwrap();
        let mut scaling_factors = vec![F::default(); data_height];
        merlin.fill_challenge_scalars(&mut scaling_factors).unwrap();

        let (num_col_samples, num_row_samples) = config.num_samples_per_node();
        let mut sampled_cols = BTreeSet::new();
        let mut sampled_rows = BTreeSet::new();

        while sampled_cols.len() < num_col_samples {
            sampled_cols.insert(rng.gen_range(0..encoded_width));
        }
        while sampled_rows.len() < num_row_samples {
            sampled_rows.insert(rng.gen_range(0..encoded_height));
        }

        // verify all cols against col_root of X, rows against row_root of Y
        for (j, (&col_idx, proof)) in sampled_cols
            .iter()
            .zip(share.proof.sampled_cols_proofs.iter())
            .enumerate()
        {
            verified &= proof.verify(
                &comm.0,
                col_idx,
                &share.data[j * data_height..(j + 1) * data_height],
            );
        }
        // sampled cols before offset, sampled rows after the offset
        let offset = num_col_samples * data_height;
        for (i, (&row_idx, proof)) in sampled_rows
            .iter()
            .zip(share.proof.sampled_rows_proofs.iter())
            .enumerate()
        {
            verified &= proof.verify(
                &comm.1,
                row_idx,
                &share.data[offset + i * data_width..offset + (i + 1) * data_width],
            );
        }

        let row_domain = Radix2EvaluationDomain::new(encoded_width).unwrap();
        let col_domain = Radix2EvaluationDomain::new(encoded_height).unwrap();

        // LHS: random scale then col-wise encode selected cols, then select sampled rows; we transposed these cols as row vector instead for easier encoding.
        let mut trans_col_matrix =
            Matrix::new(share.data[..offset].to_vec(), data_height, num_col_samples)?;
        trans_col_matrix.scale_cols(&scaling_factors)?;
        let mut selected_encoded_trans_col =
            Self::interleaved_rs_encode(&trans_col_matrix, &col_domain)?
                .retain_cols(&sampled_rows)?;
        selected_encoded_trans_col.transpose_in_place();
        let selected_encoded_col = selected_encoded_trans_col; // name alias after the transpose for readability

        // RHS: row-wise encode selected rows, then select sampled cols
        let row_matrix = Matrix::new(share.data[offset..].to_vec(), data_width, num_row_samples)?;
        let selected_encoded_row =
            Self::interleaved_rs_encode(&row_matrix, &row_domain)?.retain_cols(&sampled_cols)?;

        // Ligero-like check (Step 5)
        verified &= selected_encoded_col == selected_encoded_row;

        for ((&row, &col), &cell, proof) in izip!(
            iproduct!(sampled_rows.iter(), sampled_cols.iter(),),
            selected_encoded_row.cell_iter(),
            share.proof.cell_proofs.iter()
        ) {
            // Per-cell opening proof (Step 6)
            verified &= proof.verify(&comm.2, row * encoded_width + col, [cell]);
        }

        Ok(verified)
    }
}

// the sampled rows/cols are deterministically derived from a PRG seeded with node idx
fn indexed_rng(idx: usize) -> ChaCha20Rng {
    let mut seed = [0u8; 32];
    seed[..std::mem::size_of::<usize>()].copy_from_slice(&idx.to_le_bytes());
    ChaCha20Rng::from_seed(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::test_rng;
    use ark_bn254::Fr;
    use ark_std::UniformRand;

    #[test]
    fn test_advz_vrs() {
        let rng = &mut test_rng();
        let k = 2usize.pow(7);
        let l = 2usize.pow(7);
        let n = 2usize.pow(8);

        let pp = ZodaVRS::<Fr>::setup(k - 1, l - 1, rng).unwrap();
        let domain = Radix2EvaluationDomain::<Fr>::new(n).unwrap(); // effectively unused
        let (pk, vk) = ZodaVRS::preprocess(&pp, k - 1, l - 1, &domain).unwrap();

        let data = (0..k * l).map(|_| Fr::rand(rng)).collect();
        let data = Matrix::new(data, k, l).unwrap();
        let (cm, shares) = ZodaVRS::compute_shares(&pk, &data).unwrap();

        for (idx, share) in shares.iter().enumerate() {
            assert!(ZodaVRS::verify_share(&vk, &cm, idx, share).unwrap());
        }
    }
}
