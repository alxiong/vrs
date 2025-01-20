//! Twin Polynomials conversions

use ark_ff::Field;
use ark_poly::{
    univariate::DensePolynomial, DenseMultilinearExtension, MultilinearExtension, Polynomial,
};
use ark_std::{end_timer, iter::successors, log2, start_timer};
use p3_maybe_rayon::prelude::*;

use crate::poly::bivariate;

/// Establishing conversion of twin polynomials
pub trait MultilinearTwin<F: Field>: Polynomial<F> {
    type Twin: MultilinearExtension<F>;

    /// Return the twin polynomial of `poly`
    fn twin(&self) -> Self::Twin;

    /// Convert the corresponding evaluation point at which evaluations on twin poly is the same
    fn twin_point(
        &self,
        point: &<Self as Polynomial<F>>::Point,
    ) -> <Self::Twin as Polynomial<F>>::Point;
}

impl<F: Field> MultilinearTwin<F> for DensePolynomial<F> {
    type Twin = DenseMultilinearExtension<F>;

    /// Given a univariate poly, derive the MLE of a multilinear poly sharing the same coeff vector.
    /// Basically evaluating coeff-form MLP over boolean hypercube.
    // TODO: (alex) parallelized this?
    fn twin(&self) -> Self::Twin {
        let mut evals = self.coeffs.clone();
        let nv = log2(self.coeffs.len()) as usize;
        evals.resize(1 << nv, F::zero());

        let timer = start_timer!(|| ark_std::format!("Prepare Twin MLE (nv={})", nv));
        twin_internal(&mut evals, nv);
        end_timer!(timer);

        DenseMultilinearExtension::from_evaluations_vec(nv, evals)
    }

    /// f(x) = mlp(x, x^2, x^4, .. , x^{2^{nv-1}})
    /// where f is the univariate poly sharing the same coefficient as mlp, both in big-endian/natural expansion
    ///
    /// f(x) = c0 + c1*x + c2*x^2 + c3*x^3
    /// mlp(x1=x, x2=x^2) = c0 + c1*x1 + c2*x2 + c3*x1*x2 = f(x)
    fn twin_point(&self, point: &F) -> Vec<F> {
        let nv = log2(self.coeffs.len()) as usize;
        successors(Some(*point), |&prev| Some(prev.square()))
            .take(nv)
            .collect()
    }
}

#[inline(always)]
fn twin_internal<F: Field>(evals: &mut [F], nv: usize) {
    // for each dim, "expand" the coeff along that dim
    // for each chunk size of 2^i, split into two halves: low (subset where x_i=0) and high (x_i=1),
    // add the low to high to account for x_i=1
    for i in 0..nv {
        let step = 1 << (i + 1);
        let half = step >> 1;
        for chunk in evals.chunks_mut(step) {
            for j in 0..half {
                chunk[half + j] += chunk[j];
            }
        }
    }
}

impl<F: Field> MultilinearTwin<F> for bivariate::DensePolynomial<F> {
    type Twin = DenseMultilinearExtension<F>;

    /// similar to univariate->MLE, just concatenating all the coeffs for each row
    fn twin(&self) -> Self::Twin {
        let nv_x = log2(self.deg_x + 1) as usize;
        let nv_y = log2(self.deg_y + 1) as usize;
        let nv = nv_x + nv_y;

        // concat the all columns, following the order of (X, Y), i.e. x-axis first, y-axis second
        let mut evals: Vec<F> = (0..=self.deg_y)
            .into_par_iter()
            .flat_map(move |col_idx| {
                let mut col: Vec<F> = self.coeffs.iter().map(|row| row[col_idx].clone()).collect();
                col.resize(1 << nv_x, F::zero());
                col
            })
            .collect();
        evals.resize(1 << nv, F::zero());

        twin_internal(&mut evals, nv);

        DenseMultilinearExtension::from_evaluations_vec(nv, evals)
    }

    /// similar to univariate->MLE, expand along each axis separately
    fn twin_point(
        &self,
        point: &<Self as Polynomial<F>>::Point,
    ) -> <Self::Twin as Polynomial<F>>::Point {
        let nv_x = log2(self.deg_x + 1) as usize;
        let nv_y = log2(self.deg_y + 1) as usize;

        successors(Some(point.0), |&prev| Some(prev.square()))
            .take(nv_x)
            .chain(successors(Some(point.1), |&prev| Some(prev.square())).take(nv_y))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_poly::DenseUVPolynomial;
    use ark_std::UniformRand;
    use itertools::izip;

    use super::*;
    use crate::test_utils::test_rng;

    #[test]
    fn test_uv_twin_poly() {
        let rng = &mut test_rng();

        for d in 16..20 {
            let p = DensePolynomial::<Fr>::rand(d, rng);
            let twin = p.twin();

            for _ in 0..10 {
                let point = Fr::rand(rng);
                assert_eq!(p.evaluate(&point), twin.evaluate(&p.twin_point(&point)));
            }
        }
    }

    #[test]
    fn test_bv_twin_poly() {
        let rng = &mut test_rng();
        for (d_x, d_y) in izip!(0..5, 8..12) {
            let p = bivariate::DensePolynomial::rand(d_x, d_y, rng);
            let twin = p.twin();
            for _ in 0..10 {
                let point = (Fr::rand(rng), Fr::rand(rng));
                let twin_point = p.twin_point(&point);
                assert_eq!(p.evaluate(&point), twin.evaluate(&twin_point));
            }
        }
    }
}
