use ark_bn254::Fr;
use ark_ff::One;
use ark_serialize::CanonicalSerialize;

#[derive(Debug, Clone)]
pub struct MultilinearPoly(Vec<Fr>);

impl MultilinearPoly {
    pub fn new(coeff: Vec<Fr>) -> MultilinearPoly {
        MultilinearPoly(coeff)
    }

    pub fn eval(mut self, point: &[Fr]) -> Fr {
        assert_eq!(1 << point.len(), self.0.len());
        let mut len = self.0.len();
        for i in point.iter().rev() {
            len >>= 1;
            for j in 0..len {
                let t = self.0[j + len] * i;
                self.0[j] += t;
            }
        }
        self.0[0]
    }

    pub fn partial_eval(mut self, point: &[Fr]) -> MultilinearPoly {
        let mut len = self.0.len();
        for i in point.iter().rev() {
            len >>= 1;
            for j in 0..len {
                let t = self.0[j + len] * i;
                self.0[j] += t;
            }
        }
        self.0.truncate(len);
        MultilinearPoly(self.0)
    }
}

#[derive(Debug, Clone)]
pub struct UniVarPoly(Vec<Fr>);

impl UniVarPoly {
    pub fn new(coeff: Vec<Fr>) -> UniVarPoly {
        UniVarPoly(coeff)
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = vec![];
        self.0.iter().for_each(|x| {
            <Fr as CanonicalSerialize>::serialize_compressed(&x, &mut bytes).unwrap()
        });
        bytes
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn eval(&self, point: &Fr) -> Fr {
        let mut res = self.0.last().unwrap().clone();
        for i in self.0.iter().rev().skip(1) {
            res *= point;
            res += i;
        }
        res
    }
}

#[derive(Debug, Clone)]
pub struct UniPolyEvals {
    evals: Vec<Fr>,
    offset_inv: Fr,
}

impl UniPolyEvals {
    pub fn new(evals: Vec<Fr>, offset_inv: Fr) -> UniPolyEvals {
        UniPolyEvals { evals, offset_inv }
    }

    pub fn len(&self) -> usize {
        self.evals.len()
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = vec![];
        self.evals.iter().for_each(|x| {
            <Fr as CanonicalSerialize>::serialize_compressed(&x, &mut bytes).unwrap()
        });
        bytes
    }

    pub fn n_th_eval(&self, n: usize) -> Fr {
        self.evals[n & (self.evals.len() - 1)]
    }

    pub fn eval(self, mut point: Fr, mut root_inv: Fr, inv_2: Fr) -> Fr {
        let UniPolyEvals {
            mut evals,
            mut offset_inv,
        } = self;
        let mut len = evals.len();
        let mut inv = <Fr as One>::one();
        for _ in 0..evals.len().ilog2() {
            len >>= 1;
            let mut w = offset_inv;
            for j in 0..len {
                let t = (evals[j] - evals[j + len]) * point * w;
                evals[j] = evals[j] + evals[j + len] + t;
                w *= root_inv;
            }
            offset_inv *= offset_inv;
            inv *= inv_2;
            point *= point;
            root_inv *= root_inv;
        }
        evals[0] * inv
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::{Field, UniformRand};
    use rand::thread_rng;
    use util::mul_group::Radix2Group;

    use crate::poly::{UniPolyEvals, UniVarPoly};

    use super::MultilinearPoly;

    #[test]
    fn test_partial_eval() {
        let nv = 12;
        let mut rng = thread_rng();
        let coeff = (0..(1 << nv))
            .map(|_| <Fr as UniformRand>::rand(&mut rng))
            .collect::<Vec<_>>();
        let point = (0..nv)
            .map(|_| <Fr as UniformRand>::rand(&mut rng))
            .collect::<Vec<_>>();
        let poly = MultilinearPoly::new(coeff);
        let y1 = poly.clone().eval(&point);
        let poly = poly.partial_eval(&point[5..]);
        let y2 = poly.eval(&point[..5]);
        assert_eq!(y1, y2);

        let coeff = (0..(1 << nv))
            .map(|_| <Fr as UniformRand>::rand(&mut rng))
            .collect::<Vec<_>>();
        let group = Radix2Group::new(nv);
        let evals = group.fft(coeff.clone());

        let uni_poly = UniVarPoly::new(coeff);
        let uni_evals = UniPolyEvals::new(evals, 1.into());

        let point = <Fr as UniformRand>::rand(&mut rng);
        assert_eq!(
            uni_poly.eval(&point),
            uni_evals.eval(
                point,
                group.element_inv_at(1),
                <Fr as Field>::inverse(&2.into()).unwrap()
            )
        );
    }
}
