// For randomness (during paramgen and proof generation)
use rand::thread_rng;

// For benchmarking
use std::time::{Duration, Instant};

// Bring in some tools for using pairing-friendly curves
use ff::{Field, ScalarEngine};
use pairing::Engine;

// We're going to use the BLS12-381 pairing-friendly elliptic curve.
use pairing::bls12_381::Bls12;

// We'll use these interfaces to construct our circuit.
use bellman::{Circuit, ConstraintSystem, SynthesisError, LinearCombination};

// We're going to use the Groth16 proving system.
use bellman::groth16::{
    create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof, Proof,
};

use std::marker::PhantomData;

pub struct Benchmark<E: Engine> {
    num_constraints: usize,
    _engine: PhantomData<E::Fr>,
}

impl<E: Engine> Benchmark<E> {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            num_constraints,
            _engine: PhantomData,
        }
    }
}

impl<E: Engine> Circuit<E> for Benchmark<E> {
    fn synthesize<CS: ConstraintSystem<E>>(
    self,
    cs: &mut CS,
    ) -> Result<(), SynthesisError> {
        let mut assignments = Vec::new();
        let mut a_val = E::Fr::one();
        let mut a_var = cs.alloc_input(|| "a", || Ok(a_val))?;
        assignments.push((a_val, a_var));

        let mut b_val = E::Fr::one();
        let mut b_var = cs.alloc_input(|| "b", || Ok(b_val))?;
        assignments.push((a_val, a_var));

        for i in 0..self.num_constraints - 1 {
            if i % 2 != 0 {
                let mut c_val = a_val;
                c_val.mul_assign(&b_val);
                let c_var = cs.alloc(|| format!("{}", i), || Ok(c_val))?;

                cs.enforce(
                    || format!("{}: a * b = c", i),
                    |lc| lc + a_var,
                    |lc| lc + b_var,
                    |lc| lc + c_var,
                );

                assignments.push((c_val, c_var));
                a_val = b_val;
                a_var = b_var;
                b_val = c_val;
                b_var = c_var;
            } else {
                let mut c_val = a_val;
                c_val.add_assign(&b_val);
                let c_var = cs.alloc(|| format!("{}", i), || Ok(c_val))?;

                cs.enforce(
                    || format!("{}: a + b = c", i),
                    |lc| lc + a_var + b_var,
                    |lc| lc + CS::one(),
                    |lc| lc + c_var,
                );

                assignments.push((c_val, c_var));
                a_val = b_val;
                a_var = b_var;
                b_val = c_val;
                b_var = c_var;
            }
        }

        let mut a_lc = LinearCombination::zero();
        let mut b_lc = LinearCombination::zero();
        let mut c_val = E::Fr::zero();

        for (val, var) in assignments {
            a_lc = a_lc + var;
            b_lc = b_lc + var;
            c_val.add_assign(&val);
        }
        c_val.square();

        let c_var = cs.alloc(|| "c_val", || Ok(c_val))?;

        cs.enforce(
            || "assignments.sum().square()",
            |_| a_lc,
            |_| b_lc,
            |lc| lc + c_var,
        );

        Ok(())
    }
}
