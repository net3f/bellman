use ff::{Field, PrimeField};
use group::{CurveAffine, CurveProjective};
use pairing::{Engine, PairingCurveAffine};

use rand_core::RngCore;

use super::{PreparedVerifyingKey, Proof, VerifyingKey};

use crate::SynthesisError;

pub fn prepare_verifying_key<E: Engine>(vk: &VerifyingKey<E>) -> PreparedVerifyingKey<E> {
    let mut gamma = vk.gamma_g2;
    gamma.negate();
    let mut delta = vk.delta_g2;
    delta.negate();

    PreparedVerifyingKey {
        alpha_g1_beta_g2: E::pairing(vk.alpha_g1, vk.beta_g2),
        neg_gamma_g2: gamma.prepare(),
        neg_delta_g2: delta.prepare(),
        ic: vk.ic.clone(),
    }
}

pub fn verify_proof<'a, E: Engine>(
    pvk: &'a PreparedVerifyingKey<E>,
    proof: &Proof<E>,
    public_inputs: &[E::Fr],
) -> Result<bool, SynthesisError> {
    if (public_inputs.len() + 1) != pvk.ic.len() {
        return Err(SynthesisError::MalformedVerifyingKey);
    }

    let mut acc = pvk.ic[0].into_projective();

    for (i, b) in public_inputs.iter().zip(pvk.ic.iter().skip(1)) {
        acc.add_assign(&b.mul(i.into_repr()));
    }

    // The original verification equation is:
    // A * B = alpha * beta + inputs * gamma + C * delta
    // ... however, we rearrange it so that it is:
    // A * B - inputs * gamma - C * delta = alpha * beta
    // or equivalently:
    // A * B + inputs * (-gamma) + C * (-delta) = alpha * beta
    // which allows us to do a single final exponentiation.

    Ok(E::final_exponentiation(&E::miller_loop(
        [
            (&proof.a.prepare(), &proof.b.prepare()),
            (&acc.into_affine().prepare(), &pvk.neg_gamma_g2),
            (&proof.c.prepare(), &pvk.neg_delta_g2),
        ]
        .iter(),
    ))
    .unwrap()
        == pvk.alpha_g1_beta_g2)
}

// Verifies a batch of proofs of similar (having the same verification key) statements
// The code follows Zcash Protocol Specification, section B.2
// https://github.com/zcash/zips/blob/master/protocol/sapling.pdf
pub fn verify_batch<E: Engine, R: RngCore>(
    vk: &VerifyingKey<E>,
    proofs: &Vec<Proof<E>>,
    public_inputs: &Vec<Vec<E::Fr>>,
    rng: &mut R
) -> Result<bool, SynthesisError> {
    let n = proofs.len();
    let l = public_inputs[0].len();

    let mut acc_ab =  Vec::with_capacity(n);
    let mut acc_delta = E::G1Affine::zero().into_projective();
    let mut acc_gamma = vec![E::Fr::zero(); l];
    let mut acc_y = E::Fr::zero();

    for (proof, inputs) in proofs.iter().zip(public_inputs.iter()) {
        let z = E::Fr::random(rng);

        // acc_ab ||= ([zj]Aj, -Bj)
        let za = proof.a.mul(z).into_affine();
        let mut b_neg = proof.b;
        b_neg.negate();
        acc_ab.push((za.prepare(), b_neg.prepare()));

        // acc_delta += [zj]Cj
        acc_delta.add_assign(&proof.c.mul(z));

        // acc_gamma_i += zj * aji
        // NB: there's also an implicit input aj0 = 1, but zj * aj0 = zj are being accumulated in acc_y
        for (acc_gamma_i, &inputs_i) in acc_gamma.iter_mut().zip(inputs.iter()) {
            let mut ai = inputs_i;
            ai.mul_assign(&z);
            (*acc_gamma_i).add_assign(&ai);
        }

        // acc_y += zj
        acc_y.add_assign(&z);
    }

    // n+2 pairs of points from G1xG2 to compute a multi Miller loop
    let mut miller_loop_args = Vec::with_capacity(n+2);

    // n pairs from acc_ab = [([zj]Aj, -Bj)]
    for (a, b) in acc_ab.iter() {
        miller_loop_args.push((a, b))
    }

    // (acc_delta, delta)
    let acc_delta_ = acc_delta.into_affine().prepare();
    let delta = vk.delta_g2.prepare();
    miller_loop_args.push((&acc_delta_, &delta));

    // (psi, gamma)
    let mut acc_psi = vk.ic[0].mul(acc_y); // corresponds to implicit public inputs aj0 = 1
    for (acc_gamma_i, psi) in acc_gamma.iter().zip(vk.ic.iter().skip(1)) {
        acc_psi.add_assign(&psi.mul(acc_gamma_i.into_repr()));
    }
    let acc_psi_ = acc_psi.into_affine().prepare();
    let gamma = vk.gamma_g2.prepare();
    miller_loop_args.push((&acc_psi_, &gamma));

    let mut result = E::final_exponentiation(&E::miller_loop(miller_loop_args.iter())).unwrap();
    let y = E::pairing(vk.alpha_g1, vk.beta_g2);
    result.mul_assign(&y.pow(acc_y.into_repr().as_ref()));

    Ok(result == E::Fqk::one())
}
