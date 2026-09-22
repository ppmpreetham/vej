use burn::{Tensor, tensor::Int};

// jev composite reward score (RLCD fn)
pub fn proper_reward(
    q: Tensor<3>,
    target: Tensor<2>,
    qtype: Tensor<1, Int>,
    mask: Tensor<2>,
) -> Tensor<1> {
    let mask_3d = mask.clone().unsqueeze_dim::<3>(1);
    let target_3d = target.unsqueeze_dim::<3>(1);
    let q = q * mask_3d.clone();
    let logq = q.clone().clamp_min(1e-12).log().clamp_min(-9.21);
    let log_score = (target_3d.clone() * logq).sum_dim(1);
    let q_l2 = q.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-9);
    let sph = (target_3d.clone() * q.clone()).sum_dim(1) / q_l2;
    let mut r = log_score + sph * 0.5;
    let is_score = qtype.equal_elem(1).float();
    let sum_score = is_score.clone().sum().into_scalar::<f32>();
    if sum_score > 0.0 {
        let k = mask.sum_dim(1).unsqueeze_dim::<3>(1).clamp_min(2.0);
        let cdf_q = q.cumsum(1);
        let cdf_t = target_3d.cumsum(1);
        let rps = ((cdf_q - cdf_t).powf_scalar(2.0) * mask_3d).sum_dim(1) / (k - 1.0);
        let is_score_3d = is_score.unsqueeze_dim::<2>(1).unsqueeze_dim::<3>(2);
        r = r - rps * is_score_3d;
    }
    r.squeeze_dim::<2>(2).squeeze_dim::<1>(1)
}

/// max shannon entropy vonfidence
pub fn confidence_from_probs(p: &[f32], k: usize) -> f32 {
    if k < 2 {
        return 1.0;
    }
    let ent: f32 = p[..k]
        .iter()
        .map(|&x| {
            let x = x.clamp(1e-12, 1.0);
            -x * x.ln()
        })
        .sum();
    (1.0 - ent / (k as f32).ln()).clamp(0.0, 1.0)
}

// 0.5 to 5
pub fn clamp_temperature(t: f64) -> f64 {
    if !t.is_finite() {
        1.0
    } else {
        t.clamp(0.5, 5.0)
    }
}

// qtype, k -> bucket name
pub fn temp_bucket(qtype: u8, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    let name = match qtype {
        0 => "choice",
        1 => "score",
        _ => "noul",
    };
    format!("{name}:{size}")
}
