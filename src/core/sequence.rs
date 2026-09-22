use std::borrow::Cow;

use serde_json::Value;
use tokenizers::Tokenizer;
use super::json::{Query, render_options};

pub fn sequence(
    tokenizer: &Tokenizer,
    state: &Value,
    q: &Query<'_>,
    max_len: usize,
    head_max_len: usize,
    option_order: Option<&[usize]>,
    truncate_left: bool,
) -> (Vec<u32>, Vec<usize>) {
    let cls = tokenizer.token_to_id("[CLS]").unwrap_or(101);
    let head = tokenizer.token_to_id("[SEP]").unwrap_or(102);
    let mask = tokenizer.token_to_id("[MASK]").unwrap_or(103);

    // head tokenizer
    let mut head_ids = tokenizer
        .encode(format!("{} question: {}", q.t.as_ref(), q.ins), false)
        .map(|e| e.get_ids().to_vec())
        .unwrap_or_default();

    let opts = render_options(q);
    // vector of option token ids
    let mut opt_ids: Vec<Vec<u32>> = option_order
        .map(|order| order.iter().filter_map(|&i| opts.get(i)).collect())
        .unwrap_or_else(|| opts.iter().collect::<Vec<_>>())
        .into_iter()
        .map(|opt| {
            let mut ids = vec![mask];
            if let Ok(e) = tokenizer.encode(format!(" {}", opt.replace("[MASK]", " ")), false) {
                let t = e.get_ids();
                // max 48 tokens per option
                ids.extend_from_slice(&t[..t.len().min(48)]);
            }
            ids
        })
        .collect();

    // budget for options
    let output_len: usize = opt_ids.iter().map(|o| o.len()).sum();
    let budget = head_max_len.saturating_sub(output_len);

    let head_budget = if budget < 16 {
        let per_option = 4.max(head_max_len.saturating_sub(16) / opt_ids.len().max(1));
        opt_ids.iter_mut().for_each(|o| o.truncate(per_option));
        head_max_len.saturating_sub(opt_ids.iter().map(|o| o.len()).sum())
    } else {
        budget
    };
    head_ids.truncate(8.max(head_budget));

    let mut ids = Vec::with_capacity(max_len);
    ids.push(cls);
    ids.extend(head_ids);
    ids.push(head);

    let markers: Vec<usize> = opt_ids
        .into_iter()
        .map(|o| {
            let idx = ids.len();
            ids.extend(o);
            idx
        })
        .collect();

    ids.push(head);

    let room = max_len.saturating_sub(ids.len() + 1);
    if room > 0 {
        let criterion_str = state.as_str().map(Cow::Borrowed).unwrap_or_else(||Cow::Owned(state.to_string()));
        if let Ok(e) = tokenizer.encode(criterion_str.replace("[MASK]", " "), false) {
            let st = e.get_ids();
            if truncate_left {
                ids.extend_from_slice(&st[st.len().saturating_sub(room)..]);
            } else {
                ids.extend_from_slice(&st[..st.len().min(room)]);
            }
        }
    }

    ids.push(head);
    ids.truncate(max_len);
    let active_markers = markers.into_iter().filter(|&m| m < max_len).collect();
    (ids, active_markers)
}
