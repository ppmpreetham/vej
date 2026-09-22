use serde::Deserialize;
use serde_json::value::RawValue;
use std::borrow::Cow;
use std::collections::BTreeMap;
use strum_macros::{Display, EnumString};

#[derive(Debug, PartialEq, Eq, Display, EnumString, Deserialize)]
#[strum(serialize_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum QType { Choice, Score, Noul }

pub fn render_criterion(raw: &RawValue) -> &str {
    if let Ok(s) = serde_json::from_str::<&str>(raw.get()) {
        s
    } else {
        raw.get()
    }
}

#[derive(Deserialize)]
pub struct Query<'a> {
    pub t: QType,
    #[serde(borrow)]
    pub crit: Option<&'a RawValue>,
}

pub fn render_options<'a>(q: &'a Query<'a>) -> Vec<Cow<'a, str>> {
    let is_empty = |r: &RawValue| r.get() == "null" || r.get() == "\"\"";

    match q.t {
        QType::Choice => q.crit
            .and_then(|c| serde_json::from_str::<BTreeMap<Cow<'a, str>, &'a RawValue>>(c.get()).ok())
            .map(|map| {
                map.into_iter().map(|(k, v)| {
                    if is_empty(v) { k } else { Cow::Owned(format!("{k}: {}", render_criterion(v))) }
                }).collect()
            })
            .unwrap_or_default(),

        QType::Score => q.crit
            .and_then(|c| serde_json::from_str::<Vec<&'a RawValue>>(c.get()).ok())
            .map(|list| {
                list.into_iter().enumerate().map(|(i, c)| {
                    Cow::Owned(format!("level {i}: {}", render_criterion(c)))
                }).collect()
            })
            .unwrap_or_default(),

        QType::Noul => {
            let map = q.crit.and_then(|c| serde_json::from_str::<BTreeMap<Cow<'a, str>, &'a RawValue>>(c.get()).ok());

            let false_opt = map.as_ref()
                .and_then(|m| m.get("false"))
                .copied()
                .filter(|v| !is_empty(v))
                .map(|v| Cow::Owned(format!("false: {}", render_criterion(v))))
                .unwrap_or(Cow::Borrowed("false: no, the statement does not hold"));

            let true_opt = map.as_ref()
                .and_then(|m| m.get("true"))
                .copied()
                .filter(|v| !is_empty(v))
                .map(|v| Cow::Owned(format!("true: {}", render_criterion(v))))
                .unwrap_or(Cow::Borrowed("true: yes, the statement holds"));

            vec![false_opt, true_opt]
        }
    }
}
