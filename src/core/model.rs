use bert_burn::{
    data::BertInferenceBatch,
    model::{BertModel, BertModelOutput},
};
use burn::tensor::Device;
use burn::{
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    tensor::{activation, Bool, Int, Tensor},
};

#[derive(Module, Debug)]
pub struct DecisionModel {
    pub encoder: BertModel,
    head: Option<TransformerEncoder>,
    type_emb: Embedding,
    scorer_ln: LayerNorm,
    scorer1: Linear,
    scorer2: Linear,
    act1: Linear,
    act2: Linear,
}

impl DecisionModel {
    pub fn new(
        encoder: BertModel,
        d: usize,
        head_layers: usize,
        n_act: usize,
        device: &Device,
    ) -> Self {
        let nhead = (d / 64).max(1);
        let head = (head_layers > 0).then(|| {
            TransformerEncoderConfig::new(d, 4 * d, nhead, head_layers)
                .with_norm_first(true)
                .init(device)
        });

        Self {
            encoder,
            head,
            type_emb: EmbeddingConfig::new(3, d).init(device),
            scorer_ln: LayerNormConfig::new(d).init(device),
            scorer1: LinearConfig::new(d, d).init(device),
            scorer2: LinearConfig::new(d, 1).init(device),
            act1: LinearConfig::new(d + 4, 256).init(device),
            act2: LinearConfig::new(256, n_act).init(device),
        }
    }

    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        attention_mask: Tensor<2>,
        marker_pos: Tensor<2, Int>,
        marker_mask: Tensor<2, Bool>,
        qtype: Tensor<1, Int>,
        detach: bool,
    ) -> (Tensor<2>, Tensor<2>) {
        let mask_pad = attention_mask.clone().equal_elem(0);
        let batch = BertInferenceBatch {
            tokens: input_ids,
            mask_pad,
        };

        let BertModelOutput {
            mut hidden_states, ..
        } = self.encoder.forward(batch);

        if detach {
            hidden_states = hidden_states.detach();
        }

        let te = self.type_emb.forward(qtype.unsqueeze_dim(1));
        let mut h = hidden_states + te;

        if let Some(ref head) = self.head {
            let pad = attention_mask.equal_elem(0);
            h = head.forward(TransformerEncoderInput::new(h).mask_pad(pad));
        }

        let d = h.dims()[2];
        let idx = marker_pos
            .clone()
            .unsqueeze_dim::<3>(2)
            .expand([marker_pos.dims()[0], marker_pos.dims()[1], d]);
        let m = h.clone().gather(1, idx);

        let logits = self
            .scorer2
            .forward(activation::gelu(self.scorer1.forward(self.scorer_ln.forward(m))))
            .squeeze_dim(2);
        let logits = logits.mask_fill(marker_mask.clone().bool_not(), -1e4);

        let p = activation::softmax(logits.clone().detach(), 1);
        let k = marker_mask.float().sum_dim(1).clamp_min(2.0);
        let ent = -(p.clone() * p.clone().clamp_min(1e-9).log()).sum_dim(1) / k.clone().log();

        let top2 = if p.dims()[1] >= 2 {
            p.topk(2, 1)
        } else {
            let t1 = p.topk(1, 1);
            Tensor::cat(vec![t1.clone(), Tensor::zeros_like(&t1)], 1)
        };

        let feats = Tensor::<1>::stack(
            vec![
                top2.clone().narrow(1, 0, 1).squeeze_dim::<1>(1),
                (top2.clone().narrow(1, 0, 1) - top2.narrow(1, 1, 1)).squeeze_dim::<1>(1),
                ent.squeeze_dim::<1>(1),
                (k / 255.0).squeeze_dim::<1>(1),
            ],
            1,
        );

        let pooled = h.narrow(1, 0, 1).squeeze_dim::<2>(1);
        let act = self.act2.forward(activation::gelu(
            self.act1.forward(Tensor::cat(vec![pooled, feats], 1)),
        ));

        (logits, act)
    }
}
