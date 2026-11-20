"""Report generation with frozen decoding settings (plan section 6.1): greedy, max_new_tokens = 160,
repetition penalty 1.0. Frozen at G3."""
import torch

from clinguard.data.dataset import batch_to, build_batch

DECODING = dict(do_sample=False, max_new_tokens=160, repetition_penalty=1.0)


@torch.no_grad()
def generate_findings(model, processor, examples, batch_size=4, **overrides):
    """Generated FINDINGS text for each example, in order."""
    settings = {**DECODING, **overrides}
    device = next(p for p in model.parameters()).device
    was_training, use_cache = model.training, model.config.use_cache
    model.eval()
    model.config.use_cache = True
    out = []
    try:
        for i in range(0, len(examples), batch_size):
            enc = batch_to(build_batch(processor, examples[i:i + batch_size], with_targets=False), device)
            ids = model.generate(**enc, **settings, pad_token_id=processor.tokenizer.pad_token_id)
            new = ids[:, enc["input_ids"].shape[1]:]
            out.extend(t.strip() for t in processor.tokenizer.batch_decode(new, skip_special_tokens=True))
    finally:
        model.train(was_training)
        model.config.use_cache = use_cache
    return out
