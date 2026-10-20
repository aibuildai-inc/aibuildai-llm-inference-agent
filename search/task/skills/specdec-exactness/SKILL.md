---
name: specdec-exactness
description: >-
  How to prove a speculative decoder is lossless, and how to tell a real
  correctness bug from a broken test. Covers why the check must run in fp32,
  what the reference implementation must be, comparing sequences of unequal
  length, reading the top-2 logit gap at a divergence, distribution testing for
  sampling, and the specific harness defects that produce false failures. Read
  this before concluding that an implementation is wrong.
---

# Proving losslessness

Speculative decoding is only interesting if the output is the output the target
model would have produced anyway. That claim needs a test whose failures mean
something.

## Greedy token identity is the oracle

At temperature 0, the method's output must be **token-identical** to ordinary
greedy decoding with the same weights. Not "similar text", not "same meaning":
the same ids, in the same order.

## Run the check in fp32, not bf16

In bf16 a KV-cached decode step and a batched verification forward round
near-tied logits differently. The two paths then legitimately pick different
tokens at an exact tie, and the sequences diverge for reasons that have nothing
to do with the accept rule. A bf16 mismatch is therefore **not evidence of
non-exactness**, and a bf16 match is weak evidence of it.

Load the models in fp32 for the gate. Time in bf16. These are different runs
with different purposes; do not merge them.

Independently: call `.float()` on logits before every softmax, in every regime.
Acceptance ratios computed in low precision are unreliable even when the
weights are bf16.

## The reference must be the same weights, the simplest possible loop

Compare against a manual, uncached, argmax loop that calls the **same model
object** the method uses. Do not compare against a separately loaded copy, and
do not compare against a high-level generation helper: if the method's pipeline
uses a modified model class or a custom cache, a separately loaded stock model
is a different program, and you will be debugging the difference between two
references instead of testing the method.

Before every reference forward pass, clear any attention mask, tree mask or
other state the method set on the model. A leftover mask makes a later plain
forward of a different length fail in a way that looks like a correctness bug.

## Compare over the overlap, and read the gap

A speculative path routinely overshoots the requested token count, because a
cycle commits several tokens at once. Compare over `min(len(reference),
len(candidate))`. A test that requires equal lengths fails by construction on
every prompt.

At the first divergence, print the **reference's top-2 logit gap**. A gap of
zero, or of a couple of units in the last place, means the two paths split at a
tie — rounding, not a broken accept rule. A large gap means a real bug.

## For sampling, identity is the wrong test

Above temperature 0 the output is not deterministic, so token identity cannot
apply. Test the *distribution* instead: draw many single tokens from the method
in a fixed context, draw as many from the target's own distribution, and
compare the two with a chi-square test. Use a permissive threshold and many
draws; a fixed-context test with a few thousand draws is enough to catch an
accept rule that is wrong in the mean.

Sampling transforms — temperature, top-k, top-p — must be applied identically
to draft and target logits *before* the acceptance ratio is computed. Applying
them to one side only is a common and silent source of distribution skew.

## A generation helper is not plain greedy

Instruction-tuned checkpoints routinely ship a `generation_config.json` carrying
`repetition_penalty`, `temperature`, `top_p` and `top_k`. A repetition penalty is
a **logits processor**, so the library's generation helper applies it *even when
sampling is switched off*. The result is not argmax over the model's own logits,
and it will not match a plain argmax reference — the divergence appears at tokens
the penalty re-ranked, with a large top-2 gap rather than a tie.

So: decide which definition of "what the model would have said" your task uses,
state it once, and make both sides obey it. If that definition is plain argmax
and you call a generation helper anywhere on the decision path, neutralise every
sampling knob explicitly (`repetition_penalty=1.0, temperature=1.0, top_p=1.0,
top_k=0`) rather than trusting `do_sample=False` to mean "no processors".

A large top-2 gap at the first divergence is the tell: rounding produces ties,
a different decoding rule produces confident disagreement.

## Suspect the test before the implementation

When a correctness check fails, the most likely explanation is the check. The
recurring defect classes, in rough order of frequency:

1. The reference is a different program (separately loaded model, different
   class, different cache).
2. A pass criterion that cannot hold — most often an equal-length requirement
   against a generator that overshoots.
3. Low-precision rounding at a tie.
4. An assumption that a helper method works because the class inherits it, when
   the class overrides the forward path it depends on.
5. State left on the model by the previous call.

Rule out all five before changing the accept rule. Changing a correct accept
rule to satisfy a broken test is the expensive failure mode here.

## Verify the path you measured, not a neighbouring one

A correctness check only covers the code it actually executed. This is easy to
lose when a harness uses one configuration for verification and another for
timing -- a smaller model to keep the exactness check affordable, say, while the
speed comes from a larger one.

If the method has a fallback branch, that split can let the verification pass on
the fallback while every reported number comes from the branch that was never
checked. The gate reports success, the claim is unsupported, and nothing in the
output says so.

Before trusting a verification, ask which branch ran. If the checked and the
measured configurations differ, either bring them together or state the gap
explicitly in what you report. A method that reports "gate passed" while its
fast path is untested has documented nothing.

When the reference and the method cannot be resident at once -- a large model at
full precision is easily too big for one card twice over -- run them in
sequence: collect the reference outputs, free the model, then load the method
and compare. Peak memory is then one copy, not two.

## Methods that are deliberately not exact

Some acceptance rules trade exactness for speed and are not distribution-exact
by design. They are legitimate to study, but they must be labelled, kept behind
a flag, and never reported as lossless. If a variant cannot pass the gate, the
honest report is "faster, not exact", not a higher headline number.
