---
name: specdec-methods
description: >-
  The design space of speculative decoding: how the accept/reject rule works and
  why it is exact, the index arithmetic that is the most common bug, and the
  families of draft proposer (separate draft model, n-gram and prompt lookup,
  self-drafting heads, a lightweight head over the target's own hidden states),
  plus tree-shaped candidates and their attention mask. Read this when choosing
  or implementing a drafting strategy.
---

# The design space

Speculative decoding splits decoding into two jobs: a cheap **proposer** guesses
several next tokens, and the expensive target model **verifies** them in one
forward pass. The win comes from committing more than one token per target
pass; the cost is whatever the proposer charges.

## The accept rule, and why it is exact

For each drafted position, let `q(x)` be the proposer's probability of the token
it drew and `p(x)` the target's probability for that same position. Accept with
probability `min(1, p(x)/q(x))`: draw `u ~ U(0,1)` and accept while
`u < p(x)/q(x)`. On the first rejection, stop, and resample that position from
the **residual** distribution, the normalised positive part of `p - q`. Then
discard every later drafted token in that cycle.

With this rule the emitted sequence has exactly the target's distribution. If
every drafted token is accepted, one extra token may be sampled from the
target's own next distribution, which the verification pass already produced —
so a cycle emits between 1 and k+1 tokens and never zero.

At temperature 0 the same structure degenerates to: accept while the drafted
token equals the target's argmax.

**Both vocabularies must be trimmed to the real tokenizer length before any
softmax.** Draft and target output heads are often padded to different widths;
if `p` and `q` are normalised over different supports, the ratio is wrong in a
way that no test at temperature 0 will reveal.

## The index arithmetic, which is the bug you will write

`p[i]` — the target's distribution for drafted token `i` — comes from the
verification pass logits at index `len(prefix) - 1 + i`, because position `j`'s
logits predict token `j+1`. Off by one here produces a decoder that still runs,
still emits fluent text, and is silently not exact. Write the index arithmetic
out in a comment at the point of use, and check it against a tiny hand-worked
example before trusting any result.

## Families of proposer

**Separate draft model.** A small model from the same family, sharing the
tokenizer (required for the ratio test). Simple and strong per-token, but it
pays a full model forward pass per drafted token, which is the dominant cost at
batch 1.

**N-gram / prompt lookup.** No neural proposer at all: continue the current
context by matching a recent suffix against earlier text and copying what
followed. Free to run, and the acceptance rate is domain-dependent — strong when
the output repeats structure it has already produced, weak in open prose.

**Self-drafting heads.** Extra output heads on top of the target's own last
hidden state, each predicting one further position. The proposer then costs no
extra model pass: the heads run on a forward pass the target already made. Head
quality is lower than a real draft model, so acceptance falls, but the cost per
drafted token collapses.

**A lightweight head over the target's hidden states.** A small trained module
that consumes the target's own hidden state (rather than only token ids) and
autoregresses cheaply in that space. Typically a better acceptance rate than
plain extra heads at a similar cost, at the price of training it.

The two trained families need a training set generated from the target itself;
the quality of that data matters, but so does the decoding rule you compare
under — see `specdec-replication`.

## Tree-shaped candidates

Instead of one chain of k tokens, propose a **tree**: several alternatives at
each position, verified together in a single batched forward pass. Verification
uses a 4D additive attention mask where each node attends only to its ancestors,
and siblings share a position index. The accepted path is the deepest prefix of
the tree the accept rule admits.

Trees raise tokens-per-cycle substantially for one proposer pass, which is why
they usually beat chains at batch 1. Two cautions: the verification cost per
token is much higher for a large tree than for a short chain, so measure it
rather than extrapolating; and the mask must be cleared from the model before
any later plain forward pass.

A useful simplification: when the tree's candidates at a position are
deterministic top-k picks (a one-hot proposer distribution), multi-round
rejection sampling collapses to a single draw — sample one token from the
target's distribution; if it is one of the children, descend into that child;
otherwise emit the sampled token and stop. This is exact, and much simpler than
iterating the general rule.

## Stopping

Stop on **every** id the model's generation config lists as terminal, not only
the tokenizer's primary end-of-sequence token. Within a cycle, once a stop token
is emitted, discard everything drafted after it. Getting this wrong changes the
output length and quietly breaks token-identity tests.
