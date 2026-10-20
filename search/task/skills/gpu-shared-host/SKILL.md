---
name: gpu-shared-host
description: >-
  Working on a GPU machine other people are also using, where cards are not
  reserved and can fill up under you at any moment. Covers which device index a
  pinned process actually sees, why nvidia-smi disagrees with your process,
  surviving another tenant filling your card, keeping memory reclaimable across
  model loads, and making failures diagnosable. Read this before writing code
  that selects a GPU, loads a large model, or times anything.
---

# Sharing a GPU box

Most GPU code is written as if the machine were yours. On a shared host that
assumption produces failures that look like bugs in your model code and are not.

## The device your process sees is not the device nvidia-smi reports

When a scheduler, container or wrapper pins your process to a card, it does so
with `CUDA_VISIBLE_DEVICES`. Inside that process the allocated card is
**device 0**, whatever its physical index. A process given physical card 2 sees
exactly one device and calls it `cuda:0`.

`nvidia-smi` does **not** respect `CUDA_VISIBLE_DEVICES`. It reports every
physical card on the machine. So this apparently sensible idea is broken:

> query nvidia-smi for the card with the most free memory, then use that index

It returns a physical index your process cannot address. Asking for `cuda:2`
when only `cuda:0` exists raises an invalid-device-ordinal error, usually deep
inside a model load, far from the line that chose it.

Rules:

- **Pick among the visible devices by free memory**, never by a physical index:
  `max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])`.
  Pinned, there is exactly one device and you get it; unpinned, you avoid a card
  another tenant has filled.
- Do not hardcode `cuda:0` either. It is right only when something pinned you to
  one card, and the same code often runs in both situations - a scoring program
  called once by a pinned worker and once by an unpinned verifier. Hardcoding it
  sent one such program onto a stranger's full card, where it OOMed and then
  waited for memory that was never going to free, with an empty card beside it.
- To ask how much memory you can actually use, use the framework's own view
  (`torch.cuda.mem_get_info`), not a shell out to `nvidia-smi`.
- Choose a card with `nvidia-smi` only in an *unpinned* launcher, and pass the
  choice down through `CUDA_VISIBLE_DEVICES` rather than as an index.

## A free card is not a reserved card

Schedulers that place work on GPUs usually *place* rather than *reserve*: they
pick the emptiest card at start time and grant nothing. Another tenant may fill
it a minute later, and sharing is typically legal by design.

So a memory check at startup does not protect a load that happens later. Before
allocating something large:

- Check free memory at the moment you need it, not once at the top.
- Wait and retry, within whatever time budget you have, rather than dying on the
  first allocation failure. A transient squeeze costs minutes; an unhandled one
  can throw away hours of completed work.
- Report what you saw -- "needed 30 GB, 7 GB free after waiting 10 min" is
  actionable; `CUDA out of memory` alone is not.

## Freeing a model is harder than allocating one

In a long-lived process that loads several large models in turn, `del model`
plus an empty-cache call often does not return memory to the allocator: a
reference survives somewhere, or the arena is too fragmented for the next
contiguous block. The second or third load then fails on a machine with plenty
of apparent room.

The reliable pattern is **one heavy model per process**. Drive a sequence of
them from a parent that spawns a subprocess per item; the operating system
reclaims everything on exit. Where that is impractical, an expandable-segments
allocator setting reduces fragmentation, but it is a mitigation rather than a
fix.

The same trick solves "the reference and the candidate are each too big to hold
together": run them as two phases in two processes, passing results through a
file.

## Timing on a machine you do not control

Absolute throughput on a shared box is a property of the box that hour, not of
your code. Two defences:

- **Interleave what you compare.** Time A and B alternately, close together,
  rather than all of A then all of B, so drift hits both sides.
- **Report a load-independent statistic** beside the timing. Something that is
  deterministic for fixed inputs should reproduce exactly; if it does and the
  wall-clock moved, the machine moved.

Record which machine state a number came from. A ratio with no note about what
else was running is not reproducible, and is usually not comparable to any other
number you have.

## Make the failure legible

Shared-host failures arrive at the worst moment and often only once. When you
shell out to a subprocess, capture stderr and put it in the error you raise; a
bare non-zero exit code tells whoever reads the log nothing, and the same
failure then repeats with nothing new to learn from.
