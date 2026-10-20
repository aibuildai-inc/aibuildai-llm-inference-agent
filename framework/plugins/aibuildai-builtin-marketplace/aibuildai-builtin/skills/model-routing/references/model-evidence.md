# Past model routing facts

These facts come from old AIBuildAI runs. Use them when the task and product shape still match. They do not prove that a newer model is weak.

- `detecting-insults-in-social-commentary`: ten cheaper or mixed choices gave an invalid result. The observed choice that worked used `claude-opus-4-8` for all routed roles.
- `denoising-dirty-documents`: `claude-opus-4-8` for all roles reached the top result band in three runs. Lowering models for individual designs cut cost a little and lowered the result band.
- `leaf-classification`: `claude-sonnet-4-6` for most roles with `claude-opus-4-8` for Coder and Aggregator gave a valid result above the middle score at lower cost than Opus for all roles. The result varied across runs.
- `nomad2018-predict-transparent-conductors`: Sonnet for Designer and Coder with Haiku for the other roles reached the top result band once at lower cost than Sonnet for all roles. Several runs failed in task splitting with more than one model, so that failure did not point to one model.

Model planning and code writing often need more reasoning than simple checks. This is a starting point, not a fixed role rule. Data size, input type, output rules, tool use, and the cost of an invalid result can change the choice.
