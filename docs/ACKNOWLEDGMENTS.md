# Acknowledgments

**Author:** Mike Singleton.

This work grew out of long conversations with a small group of independent researchers. Specific debts:

- **Nelson Spence** developed the OrdVec / RankQuant formal framework that the Ordinal Quotient Preservation tests put into practice, proposed the dimension-saturation hypothesis, and hardened this reproducibility suite (shared stats module, portable paths, runner fixes; see [`SUITE_CHANGELOG.md`](SUITE_CHANGELOG.md)). `scripts/stats.py` is his code.
- **Jason Ausborn** flagged the possessive-syntax confound, which led to the matched-pairs control. He also suggested fixing the input instead of the output, which shaped the distractor design.
- **Larry Diffey** asked why language models don't talk in idioms. That question started the idiom-generation and directional-margin experiments.
- **Andrew Edmark**, **Walker Pollitt** and **Chip Aldridge** gave early reads and pressure-tested the ideas.

Any errors are the author's.
