# Provenance: recovered terminal captures

Two Qwen2-7B runs of the OrdVec layer-role test, captured from the EC2 terminal (June 2026). They are separate runs; don't merge them.

- `layer_role_qwen_run1_test1_terminal.txt`: `hsg_ordvec_layer_role_test_1.py` (earlier version, not in this repo). Ends in a print-formatting error after the table, so no JSON was saved.
- `layer_role_qwen_run2_test2_terminal.txt`: `hsg_ordvec_layer_role_test_2.py`. This run produced `results/hsg_ordvec_layer_role_results.json` and the layer table in paper v0.93 §8.3. The Gen AUC values match the JSON exactly.

What the pair shows: Q_full and Q_ordvec are identical in both runs at all 27 layers, so they're deterministic and reproducible. Gen AUC moves by 0.25 per layer on average between runs (layer 9: 0.806 vs 0.200), because it rests on 12 freshly sampled continuations scored by keyword. See docs/RESULTS.md §4.

These captures are not the Qwen semiotic-accuracy data (AUC 0.778); that run was recovered separately as `results/semiotic_accuracy_results/semiotic_accuracy_generations_qwen.csv`.
