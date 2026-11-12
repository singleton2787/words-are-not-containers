# Methods

Formal definitions of every quantity reported in this repository. Each one matches the script named under it.

## Notation

| Symbol | Meaning |
|:--|:--|
| $h_\ell(s) \in \mathbb{R}^{D}$ | hidden state of the **final token** of sentence $s$ at layer $\ell$ |
| $L$ | number of transformer layers in the model |
| $\mathcal{C} = \{\text{hedonic}, \text{companion}, \text{covenantal}, \text{moral}\}$ | affordance classes |
| $O_c$ | objects in class $c$ (5 per class, 20 total) |
| $\mathcal{V}$ | predicates: *love, need, fear, eat, own, remember, care, hate*, plus the neutral *see* |
| $\lVert\cdot\rVert$ | Euclidean ($L_2$) norm |

## 1. Object representations

Each predicate–object pair is placed in four templates, $T = \{$"I V O.", "We V O.", "She V O.", "They V O."$\}$. The object's representation under predicate $V$ is the template average at layer $\ell = L - 2$:

$$
v_V(o) \;=\; \frac{1}{|T|} \sum_{t \in T} h_{L-2}\big(t(V, o)\big)
$$

That's block 26 of 28 for Qwen2-7B and Gemma-7B, and block 30 of 32 for Mistral-7B.

<sub>Script: `scripts/hsg_fingerprint_{qwen,mistral,gemma}.py`</sub>

## 2. Class structure under a predicate

Mean distance between objects drawn from classes $c_i$ and $c_j$, taken over distinct object pairs:

$$
D_V(c_i, c_j) \;=\; \operatorname*{mean}_{\substack{o \in O_{c_i},\; o' \in O_{c_j} \\ o \neq o'}} \big\lVert v_V(o) - v_V(o') \big\rVert
$$

**Clustering ratio.** Mean between-class distance over mean within-class distance:

$$
R_V \;=\; \frac{\operatorname{mean}\{\, \lVert v_V(o) - v_V(o') \rVert : \operatorname{class}(o) \neq \operatorname{class}(o') \,\}}{\operatorname{mean}\{\, \lVert v_V(o) - v_V(o') \rVert : \operatorname{class}(o) = \operatorname{class}(o') \,\}}
$$

$R_V > 1$ means same-class objects sit closer together. Significance comes from a label-permutation test with 1,000 random reassignments of objects to classes, so the smallest reportable p is about 0.001. The effect size is Cohen's $d$ between the two distance sets.

<sub>Script: `permutation_test()` in `scripts/hsg_fingerprint_*.py`</sub>

## 3. Residual fingerprint

Objects cluster under any predicate, *see* included, so clustering alone says nothing about the predicate. The fingerprint subtracts the neutral predicate and then removes the overall expansion or compression, leaving only how $V$ *reshapes* object space:

$$
F_V(c_i, c_j) \;=\; \big[D_V(c_i, c_j) - D_{\text{see}}(c_i, c_j)\big] \;-\; \operatorname*{mean}_{(a,b)} \big[D_V(c_a, c_b) - D_{\text{see}}(c_a, c_b)\big]
$$

$F_V < 0$ means the predicate **pulls** that class pair together; $F_V > 0$ means it **pushes** the pair apart.

<sub>Script: `scripts/hsg_fingerprint_*.py` (stored as `fingerprints` in the result JSON)</sub>

### Predicate-specific residual (cross-model comparison)

To compare models, the shared response of all predicates is also removed, so what's left is specific to $V$:

$$
\tilde F_V(c_i, c_j) \;=\; F_V(c_i, c_j) \;-\; \frac{1}{|\mathcal{V}'|} \sum_{U \in \mathcal{V}'} F_U(c_i, c_j), \qquad \mathcal{V}' = \mathcal{V} \setminus \{\text{see}\}
$$

The cross-model claim that *fear*'s strongest pull (companion/hedonic) is the same in Qwen, Mistral and Gemma is made on $\tilde F$. On $F$ alone, Mistral and Gemma agree on *fear* and Qwen doesn't; see [RESULTS.md §2](RESULTS.md).

<sub>Script: `scripts/hsg_see_residual.py`</sub>

## 4. Geometry margin

For the prompt "I love $o$ because", take the final-token state at layer $\ell = \operatorname{round}(0.35\,L)$, which is layer 10 for Qwen and Gemma and 11 for Mistral:

$$
x_o = h_{\ell}(\text{"I love } o \text{ because"})
$$

Class centroids are computed **leaving the target object out**, so an object never votes for its own class:

$$
\mu_c^{(-o)} \;=\; \operatorname*{mean}_{o' \in O_c,\; o' \neq o} x_{o'}
$$

The margin is how much nearer the expected class $c^\*$ is than the nearest wrong class:

$$
m(o) \;=\; \min_{c \neq c^\*} \big\lVert x_o - \mu_c^{(-o)} \big\rVert \;-\; \big\lVert x_o - \mu_{c^\*}^{(-o)} \big\rVert
$$

$m(o) > 0$ means the prompt already sits in the right relational region **before any token is generated**.

<sub>Script: `geometry_margin()` and `leave_one_out_geometry_margins()` in `scripts/semiotic_accuracy_prediction.py`</sub>

## 5. Does the margin predict behavior?

For each prompt the model generates 12 continuations. Each continuation gets a label $y \in \{1 \text{ (relation kept)},\, 0 \text{ (relation lost)}\}$; ambiguous continuations are excluded. Labels currently come from a keyword classifier, with blind human annotation pending.

$$
\text{AUC} \;=\; \Pr\big(m_i > m_j \;\big|\; y_i = 1,\; y_j = 0\big)
$$

That's the chance that a randomly chosen kept continuation came from a higher-margin prompt than a randomly chosen lost one; 0.5 is chance. Also reported: a one-sided Mann–Whitney test of $m$ between the two groups, Spearman $\rho$, and the slope $\beta_1$ of the logistic model $\Pr(y = 1) = \sigma(\beta_0 + \beta_1 m)$.

$\beta_1$ depends on the scale of $m$, which runs about 10× larger in Qwen, so compare models by AUC, not $\beta_1$.

<sub>Script: `summarize_prediction()` in `scripts/semiotic_accuracy_prediction.py`</sub>

## 6. Idiom vs. literal separation

Each phrase (e.g. "break the ice") appears in contexts that force an idiomatic or a literal reading. Its representation is taken inside the context, and distances are cosine:

$$
d_{\cos}(a, b) \;=\; 1 - \frac{a \cdot b}{\lVert a \rVert \lVert b \rVert}
$$

$$
\Delta \;=\; \operatorname{mean}\, d_{\cos}(\text{idiomatic}, \text{literal}) \;-\; \operatorname{mean}\, d_{\cos}(\text{same reading})
$$

$\Delta > 0$ means identical words sit in different places depending on what they mean. Significance comes from a permutation test that shuffles the within/between labels 5,000 times, so the smallest reportable p is about 0.0002.

<sub>Script: `scripts/hsg_idiom_vs_literal.py`</sub>

## 7. Ordinal Quotient Preservation (OrdVec)

Keep only the **ranks** of the activations: the bitmap marks the $K$ largest coordinates of a hidden state,

$$
B_K(h) \;=\; \mathbb{1}\big[\, i \in \operatorname{top}_K(h) \,\big] \in \{0,1\}^{D}, \qquad \operatorname{overlap}(q, d) = \lvert B_K(q) \cap B_K(d) \rvert
$$

Under a uniform null, the overlap of two random $K$-subsets of $D$ coordinates is hypergeometric, $\operatorname{Hypergeom}(D, K, K)$. The test asks whether class labels assigned from bitmap overlap alone agree with those from the full vectors. If they do, the relational signal lives in the ordering of activations rather than their magnitudes. The formal framework, including the hypergeometric null and quotient compatibility, is Nelson Spence's OrdVec/RankQuant work.

<sub>Scripts: `scripts/hsg_ordvec_fiber_test_v2.py`, `hsg_ordvec_ablations.py`, `hsg_ordvec_layer_role_test_2.py`</sub>

## 8. Corpus-linguistic check (no model)

Wikipedia windows of ±10 tokens around *love*, *care* and *like* are compared with category baseline fields by Jensen–Shannon divergence:

$$
\operatorname{JS}(P \,\Vert\, Q) = \tfrac12 \operatorname{KL}(P \,\Vert\, M) + \tfrac12 \operatorname{KL}(Q \,\Vert\, M), \qquad M = \tfrac12 (P + Q)
$$

$\delta$ = mean relation-window JS minus mean baseline-category JS, tested against matched control words with 5,000 permutations.

<sub>Scripts: `scripts/wiki_affordance_field_test.py`, `scripts/wiki_relation_vs_control_ablation.py`</sub>
