# BioML-Cancer-Genomics-Portfolio

A portfolio of machine learning and statistical genomics projects — each framed as a well-posed mathematical problem and solved with the appropriate estimator, inference procedure, or dynamic program.

---

## 1. Cervical Cancer Prediction Model

Supervised learning for early cancer detection and risk assessment.

### Problem Setup

Given $n$ patient feature vectors $\mathbf{x}_i \in \mathbb{R}^{p}$ with labels $y_i \in \{0,1\}$, learn a classifier minimizing regularized empirical risk $\hat{f} = \arg\min_f \frac{1}{n}\sum_i \ell(y_i, f(\mathbf{x}_i)) + \lambda\,\Omega(f)$, and select the function class and hyperparameters by cross-validated ROC analysis.

### Dimensionality Reduction & Feature Selection

**PCA** finds the orthogonal projection maximizing retained variance — the top-k eigenvectors of the sample covariance:

$$\mathbf{C} = \frac{1}{n-1}\mathbf{X}_c^{\top}\mathbf{X}_c = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{\top}, \qquad \mathbf{T} = \mathbf{X}_c \mathbf{V}_{(k)}, \qquad \max_{k}\ \frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j=1}^{p}\lambda_j}.$$

**Recursive Feature Elimination (RFE)** greedily removes the feature with the smallest model importance $|w_j|$ (or impurity-based importance) and refits, ranking features by elimination order.

### Model Zoo

- **Support Vector Machine** — the margin maximizer, solved in its dual form with kernel $K(\mathbf{x}, \mathbf{x}')$:

$$\min_{\mathbf{w},b,\boldsymbol{\xi}}\ \frac{1}{2}\lVert \mathbf{w} \rVert^2 + C\sum_{i=1}^{n}\xi_i \quad \text{s.t.}\quad y_i(\mathbf{w}^{\top}\boldsymbol{\phi}(\mathbf{x}_i) + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0,$$

$$\Longrightarrow\ \max_{\boldsymbol{\alpha}}\ \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j), \qquad 0 \le \alpha_i \le C.$$

- **Naive Bayes** — Bayes' rule under conditional independence, $\hat{y} = \arg\max_c\ P(y = c)\prod_{j=1}^{p} P(x_j \mid y = c)$.
- **Random Forest** — bootstrap-aggregated trees splitting on Gini impurity decrease, $\Delta I_G = I_G(\mathcal{S}) - \sum_{m \in \{L,R\}} \frac{|\mathcal{S}_m|}{|\mathcal{S}|} I_G(\mathcal{S}_m)$ with $I_G = 1 - \sum_c p_c^2$.
- **Gradient Boosting** — stagewise additive modeling, $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot h_m(\mathbf{x})$ with $h_m$ fit to the negative gradient $-\left[\partial \ell / \partial F\right]_{F = F_{m-1}}$.

### Hyperparameter Search & Evaluation

Grid search over $\Theta$ selects $\theta^{*} = \arg\max_{\theta \in \Theta} \frac{1}{K}\sum_{k=1}^{K} \mathrm{AUC}\big(\mathcal{F}_k^{\mathrm{val}};\, \theta\big)$ via `GridSearchCV`, where

$$\mathrm{AUC} = P\big(f(\mathbf{x}^{+}) > f(\mathbf{x}^{-})\big) = \int_{0}^{1} \mathrm{TPR}\big(\mathrm{FPR}^{-1}(u)\big)\, du$$

is estimated from the ROC curve on held-out folds.

### Technologies
Python · scikit-learn · data cleaning & feature engineering · ROC analysis · visualization

---

## 2. Genomic Splice Site Prediction (HMM)

Exon region identification through splice site detection as inference in a **Hidden Markov Model**.

### Model Definition

An HMM $\lambda = (\mathcal{Q}, \mathcal{O}, \mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$ with hidden states $\mathcal{Q}$ (pre-site / consensus / post-site), DNA observations $\mathcal{O} = \{A, C, G, T\}$, transition matrix $A_{ij} = P(q_{t+1} = j \mid q_t = i)$, emission matrix $B_j(o) = P(o_t = o \mid q_t = j)$, and initial distribution $\pi_i = P(q_1 = i)$.

**Model topology:**
```
Donor Model:    [Start] → [Pre-site] → [Consensus] → [Post-site] → [End]
Acceptor Model: [Start] → [Pre-site] → [Consensus] → [Post-site] → [End]
```

### Inference

**Decoding (Viterbi)** — the most likely state path via dynamic programming over trellis values

$$\delta_t(j) = \max_{i \in \mathcal{Q}} \left[ \delta_{t-1}(i)\, A_{ij} \right] \cdot B_j(o_t), \qquad \mathbf{q}^{*} = \arg\max_{\mathbf{q}}\ P(\mathbf{q} \mid \mathbf{o}, \lambda).$$

**Likelihood (Forward algorithm)** — $P(\mathbf{o} \mid \lambda) = \sum_j \alpha_T(j)$ with the recursion $\alpha_t(j) = \left[\sum_i \alpha_{t-1}(i)\, A_{ij}\right] B_j(o_t)$, computed in $O(T\,|\mathcal{Q}|^2)$.

**Parameter estimation** — Baum–Welch (EM): expected transition/emission counts from posterior state probabilities, iterated to a local optimum of $P(\mathbf{o} \mid \lambda)$.

### Evaluation

Site-level predictions are scored against curated annotations with a confusion matrix, reporting sensitivity $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ and specificity $\mathrm{TN}/(\mathrm{TN}+\mathrm{FP})$ for donor and acceptor models separately.

---

## 3. Microarray Data Analysis Projects

Statistical genomics workflows on Affymetrix expression arrays.

### 3.1 Preprocessing (GSE50567)

Raw probe intensities pass through the RMA-style pipeline: background correction, quantile normalization (forcing a common empirical CDF across arrays), and median-polish summarization,

$$\log_2 \mathrm{PM}_{gj} = \mu_g + a_j + \varepsilon_{gj} \quad \Longrightarrow \quad \hat{\mu}_g = \operatorname{median\text{-}polish}\big(\log_2 \mathrm{PM}_{g\cdot}\big),$$

yielding one $\log_2$ expression estimate per probeset $g$ and array $j$.

### 3.2 Differential Expression: limma

For each gene, fit a linear model on the design matrix $\mathbf{X}$,

$$\mathbb{E}[\mathbf{y}_g] = \mathbf{X}\boldsymbol{\beta}_g,$$

and replace the raw variance estimator $s_g^2$ with an **empirical-Bayes moderated** version that shrinks toward the prior estimated across all genes:

$$\tilde{s}_g^{\,2} = \frac{d_0 s_0^2 + d_g s_g^2}{d_0 + d_g}, \qquad \tilde{t}_g = \frac{\hat{\beta}_g}{\tilde{s}_g \sqrt{v_g}} \;\sim\; t_{\,d_0 + d_g}\ \text{under } H_0.$$

This moderated $\tilde{t}$-statistic is the key to stable inference with small $n$: genes borrow variance information from the whole transcriptome.

### 3.3 Enrichment Analysis

**Over-representation (GO / Reactome)** — one-sided hypergeometric test for set $\mathcal{S}$ of size $K$ among $N$ annotated genes, with $n$ significant hits and $k$ overlapping:

$$p_{\mathcal{S}} = P(X \ge k) = \sum_{i=k}^{\min(n,K)} \frac{\binom{K}{i}\binom{N-K}{n-i}}{\binom{N}{n}}.$$

**GSEA** — Kolmogorov–Smirnov-like running-sum statistic over the list ranked by a differential score (see the ES formula in my [sepsis-scrna-dynamics](https://github.com/xqiu625/sepsis-scrna-dynamics) repo), with phenotype-permutation $p$-values.

**Multiple testing** — Benjamini–Hochberg FDR at $q = 0.05$ across all tested sets.

### Projects
- **GSE19383** — differential expression + GO/Reactome over-representation analysis (R Markdown).
- **GSE50567** — full QC pipeline + limma + GO/Reactome/GSEA.

---

*All projects include comprehensive documentation and reproducible analysis workflows.*
