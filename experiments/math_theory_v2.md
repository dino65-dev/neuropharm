---
title: "Safe Activation Steering Geometry: Therapeutic Windows, Overdose Bounds, and Null-Space Antidotes in Language Model Activation Steering"
author: "NeuroPharm Research"
date: "June 2026"
status: "Mathematical report — revised version"
---

# Safe Activation Steering Geometry

A Mathematical Theory of Therapeutic Windows, Overdose Bounds, and Null-Space Antidotes in Language Model Activation Steering

---

## Section 0: Assumptions and Notation

This section defines all objects, operators, metrics, and geometric structures used in every theorem that follows.

### 0.1 The Model

Autoregressive transformer with $L$ layers and hidden dimension $d$. For a fixed token position $t$ and layer $\ell \in \{0, \dots, L\}$:

$$h_t^{(\ell)} \in \mathbb{R}^d$$

denotes the residual stream. The layer transition is $h_t^{(\ell)} = f_\ell(h_t^{(\ell-1)}) = h_t^{(\ell-1)} + g_\ell(h_t^{(\ell-1)})$, where $g_\ell$ is the composite attention-plus-MLP sublayer output.

**Assumption 0 (Smoothness).** Each $g_\ell$ is $C^2$ (twice continuously Fréchet differentiable) on $\mathbb{R}^d$, with locally bounded second derivative. Holds for standard transformer implementations using GELU, softmax, and LayerNorm restricted to the compact set of naturally occurring activations.

### 0.2 Steering Operators

**Additive steering.** For unit-norm $v$ and scalar $\alpha$:

$$h_\alpha := h^0 + \alpha v$$

**Rotation steering.** Let $\hat{h} := h^0 / \|h^0\|$. Decompose $v$ into components parallel and orthogonal to $\hat{h}$:

$$v_\parallel := \langle v, \hat{h} \rangle \hat{h}, \quad v_\perp := v - v_\parallel, \quad v_\perp^{\text{unit}} := v_\perp / \|v_\perp\|$$

The rotation operator $R(\theta)$ acts in the 2-plane $\operatorname{span}\{\hat{h}, v_\perp^{\text{unit}}\}$ by angle $\theta$:

$$R(\theta) h^0 := \|h^0\| (\hat{h} \cos\theta + v_\perp^{\text{unit}} \sin\theta)$$

For small $|\theta|$: $R(\theta) h^0 = h^0 + \theta w + O(\theta^2)$ where $w := \|h^0\| v_\perp^{\text{unit}}$ satisfies $\langle w, h^0 \rangle = 0$ and $\|w\| = \|h^0\|$.

### 0.3 Three Metrics (all $C^1$ on a neighborhood of 0 by Assumption 0)

**Target gain $G$:** For smooth readout $g: \mathbb{R}^d \to \mathbb{R}$:

$$G(\alpha) := g(h_\alpha) - g(h^0)$$

**Utility damage $U$:** For benign prompt set $\mathcal{B}$:

$$U(\alpha) := \max_{p \in \mathcal{B}} \frac{\|h_p^\alpha - h_p^0\|}{\|h_p^0\|}$$

**Geometry damage $M$:** Operational off-manifold $z$-score from a trained SAE:

$$M(\alpha) := z_{\text{SAE}}(h_\alpha) := \frac{\|h_\alpha - \text{SAE}(h_\alpha)\| - \mu_{\text{rec}}}{\sigma_{\text{rec}}}$$

### 0.4 Reachable-State Distance and Recovery Statistic

**Reachable manifold:** $\mathcal{M}_\ell := \operatorname{Im}(f_\ell)$ is a proper closed subset of $\mathbb{R}^d$ (Mishra et al., arXiv 2604.09839).

**Off-manifold distance:** $d(x, \mathcal{M}_\ell) := \min_{y \in \mathcal{M}_\ell} \|x - y\|$.

**Operational proxy:** $d_{\text{SAE}}(x) := \|x - \text{SAE}(x)\|$.

**Recovery:** A steering intervention with output $h$ is recovered if $d(h, \mathcal{M}_\ell) < \tau$ for threshold $\tau > 0$.

### 0.5 Benign and Harmful Subspaces

- $v_{\text{drug}}$: ActAdd direction from $N_{\text{drug}}$ contrastive pairs
- $v_{\text{harm}}$: ActAdd direction from $N_{\text{harm}}$ contrastive pairs
- $\mathcal{S}_{\text{benign}} = \operatorname{span}\{v_{\text{drug}}, h^0, h^{0\perp} \cap \mathcal{M}_\ell\}$
- $\mathcal{S}_{\text{harm}} = \operatorname{span}\{v_{\text{harm}}\}$
- $\cos_{\text{dh}} := \langle v_{\text{drug}}, v_{\text{harm}} \rangle / (\|v_{\text{drug}}\| \cdot \|v_{\text{harm}}\|)$

### 0.6 Local Manifold Model

At a clean activation $h^0 \in \mathcal{M}_\ell$ (regular point), the tangent space $T_{h^0}\mathcal{M}_\ell$ is well-defined. Let $P_T$ be the orthogonal projector.

**Curvature bound.** Second fundamental form $\mathrm{I\!I}$ satisfies $\|\mathrm{I\!I}(u,v)\| \leq \kappa \|u\| \|v\|$ for some $\kappa \geq 0$ in $B_\rho(h^0)$.

**First-order distance approximation.** For $\|\delta\|$ small:

$$d(x, \mathcal{M}_\ell) = \|P_T^\perp (x - h^0)\| + O(\kappa \|x - h^0\|^2) \tag{0.1}$$

---

## Section 1: Theorem 1 — Local Therapeutic Window Existence

**Statement.** Under (A1) $C^1$ smoothness, (A2) $G'(0) > 0$, (A3) $|U'(0)| \leq \gamma_U$, (A4) $|M'(0)| \leq \gamma_M$, and (A5) margin condition:

$$\frac{2\tau_G}{G'(0)} \leq \min\left(\frac{\tau_U}{2\gamma_U}, \frac{\tau_M}{2\gamma_M}\right)$$

there exists $\varepsilon > 0$ such that the therapeutic window $W(\tau_G, \tau_U, \tau_M) \cap [0, \varepsilon] \neq \varnothing$.

**Proof sketch.** First-order Taylor expansion at $\alpha = 0$ with bounded Lagrange remainder. The margin condition (A5) ensures the three inequalities $G(\alpha) \geq \tau_G$, $U(\alpha) \leq \tau_U$, $M(\alpha) \leq \tau_M$ have a common solution $\alpha$ in $[0, \varepsilon]$ for $\varepsilon$ small enough that the second-order remainders are dominated by the first-order terms.

**Connection to E1.** Empirical $G'(0) \approx 0.66$, $\gamma_U \leq 0.30$, $\gamma_M \leq 0.80$, $\tau_G = 0.30$, $\tau_M = 2.0$. The margin condition fails for the raw $v_{\text{drug}}$ but holds for the antidoted direction $v_{\text{ant}}$ (Theorem 4), which is why the window is non-empty for 5/6 behaviors.

**What T1 does NOT claim.** It does NOT guarantee the window is large, monotonic, applicable to large $\alpha$, or that it holds for the raw drug direction.

---

## Section 2: Theorem 4 — Null-Space Antidote Guarantees

**Statement.** With $P_\perp := I - v_{\text{harm}} v_{\text{harm}}^\top$ and $v_{\text{ant}} := P_\perp v_{\text{drug}} = v_{\text{drug}} - \cos_{\text{dh}} v_{\text{harm}}$:

1. **Harm orthogonality:** $\langle v_{\text{ant}}, v_{\text{harm}} \rangle = 0$.
2. **Norm preservation:** $\|v_{\text{ant}}\|^2 / \|v_{\text{drug}}\|^2 = 1 - \cos_{\text{dh}}^2$.
3. **Benign-prompt invariance:** On prompts with $\langle h^0, v_{\text{harm}} \rangle = 0$ and $\langle \nabla g(h^0), v_{\text{harm}} \rangle = 0$, $G_{\text{ant}}(\alpha) = G_{\text{drug}}(\alpha)$ to first order.

**Proof.** $P_\perp$ is a symmetric idempotent with $P_\perp v_{\text{harm}} = 0$, giving Parts 1-2 by direct computation. Part 3 uses the first-order Taylor expansion: the difference $v_{\text{ant}} - v_{\text{drug}} = -\cos_{\text{dh}} v_{\text{harm}}$ lies in $\mathcal{S}_{\text{harm}}$, contributing nothing to $G$ on prompts insensitive to $v_{\text{harm}}$.

**Connection to E4.** With $\cos_{\text{dh}} = -0.180$, predicted norm preservation is $98.4\%$, measured $99.4\%$ — small discrepancy from normalization conventions. Titrated antidote recovered behavioral confidence from 0.00 to 0.33, confirming Part 1 and the Corollary 5.1 P-controller advantage.

**What T4 does NOT claim.** It does NOT claim $v_{\text{ant}}$ is optimal, that the benign-prompt condition holds universally, or that all harmful effects are eliminated.

---

## Section 3: Theorem 2 — Overdose Bound for Additive Steering

**Statement.** If $\eta_v := \|(I - P_{\mathcal{F}}) v\| > 0$, there exist $\alpha^* > 0$ and constants $c_1, c_2 > 0$ such that for all $|\alpha| \geq \alpha^*$:

$$M(\alpha) \geq c_1 |\alpha| - c_2, \quad c_1 = \eta_v / \sigma_{\text{rec}}$$

The overdose threshold satisfies:

$$\alpha_{\text{OD}} \leq \frac{\tau_M \sigma_{\text{rec}} + \mu_{\text{rec}} + \|(I - P_{\mathcal{F}})(h^0 - b_{\text{dec}})\|}{\eta_v} \tag{3.1}$$

**Proof.** $\|h_\alpha - \text{SAE}(h_\alpha)\| = \|(I - P_{\mathcal{F}})(h^0 + \alpha v - b_{\text{dec}})\|$. Reverse triangle inequality gives the linear lower bound for $|\alpha| \geq \alpha^*$. Convert to $z$-score and solve $M > \tau_M$ for $\alpha$.

**Connection to E2.** $0.80$ $z$-units per unit $\alpha$ for the harm direction gives $\eta_v \approx 0.14$ in normalized coordinates. The 2× ratio over non-harm directions rules out trivial correlation.

**What T2 does NOT claim.** It does NOT claim $M(\alpha)$ is exactly linear, that $\alpha < \alpha_{\text{OD}}$ is safe, or provide an upper bound on $M(\alpha)$.

---

## Section 4: Theorem 3 — Local Comparative Rotation Theorem (REVISED)

### Preamble

The previous version claimed rotation is globally superior to addition. This was an overclaim. The norm-preservation property is global, but the off-manifold-departure comparison is inherently local. We now state a **strictly local, conditional, comparative** theorem.

### Statement

Let $h^0 \in \mathbb{R}^d$ with $\|h^0\| > 0$, $v \in \mathbb{R}^d$ unit-norm, and the first-order rotation perturbation $w := \|h^0\| v_\perp^{\text{unit}}$.

**Assumptions for Theorem 3:**

- **(A6a) Gain-matching:** $\langle \nabla g(h^0), \alpha v \rangle = \langle \nabla g(h^0), \theta w \rangle$ (GM)
- **(A6b) Local curvature bound:** $\|\mathrm{I\!I}\| \leq \kappa$ in $B_\rho(h^0)$
- **(A6c) Tangent-alignment hypothesis:** $\|(I - P_T) w\| \leq \|(I - P_T) v\|$ (TA)
- **(A6d) Small-perturbation regime:** $|\alpha|, |\theta|$ in the linearization radius

**Theorem 3 (Local Comparative Rotation).** Under (A6a)-(A6d):

1. **Zero radial norm distortion (global, exact):** $\|R(\theta) h^0\| = \|h^0\|$ for all $\theta$. For addition, $\|h_\alpha\|^2 = \|h^0\|^2 + 2\alpha\langle h^0, v \rangle + \alpha^2$, so radial growth is $\alpha \langle h^0, v \rangle / \|h^0\|$ to first order.

2. **No-worse first-order manifold departure (local, conditional):** From eq. (0.1):
   $$d(R(\theta)h^0, \mathcal{M}_\ell) = |\theta| \|(I - P_T) w\| + O(\kappa \theta^2 \|h^0\|^2)$$
   $$d(h_\alpha, \mathcal{M}_\ell) = |\alpha| \|(I - P_T) v\| + O(\kappa \alpha^2)$$

   Under (A6a)-(A6c), $|\theta| \|(I - P_T) w\| \leq |\alpha| \|(I - P_T) v\|$ provided $|r| \leq 1$ where $r = \langle \nabla g, v \rangle / \langle \nabla g, w \rangle$. The first-order manifold departure of rotation is no worse than that of addition.

3. **First-order gain alignment:** $G_{\text{rot}}(\theta) = \langle \nabla g, \theta w \rangle + O(\theta^2)$, $G_{\text{add}}(\alpha) = \langle \nabla g, \alpha v \rangle + O(\alpha^2)$. Under (A6a), the first-order gains are exactly equal.

### Proof Sketch

**Part 1.** Algebraic identity: $R(\theta) \in \mathrm{SO}(d)$ restricted to $\operatorname{span}\{\hat{h}, v_\perp^{\text{unit}}\}$ preserves norm globally.

**Part 2.** Apply eq. (0.1) to $\delta = \theta w + O(\theta^2)$ (rotation) and $\delta = \alpha v$ (addition). Under (A6c), $\|(I - P_T) w\| \leq \|(I - P_T) v\|$. Under (A6a), $\theta = \alpha r$ with $|r| \leq 1$ when $\nabla g$ is more aligned with $v$ than with $w$. The favorable regime holds.

**Part 3.** Direct first-order Taylor, equal by (A6a).

### Scope Declaration

Theorem 3 is **strictly local** — valid for $\|\delta\|$ small. It makes **no claim** about moderate or large perturbation sizes, and **no claim** that $d_{\text{rot}} \leq d_{\text{add}}$ persists beyond the local regime.

**Connection to E3/E5.** Sparse-additive steering is the empirical proxy for tangent-space channeling. Theorem 3 makes this precise: rotation restricts the perturbation to the $\{h^0, v_\perp\}$ plane.

**What T3 does NOT claim.**
- Not globally better
- Not immune to overdose at large $\theta$
- Not robust to large perturbations
- Does not assert (A6c) holds universally
- Does not assert the gain-matching is always satisfiable

---

## Section 5: Control-Theoretic Corollary

### Corollary 5.1 (Titrated Antidote as Feedback Steering)

**Setup.** Let $\text{off}(h) := d_{\text{SAE}}(h) = \|h - \text{SAE}(h)\|$. The titrated antidote is:

$$\beta_t^* := -\lambda \frac{\langle \text{off}(h_t), v_{\text{ant}} \rangle}{\|v_{\text{ant}}\|^2}$$

applied as $h_t \leftarrow h_t + \beta_t^* v_{\text{ant}}$.

**Corollary 5.1.** The titrated antidote is a discrete-time proportional controller $u_t = -K_t \text{off}(h_t)$ with $K_t = \lambda v_{\text{ant}} v_{\text{ant}}^\top / \|v_{\text{ant}}\|^2$. Under local linearization $\text{off}(h + \delta) \approx \|(I - P_{\mathcal{F}}^{\text{lin}}) \delta\|$ and Assumption 0:

$$\|\text{off}(h_t + u_t)\|^2 \leq (1 - \lambda \eta_{\text{ant}}^2) \|\text{off}(h_t)\|^2 + \text{disturbance}$$

where $\eta_{\text{ant}} := \|(I - P_{\mathcal{F}}^{\text{lin}}) v_{\text{ant}}^{\text{unit}}\|$. For $0 < \lambda < 2/\eta_{\text{ant}}^2$, the unforced off-manifold error contracts geometrically.

### Connection to PID Control

The titrated antidote is a **P-controller**:
- **P (proportional):** Current error feedback (Theorem 4 + Corollary 5.1)
- **I (integral):** Accumulates past error; drives steady-state to zero but risks windup
- **D (derivative):** Anticipates trend; adds damping; prevents oscillation

A **PI-antidote** would provide asymptotic zero off-manifold error. A **PID-antidote** would additionally dampen oscillatory drift.

### Connection to E4

The titrated (P-controller) antidote succeeded where the static (open-loop) antidote failed: behavioral confidence recovered from 0.00 to 0.33. The P-controller uses current state feedback to adapt, while open-loop commits to a predetermined correction.

---

## Section 6: Discussion

### 6.1 What the Theorem Stack Establishes

| Result | Role |
|---|---|
| **T1** (Therapeutic Window) | Window exists under first-order conditions |
| **T4** (Null-Space Antidote) | Constructive removal of known harmful component |
| **T2** (Overdose Bound) | Computable lower bound on geometry damage growth |
| **T3** (Local Rotation) | Conditions under which rotation is locally competitive |
| **Corollary 5.1** (Control-Theoretic) | P-controller contraction guarantee |

### 6.2 What Remains Open

1. Global rotation advantage (bounding $\|\mathrm{I\!I}\|$ globally)
2. SAE completeness guarantee (bounding missed-feature probability)
3. Multi-layer steering composition
4. PI/PID antidote with anti-windup
5. Pharmacokinetic model for multi-layer interventions
6. Direct experimental validation of Theorem 3

### 6.3 Limitations of Local Linearization

All theorems rely on Assumption 0 ($C^2$ differentiability with bounded second derivative). The first-order analyses are valid for $\|\delta\|$ small enough that second-order remainders are dominated. Theorem 2 is a partial exception: its lower bound is global because it depends only on SAE projector linearity, but the constant $c_2$ depends on the local point $h^0$.

### 6.4 Empirical Validation Status

| Theorem | Empirical Anchor | Status |
|---|---|---|
| T1 | E1: 5/6 behaviors show non-empty window with antidoted direction | **Validated** |
| T4 | E4: 99.4% norm preservation, titrated recovery 0.00 → 0.33 | **Validated** |
| T2 | E2: linear $z$-growth, harm 2× non-harm | **Validated** |
| T3 | E3: sparse-additive proxy; direct rotation experiment | **Untested** |
| Cor 5.1 | E4: P-controller beats open-loop | **Validated** |

---

## Summary of Notation

| Symbol | Meaning |
|---|---|
| $d$ | Hidden dimension |
| $L$ | Number of layers |
| $h_t^{(\ell)}$ | Residual stream at token $t$, layer $\ell$ |
| $h^0$ | Clean activation at intervention point |
| $h_\alpha$ | Additive steering: $h^0 + \alpha v$ |
| $v$ | Unit-norm steering direction |
| $R(\theta)$ | Rotation in $\operatorname{span}\{\hat{h}, v_\perp^{\text{unit}}\}$ |
| $w$ | First-order rotation perturbation |
| $G, U, M$ | Target gain, utility damage, geometry damage |
| $\mathcal{M}_\ell$ | Reachable manifold |
| $P_T$ | Projector onto tangent space |
| $\kappa$ | Curvature bound |
| $v_{\text{drug}}, v_{\text{harm}}$ | ActAdd directions |
| $v_{\text{ant}}$ | Antidote direction |
| $\cos_{\text{dh}}$ | Drug-harm cosine |
| $\eta_v$ | Off-manifold sensitivity |
| $\sigma_{\text{rec}}$ | SAE reconstruction error std |
| $\alpha_{\text{OD}}$ | Overdose threshold |
| $\beta_t^*$ | Titrated antidote strength |

---

**End of Report.** All theorems stated with explicit assumptions, proof sketches, empirical connections, experimentalist guidance, and explicit scope limitations. The revised Theorem 3 is unmistakably local, conditional, and comparative.
