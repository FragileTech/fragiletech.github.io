## 12. Belief Dynamics: Prediction, Update, Projection

:::{admonition} Researcher Bridge: Bayes Filter with Safety Projection
:class: info
:name: rb-bayes-filter
The predict-update loop is standard HMM/POMDP filtering. The extra step is projection by the Sieve, which removes or downweights unsafe belief mass. Think "Bayes filter plus constraints."
:::

Sections 2–9 describe geometry, metrics, and effective macro dynamics. What they do *not* yet encode is the irreversibility of online learning: boundary observations and constraint enforcement are not invertible operations. This section states the belief-evolution template directly as **filtering + projection** on the discrete macro register.

**Relation to prior work.** The predict–update recursion below is standard Bayesian filtering for discrete latent states (HMM/POMDP belief updates) {cite}`rabiner1989tutorial,kaelbling1998planning`. The additional ingredient emphasized here is the explicit **projection/reweighting layer** induced by safety and consistency checks (Section 3): belief updates are not just “Bayes + dynamics”, but “Bayes + dynamics + constraints”.

(sec-why-purely-closed-simulators-are-insufficient)=
### 12.1 Why Purely Closed Simulators Are Insufficient

A purely closed internal simulator can roll forward hypotheses, but it cannot *incorporate new boundary information* without a non-invertible update. Two irreversibilities are unavoidable:
1. **Assimilation:** boundary observations $x_{t+1}$ update the macro belief (Bayesian correction).
2. **Constraint enforcement:** the Sieve applies online projections/reweightings that remove unsafe/inconsistent mass (Gate Nodes / Barriers).

Both operations are information projections: they reduce uncertainty and/or discard parts of state-space mass in a way that cannot be undone from the post-update state alone.

(sec-filtering-template-on-the-discrete-macro-register)=
### 12.2 Filtering Template on the Discrete Macro Register

Let $p_t\in\Delta^{|\mathcal{K}|-1}$ be the macro belief over $K_t$.

**Prediction (model step).** Given the learned macro kernel $\bar{P}(k'\mid k,a_t)$ (Section 2.8), define the one-step predicted belief

$$
\tilde p_{t+1}(k') := \sum_{k\in\mathcal{K}} p_t(k)\,\bar{P}(k'\mid k,a_t).
$$
**Update (observation step).** Given an emission/likelihood model $L_{t+1}(k'):=p(x_{t+1}\mid k')$ (or any calibrated score proportional to likelihood), the posterior belief is

$$
p_{t+1}(k')
:=
\frac{L_{t+1}(k')\,\tilde p_{t+1}(k')}{\sum_{j\in\mathcal{K}} L_{t+1}(j)\,\tilde p_{t+1}(j)}.
$$
This is the standard Bayesian filtering recursion for a discrete latent state (HMM/POMDP belief update) {cite}`rabiner1989tutorial,kaelbling1998planning`. Units: probabilities are dimensionless; log-likelihoods and entropies are measured in nats.

(sec-sieve-events-as-projections-reweightings)=
### 12.3 Sieve Events as Projections / Reweightings

When a check triggers, we apply a *projection-like* operator to the belief state. Two common forms are:

- **Hard projection (mask + renormalize):**

  $$
  p'_{t}(k)\propto p_t(k)\cdot \mathbb{I}\!\left[\text{feasible}(k)\right].
  $$
  Example: feasibility defined by a cost budget $V(k)\le V_{\max}$ (CostBoundCheck).

- **Soft reweighting (exponential tilt):**

  $$
  p'_t(k)\propto p_t(k)\,\exp\!\left(-\lambda\cdot \text{penalty}(k)\right),
  $$
  which implements a differentiable “push away” from unstable regions.

These are classical constrained-inference moves (mirror descent / I-projection style), and they are the belief-space counterpart of the Gate Nodes.

(sec-over-under-coupling-as-forgetting-vs-ungrounded-inference)=
### 12.4 Over/Under Coupling as Forgetting vs Ungrounded Inference

The coupling window in Theorem {prf:ref}`thm-information-stability-window-operational` reflects a trade-off:
- **Over-coupling:** noisy or overly aggressive updates drive mixing; the macro register loses stable structure (forgetting / symbol dispersion).
- **Under-coupling:** insufficient boundary information causes internal rollouts to dominate (model drift / ungrounded inference; Mode D.C).

The Sieve (Sections 3–6) is the control layer that keeps the agent inside the regime where macrostates remain stable *and* grounded.

(sec-optional-operator-valued-belief-updates)=
### 12.5 Optional: Operator-Valued Belief Updates (GKSL / "Lindblad" Form)

This subsection is optional. It provides a rigorous way to parameterize belief evolution so that **positivity** and **normalization** are structural (by construction), and so that “conservative prediction” and “dissipative grounding” are separated in the update law.

The starting point is to represent an internal belief not only as a vector $p_t\in\Delta^{|\mathcal{K}|-1}$, but as a positive semidefinite operator.

:::{prf:definition} Belief operator
:label: def-belief-operator

Let $\varrho_t\in\mathbb{C}^{d\times d}$ satisfy $\varrho_t\succeq 0$ and $\mathrm{Tr}(\varrho_t)=1$. Diagonal $\varrho_t$ reduces to a classical probability vector; non-diagonal terms can be used to encode correlations/uncertainty structure in a learned feature basis.

:::
:::{prf:definition} GKSL generator
:label: def-gksl-generator

A continuous-time, Markovian, completely-positive trace-preserving (CPTP) evolution has a generator of the Gorini–Kossakowski–Sudarshan–Lindblad (GKSL) form {cite}`gorini1976completely,lindblad1976generators`:

$$
\frac{d\varrho}{ds}
=
\underbrace{-i[H,\varrho]}_{\text{conservative drift}}
\;+\;
\underbrace{\sum_{j} \gamma_j\left(L_j\varrho L_j^\dagger-\frac12\{L_j^\dagger L_j,\varrho\}\right)}_{\text{dissipative update}},
$$
where {math}`H=H^\dagger` is Hermitian, {math}`\gamma_j\ge 0` are rates, and {math}`\{L_j\}` are (learned) operators.

**Operational interpretation (within this document).**
- The commutator term is a structured way to represent **reversible internal prediction** (it preserves $\mathrm{Tr}(\varrho)$ and the spectrum of $\varrho$).
- The dissipator is a structured way to represent **irreversible assimilation / disturbance** while preserving positivity and trace.

This is a modeling choice, not a claim about literal quantum physics: it is used here purely as a convenient, well-posed parametrization of CPTP belief updates.

*Note (WFR Embedding).* The GKSL generator embeds naturally into the Wasserstein-Fisher-Rao framework (Section 20.5): the commutator $-i[H, \varrho]$ corresponds to **transport** (continuous belief flow), while the dissipator $\sum_j \gamma_j(\cdot)$ corresponds to **reaction** (discrete mass creation/destruction). This provides a geometric foundation for the otherwise algebraic GKSL construction.

:::

::::{admonition} Physics Isomorphism: Lindblad Master Equation
:class: note
:name: pi-lindblad

**In Physics:** The GKSL (Gorini-Kossakowski-Sudarshan-Lindblad) equation describes the evolution of open quantum systems: $\dot{\varrho} = -i[H,\varrho] + \sum_k \gamma_k(L_k\varrho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \varrho\})$. It is the most general Markovian, completely positive, trace-preserving (CPTP) evolution {cite}`lindblad1976generators,gorini1976completely`.

**In Implementation:** The belief density evolution (Definition {prf:ref}`def-gksl-generator`):

$$
\mathcal{L}_{\text{GKSL}}(\varrho) = -i[H_{\text{eff}}, \varrho] + \sum_k \gamma_k \left( L_k \varrho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \varrho\} \right)
$$
**Correspondence Table:**
| Open Quantum Systems | Agent (Belief Dynamics) |
|:---------------------|:------------------------|
| Density matrix $\varrho$ | Belief distribution $\rho$ |
| Hamiltonian $H$ | Effective potential $\Phi_{\text{eff}}$ |
| Lindblad operators $L_k$ | Jump operators (chart transitions) |
| Decoherence rate $\gamma_k$ | Transition rates |
| CPTP evolution | Probability-preserving dynamics |

**Diagnostic:** MECCheck (Node 22) monitors $\|\dot{\varrho} - \mathcal{L}_{\text{GKSL}}(\varrho)\|_F^2$.
::::

(sec-master-equation-consistency-defect)=
#### 12.5.3 Master-Equation Consistency Defect (Node 22)

If an implementation maintains an operator belief $\varrho_t$ and produces an empirical update $\varrho_{t+1}$ (e.g., after a boundary update + Sieve projection), then a **consistency defect** compares it to the GKSL-predicted infinitesimal update:

$$
\mathcal{L}_{\text{MEC}}
:=
\left\|
\frac{\varrho_{t+1}-\varrho_t}{\Delta t}
\;-\;
\mathcal{L}_{\text{GKSL}}(\varrho_t)
\right\|_F^2,
$$
where $\mathcal{L}_{\text{GKSL}}(\cdot)$ denotes the right-hand side of Definition 11.5.2. This is the quantity monitored by MECCheck (Node 22).

(sec-residual-event-codebook)=
#### 12.5.4 Residual-Event ("Jump") Codebook (Links to Section 3.3.B)

The GKSL form becomes implementable if we can parameterize a *finite* family of disturbance/update types. With the nuisance/texture split (Section 2.2b), the disturbance library should attach to the **structured nuisance** channel, not to texture. A practical route is a discrete codebook over one-step nuisance residuals:
1. Compute a one-step prediction $(k_{t+1}^{\text{pred}}, z_{n,t+1}^{\text{pred}}):=S(K_t,z_{n,t},a_t)$ from the world model (macro + nuisance only).
2. Encode the next observation to obtain $(K_{t+1}, z_{n,t+1}, z_{\mathrm{tex},t+1})$ via the shutter.
3. Form the **nuisance residual** $\Delta z_{n,t}:=z_{n,t+1}-z_{n,t+1}^{\text{pred}}$.
4. Quantize $\Delta z_{n,t}$ with a second VQ module to obtain $J_t\in\{1,\dots,|\mathcal{J}|\}$.

Texture $z_{\mathrm{tex}}$ is treated as an emission/likelihood residual: it is used to model $p(x_t\mid K_t,z_{n,t},z_{\mathrm{tex},t})$ but is not used to define jump types. This is the formal reconciliation: “jumps” model **structured disturbances**, while “texture” models **measurement detail**.

The index $J_t$ can be used in two ways:
- **Classical residual modeling:** store representative nuisance residual vectors (or residual distributions) per code and train a conditional noise model $p(\Delta z_n\mid J)$.
- **Operator-valued modeling (optional):** associate each residual code $j$ with a learned low-rank operator $L_j$ and let rates $\gamma_j$ be predicted online; this is the operator analogue of a mixture-of-disturbances model.

The core engineering benefit is identifiability: the agent exposes a discrete label for “what kind of unmodeled disturbance happened”, rather than forcing the macro register to absorb it.

(sec-update-vs-evidence-check-and-metric-speed-limit)=
#### 12.5.5 Update vs Evidence Check (Node 23) and Metric Speed Limit (Node 24)

Even without operator beliefs, the same “no free update” principle can be monitored in classical terms:
- **Update vs evidence (NEPCheck).** Penalize belief updates that change faster than boundary information supports:

  $$
  \mathcal{L}_{\text{NEP}}
  :=
  \mathrm{ReLU}\!\left(D_{\mathrm{KL}}(p_{t+1}\Vert p_t)-I(X_t;K_t)\right)^2.
  $$
  This is a conservative audit metric: it does not assert a physical entropy law, but it detects ungrounded internal updating relative to measured boundary coupling (Node 13).
- **Metric speed limit (QSLCheck).** Impose a hard/soft bound on how far internal state may move per step under the state-space metric:

  $$
  \mathcal{L}_{\text{QSL}}:=\mathrm{ReLU}\!\left(d_G(z_{t+1},z_t)-v_{\max}\right)^2,
  $$
  which is a geometry-consistent generalization of KL-per-update constraints (ZenoCheck).

::::{note} Connection to RL #19: POMDP Belief Updates as Degenerate Belief Dynamics
**The General Law (Fragile Agent):**
Belief evolution follows the **Filtering + Projection Template** on the discrete macro register:

$$
p_{t+1}(k') = \frac{L_{t+1}(k')\, \tilde{p}_{t+1}(k')}{\sum_j L_{t+1}(j)\, \tilde{p}_{t+1}(j)}, \quad \tilde{p}_{t+1}(k') = \sum_k p_t(k)\, \bar{P}(k'|k,a_t)
$$
with **Sieve projections** applied after each update: hard masking or soft reweighting to enforce feasibility constraints.

**The Degenerate Limit:**
Remove the Sieve projections ($\text{feasible}(k) = 1$ for all $k$). Use continuous beliefs without discrete macro-register.

**The Special Case (Standard RL):**

$$
b_{t+1}(s') \propto O(o_{t+1}|s') \sum_s T(s'|s,a) b_t(s)
$$
This recovers standard **POMDP belief updates** {cite}`kaelbling1998planning` without safety constraints.

**What the generalization offers:**
- **Safety-aware beliefs**: Sieve projections (Section 12.3) remove probability mass from unsafe states *before* action selection
- **Discrete auditable symbols**: $H(K) \le \log|\mathcal{K}|$ provides hard capacity bound; standard POMDPs have unbounded continuous beliefs
- **Constraint enforcement**: Gate Nodes trigger belief reweighting when diagnostics fail (NEPCheck, QSLCheck)
- **Operator-valued updates**: Section 12.5 extends to GKSL/Lindblad form for quantum-like belief decoherence
::::



(sec-correspondence-table-filtering-control-template)=
## 13. Correspondence Table: Filtering / Control Template

The table below is a dictionary from standard **filtering and constrained inference** to the Fragile Agent components. It is purely classical: belief evolution is “predict → update → project”.

| Filtering / Control Object                                | Fragile Agent Equivalent                       | Role                          |
|:----------------------------------------------------------|:-----------------------------------------------|:------------------------------|
| Belief state $p_t(k)$                                     | Macro belief over $\mathcal{K}$                | Summary statistic for control |
| Prediction $\tilde p_{t+1}=\bar{P}^\top p_t$              | Macro dynamics model $\bar{P}(k'\mid k,a)$     | One-step forecast             |
| Likelihood $L_{t+1}(k)=p(x_{t+1}\mid k)$                  | Shutter/emission score for macrostates         | Boundary grounding signal     |
| Bayes update $p_{t+1}\propto L_{t+1}\odot \tilde p_{t+1}$ | Assimilation step                              | Incorporate observations      |
| Projection / reweighting $p'_t$                           | Sieve checks (CostBoundCheck, CompactCheck, …) | Enforce feasibility/stability |
| Entropy $H(p_t)$                                          | Macro uncertainty / symbol mixing              | Detect collapse vs dispersion |
| KL-control $D_{\mathrm{KL}}(\pi\Vert\pi_0)$               | Control-effort regularizer                     | Penalize deviation from prior |



(sec-duality-of-exploration-and-soft-optimality)=
