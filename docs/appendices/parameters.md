(sec-appendix-b-units-parameters-and-coefficients)=
## Appendix B: Units, Parameters, and Coefficients (Audit Table)

(sec-appendix-b-base-units)=
### B.1 Base Units (Information + Steps)

We use a purely information-theoretic unit system:
- **Information / cost:** nats ($\mathrm{nat}$).
- **Interaction time $t$:** discrete environment steps ($\mathrm{step}$).
- **Computation time $s$:** internal solver time (continuous; normalized units unless mapped to wall-clock).
- **Scale time $\tau$:** depth coordinate (dimensionless).
- **Memory time $t'$:** discrete past index ($\mathrm{step}$, with $t' < t$).

Conventions:
- Entropies $H(\cdot)$, mutual information $I(\cdot;\cdot)$, and divergences $D_{\mathrm{KL}}$ are measured in $\mathrm{nat}$.
- Value/cost scalars ($V$, $F_t$, budgets, thresholds) are measured in $\mathrm{nat}$.
- Per-step rates (HJB terms, $\Delta V$, any “cost rate”) are measured in $\mathrm{nat/step}$ (interaction time).
- Latent coordinates ($z$, $z_n$, $z_{\mathrm{tex}}$, code embeddings $e_k$) are treated as **normalized/dimensionless**; any physical units should be absorbed into preprocessing and encoder normalization.

(sec-appendix-b-parameter-coefficient-units)=
### B.2 Parameter / Coefficient Units (by Role)

| Symbol                                                                               | Meaning (context)                                                                                   | Units                                          |
|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------|
| $t$                                                                                  | interaction step index                                                                              | $\mathrm{step}$                                |
| $t'$                                                                                 | memory time index ($t' < t$)                                                                        | $\mathrm{step}$                                |
| $s$                                                                                  | computation time (internal solver)                                                                  | solver-time units (normalized)                 |
| $\tau$                                                                               | scale time (depth)                                                                                  | dimensionless                                  |
| $\Delta t$                                                                           | optional mapping from steps to wall-clock                                                           | $\mathrm{s/step}$                              |
| $r_t$                                                                                | reward/cost per step                                                                                | $\mathrm{nat}$                                 |
| $\mathcal{R}$                                                                        | reward/cost rate (when written as a rate)                                                           | $\mathrm{nat/step}$                            |
| $V$                                                                                  | value / cost-to-go                                                                                  | $\mathrm{nat}$                                 |
| $\Delta V$                                                                           | value change per step                                                                               | $\mathrm{nat/step}$                            |
| $\mathfrak{D}$                                                                       | control-effort / regularization rate term                                                           | $\mathrm{nat/step}$                            |
| $\lambda$                                                                            | Lyapunov rate in $\dot V\le -\lambda V$ (continuous-time form)                                      | $s^{-1}$                                       |
| $\gamma$                                                                             | discount factor (MaxEnt RL)                                                                         | dimensionless                                  |
| $H$                                                                                  | horizon / planning depth                                                                            | $\mathrm{step}$                                |
| $T_c$                                                                                | entropy-regularization coefficient (MaxEnt RL)                                                      | dimensionless                                  |
| $\beta$                                                                              | exponential-family scale in $\exp(-\beta V)$                                                        | dimensionless                                  |
| $\Theta$                                                                             | local conditioning proxy (Definition {prf:ref}`def-local-conditioning-scale`)                       | dimensionless (when $z$ normalized)            |
| $\epsilon$                                                                           | numeric stabilizer / threshold                                                                      | inherits compared quantity                     |
| $\eta$                                                                               | step size / learning-rate symbol                                                                    | dimensionless (in normalized coordinates)      |
| $\beta$                                                                              | VQ-VAE commitment weight (Section 2.2b / 3.3)                                                       | dimensionless                                  |
| $\beta_n$                                                                            | nuisance KL weight (structured residual)                                                            | dimensionless                                  |
| $\beta_{\mathrm{tex}}$                                                               | texture KL weight (reconstruction-only residual)                                                    | dimensionless                                  |
| $\beta_K$                                                                            | macro codelength weight (rate term)                                                                 | dimensionless                                  |
| $\lambda_{\text{use}}$                                                               | codebook usage regularizer weight                                                                   | dimensionless                                  |
| $\lambda_{\text{*}}$                                                                 | composite-loss weights (e.g. $\lambda_{\text{shutter}},\lambda_{\text{ent}},\lambda_{\text{zeno}}$) | dimensionless                                  |
| $\lambda,\mu,\nu$                                                                    | VICReg component weights                                                                            | dimensionless                                  |
| $\alpha,\beta,\gamma,\delta$                                                         | scaling coefficients (Section 3.2)                                                                  | dimensionless                                  |
| $V_{\text{max}},V_{\text{limit}},V_{\text{proxy}},V_{\text{true}},B_{\text{switch}}$ | risk/cost budgets and thresholds                                                                    | $\mathrm{nat}$                                 |
| $\lambda_{\text{in}},\lambda_{\text{mix}}$                                           | grounding/mixing information rates                                                                  | $\mathrm{nat/step}$                            |
| $q_{k\to k'}$                                                                        | macro transition probabilities                                                                      | dimensionless                                  |
| $C_{\partial}$                                                                       | boundary information capacity                                                                       | $\mathrm{nat}$                                 |
| $I_{\text{bulk}}$                                                                    | bulk information volume                                                                             | $\mathrm{nat}$                                 |
| $\ell$                                                                               | boundary resolution scale                                                                           | boundary-length units (chosen)                 |
| $\eta_\ell$                                                                          | boundary area-per-nat at resolution $\ell$                                                          | $[dA_G]/\mathrm{nat}$                          |
| $\Lambda$                                                                            | curvature/capacity offset constant (metric law)                                                     | $[z]^{-2}$                                     |
| $\kappa$                                                                             | coupling in $R_{ij}-\\tfrac12R G_{ij}+\\Lambda G_{ij}=\\kappa T_{ij}$                               | chosen so $\kappa T_{ij}$ has units $[z]^{-2}$ |
| $U(z)$                                                                               | hyperbolic information potential $-d_{\mathbb{D}}(0,z)$ (Section 21.1)                              | $\mathrm{nat}$                                 |
| $T_c(\tau)$                                                                          | generative temperature schedule (Section 21.2)                                                      | dimensionless                                  |
| $\kappa_T$                                                                           | temperature annealing rate (Section 21.2)                                                           | $\tau^{-1}$                                    |
| $\phi_c$                                                                             | Möbius automorphism moving $c$ to origin (Section 21.3)                                             | dimensionless (isometry)                       |
| $\lambda(z)$                                                                         | conformal factor $2/(1-\lvert z\rvert^2)$ (Section 21.4)                                            | $[z]^{-1}$                                     |
| $\sigma_{\text{tex}}$                                                                | base texture standard deviation (Section 21.4)                                                      | $[z_{\text{tex}}]$                             |
| $R_{\text{cutoff}}$                                                                  | geometric stopping radius (Section 21.5)                                                            | dimensionless                                  |
| $\epsilon_{\text{conv}}$                                                             | convergence stopping threshold (Section 21.5)                                                       | $\mathrm{nat}/\tau$                            |
| $S_{\mathrm{OM}}$                                                                    | Onsager-Machlup stochastic action (Section 22.1)                                                    | $\mathrm{nat}$                                 |
| $\Phi_{\text{gen}}$                                                                  | generative potential $\alpha U + (1-\alpha)V_{\text{critic}}$ (Section 22.3)                        | $\mathrm{nat}$                                 |
| $\alpha$                                                                             | generation-control interpolation parameter (Section 22.3)                                           | dimensionless                                  |
| $\gamma$                                                                             | friction coefficient in overdamped limit (Section 22.4)                                             | $s^{-1}$                                       |
| $\lambda_{\text{jump}}$                                                              | Poisson jump intensity (Section 22.2)                                                               | $s^{-1}$                                       |
| $\eta$                                                                               | multiplicative jump factor for mass (Section 22.2)                                                  | dimensionless                                  |
| $\mathcal{A}_y$                                                                      | sub-atlas for class $y$ (Section 25.1)                                                              | —                                              |
| $V_y$                                                                                | class-conditioned potential (Section 25.2)                                                          | $\mathrm{nat}$                                 |
| $\beta_{\text{class}}$                                                               | class temperature (inverse semantic diffusion) (Section 25.2)                                       | dimensionless                                  |
| $\mathcal{B}_y$                                                                      | attractor basin for class $y$ (Section 25.2)                                                        | —                                              |
| $\gamma_{\text{sep}}$                                                                | class separation strength (Section 25.3)                                                            | dimensionless                                  |
| $\lambda_{i\to j}^{\text{sup}}$                                                      | class-modulated jump rate (Section 25.3)                                                            | $s^{-1}$                                       |
| $\mathcal{L}_{\text{purity}}$                                                        | chart purity loss $H(Y\mid K)$ (Section 25.4)                                                       | $\mathrm{nat}$                                 |
| $\mathcal{L}_{\text{balance}}$                                                       | load balance loss (Section 25.4)                                                                    | $\mathrm{nat}$                                 |
| $\mathcal{L}_{\text{metric}}$                                                        | metric contrastive loss (Section 25.4)                                                              | $\mathrm{nat}$                                 |
| $\mathcal{L}_{\text{route}}$                                                         | route alignment loss (Section 25.4)                                                                 | $\mathrm{nat}$                                 |
| $\epsilon_{\text{purity}}$                                                           | purity threshold (Section 25.1)                                                                     | dimensionless                                  |

(sec-appendix-b-symbol-overload)=
### B.3 Symbol Overload (Important)

Some Greek letters are intentionally overloaded in different submodels:
- $\beta$ appears as (i) the exponential-family scale in $\exp(-\beta V)$, (ii) the VQ-VAE commitment weight, and (iii) the inverse-temperature/softmax scale in some contrastive losses; treat each by its local definition and units above.
- $\tau$ appears as (i) scale time (Section 1.3), (ii) the entropy-weight coefficient in Section 2.11.3, and (iii) the temperature in some contrastive losses; use local definitions.
- $\gamma$ appears as (i) discount factor, (ii) the World Model volatility scaling coefficient $\gamma$ (Section 3.2), and (iii) friction coefficient in overdamped dynamics (Section 22.4).
- $\lambda$ appears as (i) Lyapunov rate ($s^{-1}$), (ii) generic loss weights (dimensionless), and (iii) other Lagrange multipliers (units stated locally).



