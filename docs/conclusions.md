## 19. Conclusion

The Fragile Agent is a capacity- and stability-constrained control specification: the environment is described by an observation generator $P_{\partial}$, the agent’s internal state is a split macro/micro bundle, and stability is enforced by explicit defect functionals rather than implicit optimizer heuristics.

1. Discretizing the macro channel $K$ turns enclosure, capacity, and grounding into well-typed information constraints ($I(X;K)$, $H(K)$, closure cross-entropy).
2. The critic induces a Fisher/Hessian sensitivity geometry; the policy becomes a regulated flow on a curved manifold, with stability and coupling audited by Gate Nodes and Barriers.
3. Exploration and control are expressed via MaxEnt/KL-control and causal path entropy on $\mathcal{K}$; belief evolution is filtering + projection (Section 11).
4. When representational complexity is constrained by finite interface capacity, the latent metric obeys a capacity-constrained consistency law; deviations yield computable consistency defects (Section 18).
5. The hybrid discrete-continuous state space admits a canonical Wasserstein-Fisher-Rao geometry (Section 20), unifying transport (continuous motion within charts) and reaction (discrete jumps between charts) in a single variational principle.

Appendix A records the full derivations. Appendix B consolidates notation and all regularization losses.



(sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces)=
