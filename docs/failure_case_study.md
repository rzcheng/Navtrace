# Failure and recovery case study for review

Read `Replay::process`, `Ekf::update_gps`, `audit_gps_events`,
`state_statistics`, and `sensitivity_matrix`; then inspect the
[actual M3 results](m3_results.md). Generated implementation is not evidence
that Ryan has personally authored or understood every component.

1. **What went wrong:** after little GPS conditioning, yaw rate and bias remain
   coupled. During loss, the trajectory drifts. In M2 seed 70002, returning GPS
   with a common offset is accepted while prediction uncertainty is large.
2. **Why the gate behaved that way:** the first bad fix has directional NIS
   contributions 1.759+.289=2.048. After repeated corrections, the prediction is
   confident around a displaced position. The first clean fix has contributions
   56.436+112.590=169.026 and is rejected. NIS cannot identify which source is right.
3. **What evidence identifies the mechanism:** unchanged C++ trajectories,
   saved pre/post matrices, independent K/Joseph/NIS calculations, directional
   ellipses, zero rejected-GPS corrections, and paired new-seed failures. A second
   route, weak motion followed by clean GPS, also fails without corrupt fixes.
4. **What changed:** diagnostics, independent analysis, and a bounded experiment.
   The estimator and parameters did not change. An offline absent-outage label
   bug was separately corrected with a failing-then-passing regression.
5. **When it helps:** the new tooling distinguishes weak information, covariance
   mismatch, and numerical errors. It does not itself improve localization. The
   original gate helps later-offset cases but loses badly after early offsets.
6. **What remains ambiguous:** a smooth common GPS offset and accumulated
   dead-reckoning drift may be indistinguishable without another absolute
   reference or a stated trust assumption. Rank and PSD do not settle that choice.

Five questions to answer using the code and saved records:

- Why does gyro-only rank equal one, and how can moving straight GPS distinguish
  yaw rate from bias even without a turn?
- Why is position NEES based on P[:2,:2], and why did using the full inverse's
  position block change the independent hand result from 1 to 4/3?
- How can a small directional residual contribute more NIS than a large one?
  Reconstruct both terms of the 40 s and 45 s seed 70002 decisions.
- Why do 30 independent runs at one time support a different interpretation
  from 6,001 correlated samples in one trajectory? What does the 2 s sanity
  ensemble fail to test about a 20 s outage?
- Why can a run meet the two-second settling criterion and still end wrong?
  What new trust assumption would a reacquisition policy need, and which
  persistent-offset negative control could refute it?

An explainable engineering claim would name the synthetic scenarios, unchanged
baselines, evaluation protocol, and observed failures. Do not turn these results
into a universal robustness, recovery guarantee, or real-world performance claim.
