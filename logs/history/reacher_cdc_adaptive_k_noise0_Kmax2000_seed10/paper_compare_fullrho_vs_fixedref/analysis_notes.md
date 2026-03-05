# Analysis Notes: full-rho adaptive vs fixed reference

## Scope
- Adaptive uses rho=0.001~1.5 combined from two summaries.
- Fixed reference uses rho=0.005 summary_by_K sweep.

## Data provenance warning
- low-rho (0.001~0.2): reconstructed from waypoint-tagged runs.
- high-rho (0.3~1.5): adaptive summary rows.
- Interpretation of absolute levels should be conservative across these sources.

## Fixed reference
- K*: 90
- success_rate(K*): 1.0000
- median_solve_ms(K*): 289.526
- median_sigma_min_Ag(K*): 8.362e-08

## Tau and collapse marker
- tau (from fixed turning/min point): 1.196e-09
- K_turn: 50

## Validation
- rho set exact match: True
- runs==10 for all rho: True

## Selected recovery trajectories
- rho=0.05, seed=3, collapse=0, K-up=55, recover=56, score=26.793
- rho=0.1, seed=4, collapse=14, K-up=24, recover=25, score=25.703
- rho=0.2, seed=5, collapse=0, K-up=15, recover=16, score=12.616
- rho=0.05, seed=0, collapse=0, K-up=44, recover=45, score=21.893

## Paper-aligned takeaway template
- Adaptive uses near-threshold K allocation over rho while avoiding unnecessary large fixed K.
- Failures are concentrated near low sigma_min; K increase aligns with sigma_min recovery and reduced failures.
- Mixed source condition (waypoint/non-waypoint) is explicitly reported.
