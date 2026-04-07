# All Runs Needed

## pen-binary-v0 (1M steps)

| nq | mq | drop | seeds | purpose | status | job ID | score |
|----|-----|------|-------|---------|--------|--------|-------|
| 2 | 1 | 0 | 0 | grid | done | 49894145 | -9944.0 |
| 2 | 2 | 0 | 0 | grid | done | 49894145 | -4454.5 |
| 2 | 2 | 0 | 1 | grid | done | 49894146 | -4968.0 |
| 2 | 2 | 0 | 2 | grid | done | 49894146 | -5003.0 |
| 2 | 2 | 0 | 3 | grid | done | 49894147 | -4764.0 |
| 2 | 2 | 0 | 4 | grid | done | 49894147 | -4159.0 |
| 4 | 1 | 0 | 0 | grid | done | 49894148 | -9537.0 |
| 4 | 2 | 0 | 0 | grid | done | 49894148 | -2921.5 |
| 6 | 1 | 0 | 0 | grid | done | 49894149 | -5300.5 |
| 6 | 2 | 0 | 0 | grid | done | 49894149 | -2188.0 |
| 10 | 1 | 0 | 0 | grid | done | 49894150 | -3672.5 |
| 10 | 2 | 0 | 0 | grid | done | 49894150 | -2093.0 |
| 10 | 2 | 0 | 1 | grid | done | 49894161 | -2137.5 |
| 10 | 2 | 0 | 2 | grid | done | 49894161 | -2036.0 |
| 10 | 2 | 0 | 3 | grid | done | 49894162 | -2081.5 |
| 10 | 2 | 0 | 4 | grid | done | 49894162 | -2041.5 |
| 2 | 1 | 0.01 | 0 | dropout x mq | done | 50038645 | -3551.5 |
| 2 | 2 | 0.005 | 0 | rate sweep | done | 50038557 | -3002.0 |
| 2 | 2 | 0.01 | 0 | rate sweep | done | 50038557 | -2835.0 |
| 2 | 2 | 0.01 | 1 | headline seeds | done | 50038572 | -3063.5 |
| 2 | 2 | 0.01 | 2 | headline seeds | done | 50038572 | -3453.0 |
| 2 | 2 | 0.01 | 3 | headline seeds | done | 50038577 | -2679.5 |
| 2 | 2 | 0.01 | 4 | headline seeds | done | 50038577 | -2610.5 |
| 2 | 2 | 0.02 | 0 | rate sweep | done | 50038619 | -2679.5 |
| 2 | 2 | 0.05 | 0 | rate sweep | done | 50038619 | -3141.0 |
| 2 | 2 | 0.1 | 0 | rate sweep | done | 50038645 | -3128.0 |
| 4 | 2 | 0.01 | 0 | dropout x N | done | 50038669 | -2576.5 |
| 6 | 2 | 0.01 | 0 | dropout x N | done | 50038680 | -2272.5 |
| 10 | 2 | 0.01 | 0 | dropout x N | done | 50038669 | -2149.5 |

### pen-binary DIAG runs

| nq | mq | drop | seed | purpose | status | job ID | score |
|----|-----|------|------|---------|--------|--------|-------|
| 2 | 1 | 0 | 0 | 2x2 diagnostic | done | 49894177 | -9940.5 |
| 2 | 2 | 0 | 0 | 2x2 diagnostic | done | 49894177 | -5285.5 |
| 2 | 1 | 0.01 | 0 | 2x2 diagnostic | done | 49894178 | -2894.5 |
| 2 | 2 | 0.01 | 0 | 2x2 diagnostic | **TODO** | — | — |
| 4 | 2 | 0 | 0 | sharpness vs N | done | 49894179 | -2866.5 |
| 6 | 2 | 0 | 0 | sharpness vs N | done | 49894184 | -2339.0 |
| 10 | 2 | 0 | 0 | sharpness vs N | done | 49894185 | -2008.0 |

## door-binary-v0 (1M steps)

| nq | mq | drop | seeds | purpose | status | job ID | score |
|----|-----|------|-------|---------|--------|--------|-------|
| 2 | 1 | 0 | 0 | death spiral | done | 49894172 | -11249.0 |
| 2 | 2 | 0 | 0 | baseline | done | 49894172 | -17151.5 |
| 2 | 2 | 0 | 1 | baseline | done | 49894173 | -4224.5 |
| 2 | 2 | 0 | 2 | baseline | done | 49894174 | -5387.5 |
| 10 | 1 | 0 | 0 | grid | **CRASHED** | 49894168 | — |
| 10 | 2 | 0 | 0 | baseline | done | 49894175 | -4637.0 |
| 10 | 2 | 0 | 1 | baseline | done | 49894175 | -3344.0 |
| 10 | 2 | 0 | 2 | baseline | done | 49894176 | -4401.5 |
| 2 | 1 | 0.01 | 0 | dropout fix | done | 50044220 | -5219.5 |
| 2 | 2 | 0.01 | 0 | dropout fix | done | 50038680 | -5187.5 |
| 2 | 2 | 0.01 | 1 | dropout seeds | done | 50044212 | -3924.0 |
| 2 | 2 | 0.01 | 2 | dropout seeds | done | 50044212 | -3832.0 |
| 10 | 2 | 0.01 | 0 | dropout neutral | done | 50044220 | -4983.0 |

### door-binary DIAG runs

| nq | mq | drop | seed | purpose | status | job ID | score |
|----|-----|------|------|---------|--------|--------|-------|
| 2 | 1 | 0 | 0 | diagnostic | done | 49894186 | -11594.5 |
| 2 | 2 | 0 | 0 | diagnostic | done | 49894186 | -13287.5 |
| 2 | 2 | 0.01 | 0 | sharpness transfer | done | 50044235 | -4190.0 |

## antmaze-large-diverse-v2 (300k steps)

| nq | mq | drop | seeds | purpose | status |
|----|-----|------|-------|---------|--------|
| 2 | 2 | 0 | 0,1,2 | ensemble redundant | TODO |
| 10 | 2 | 0 | 0,1,2 | ensemble redundant | TODO |

## locomotion — walker2d-medium-v0 or halfcheetah-medium-v0 (250k steps)

| nq | mq | drop | seeds | purpose | status |
|----|-----|------|-------|---------|--------|
| 2 | 2 | 0 | 0,1,2 | dense reward domain | TODO |
| 2 | 2 | 0.01 | 0,1,2 | dropout on dense reward | TODO |
| 10 | 2 | 0 | 0,1,2 | baseline | TODO |

## Summary

- **pen-binary**: 29/29 standard done, 6/7 DIAG done (1 TODO: nq2 mq2 drop0.01 diag)
- **door-binary**: 13/13 standard done (includes 1 crash: nq10 mq1 drop0), 3/3 DIAG done
- **antmaze**: 0/6 TODO
- **locomotion**: 0/9 TODO
- **Total done**: 51 runs completed, 1 crashed, 1 DIAG TODO, 15 new-env TODO

### Crashed run to re-submit
- `door-binary-v0,0,10,1,0,1000000` — crashed in first batch (job 49894168), was paired with pen-binary nq10 drop0.01. Needs solo re-submit.

### DIAG TODO
- `DIAG=1 pen-binary-v0,0,2,2,0.01,1000000` — the mq2 dropout diagnostic (mq1 version exists from job 49894178)
