JAX implementation of the variational quantum transformer for synthetic language generation ([arXiv:2609.18565](https://arxiv.org/abs/2609.18565)). Compact NISQ encoder–connector–decoder circuits keep an autoregressive next-token interface with a two-qubit readout over a 4-token vocabulary.

```
pip install -e ".[tests]"
python scripts/smoke.py
pytest
```

## What does not match the paper

- Which two qubits are kept after each partial trace is not named in the text; this repo keeps the first two wires of each variational block (top wires in Fig. 2) and the decoder ancilla after the connector.
- Decoder stacking is implemented as: every layer couples the same encoder state ρ_E into a running 2-qubit ancilla (fresh |00⟩ on the first layer), then re-tensors the original context ρ_x. The paper’s wiring of ρ_E vs. the previous decoder output is underspecified.
- Strongly entangling layers follow PennyLane’s default ranges r_l = (l mod (n−1))+1; the paper cites PennyLane but does not print the ranges.
- Angle initialization is uniform on [0, 2π] (not stated). Grammar-score components are the four quantities named in §4; the paper does not give the exact aggregation beyond “averages”.
- The compact classical transformer baseline is not implemented. Smoke training uses a few Adam steps, not 20 epochs × 3 seeds.
