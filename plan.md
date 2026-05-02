# Prism-Federated — System Design: Privacy-Preserving Federated Learning

## 1. Problem Statement

Modern AI requires massive datasets, but data is often siloed in hospitals or on personal devices due to privacy laws (GDPR, HIPAA). Federated Learning allows training a global model without the data ever leaving its original location.

## 2. Key Requirements

- **Privacy:** Raw data never moves; only model weights (or gradients) are transmitted. Production systems should add TLS for transport, differential privacy (e.g. Opacus), and/or secure aggregation (MPC) depending on threat model.
- **Network Resilience:** Must handle "stragglers" (nodes that disconnect or have slow internet) via timeouts, partial client sampling, and asynchronous or buffered aggregation strategies.
- **Security:** Protection against "model poisoning" (malicious nodes trying to bias the model) via robust aggregation, anomaly scoring, or trusted execution environments — not covered in the MVP code path.
- **Communication Efficiency:** Minimize update size (compression, selective layer updates, structured pruning).

## 3. High-Level Architecture

The system uses a **central aggregator** and **distributed clients**:

1. **Initialization:** The server sends a base model (e.g., a CNN for medical imaging) to a subset of participating clients.
2. **Local training:** Each client trains the model on its *local* data for a few epochs.
3. **Weight upload:** Clients send weight tensors to the server (this repository: **FedAvg** over gRPC using Flower; no MPC in MVP).
4. **Global aggregation:** The server combines updates into a new **global model** (FedAvg: sample-size–weighted average).
5. **Iteration:** The process repeats until stopping criteria (fixed rounds or target metric).

## 4. Tech Stack

| Component | Technology |
| :--- | :--- |
| FL framework (MVP) | [Flower](https://flower.ai/) + PyTorch |
| Communication (MVP) | gRPC (Flower default; local demo uses cleartext on `127.0.0.1`) |
| Orchestration (future) | Kubernetes / KubeEdge for edge nodes |
| Differential privacy (future) | Opacus (PyTorch) |
| Strong privacy transport (future) | TLS certificates + optional SecAgg / MPC |

## 5. Unique Challenges

- **Non-IID data:** Nodes may have very different label or feature distributions, hurting convergence. Mitigations: more rounds, personalized layers, FedProx, scaffold, or better client sampling.
- **Incentivization:** How to reward nodes for high-quality participation without leaking private information.
- **Verification:** Demonstrating model improvement without centralizing raw data (e.g. held-out local metrics aggregated by the server).

## 6. MVP Scope (This Repository)

Runnable **FedAvg on MNIST** with **IID shards** per client:

- `fed_ml/model.py` — small CNN classifier.
- `fed_ml/data.py` — MNIST download into `./data`, IID partition by client id.
- `fed_ml/client.py` — `NumPyClient` with local SGD training.
- `run_federated.py` — one **Flower server** process and **N client** processes (no Ray; avoids optional `flwr` simulation backends).

Out of scope for the MVP: MPC, Opacus noise, Kubernetes, poisoning defenses, medical imaging datasets.

## 7. Environment and Execution

```bash
cd Prism-Federated
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_federated.py --num-clients 3 --num-rounds 2
```

The parent process downloads MNIST once into `./data` before spawning workers (avoids corrupting archives when several clients download in parallel). First run needs network access. By default the server uses **127.0.0.1 with an ephemeral free port** so repeated runs do not hit “port already in use.” Pass `--server-address 127.0.0.1:8080` only when you need a fixed port. Other flags: `--batch-size`, `--local-epochs`, `--startup-delay`. Expect Flower deprecation warnings for `start_server` / `start_numpy_client` until the project migrates to SuperLink / SuperNode or `flwr run`.

## 8. Roadmap

1. Non-IID splits (pathological or Dirichlet label skew) and metric dashboards.
2. TLS for gRPC; document threat model vs. cleartext lab setups.
3. Straggler handling: `fraction_fit` < 1, round timeouts, async FL patterns.
4. Optional Ray-backed Flower simulation for larger-scale tests.
5. Integration tests in CI (few rounds, CPU-only).
