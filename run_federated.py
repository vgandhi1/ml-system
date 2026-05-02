#!/usr/bin/env python3
"""Run a small Flower FedAvg job: one server process and N client processes (no Ray)."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import socket
import time

import flwr as fl

from fed_ml.client import MnistFlowerClient
from fed_ml.data import ensure_mnist_downloaded


def _server_main(
    server_address: str,
    num_rounds: int,
    num_clients: int,
) -> None:
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
        ),
    )
    print("Federated run complete.")
    if history.losses_distributed:
        rnd, loss = history.losses_distributed[-1]
        print(f"  Last distributed eval — round {rnd}, loss={loss:.4f}")
    acc_hist = history.metrics_distributed.get("accuracy")
    if acc_hist:
        rnd, acc = acc_hist[-1]
        print(f"  Last distributed eval — round {rnd}, accuracy={float(acc):.4f}")


def _client_main(
    server_address: str,
    client_id: int,
    num_clients: int,
    batch_size: int,
    local_epochs: int,
) -> None:
    client = MnistFlowerClient(
        client_id=client_id,
        num_clients=num_clients,
        batch_size=batch_size,
        local_epochs=local_epochs,
    )
    fl.client.start_numpy_client(server_address=server_address, client=client)


def _default_server_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        _host, port = s.getsockname()
    return f"127.0.0.1:{port}"


def main() -> None:
    parser = argparse.ArgumentParser(description="FedAvg MNIST via Flower (multi-process).")
    parser.add_argument(
        "--server-address",
        default=None,
        metavar="HOST:PORT",
        help="gRPC address (default: 127.0.0.1 with an ephemeral free port).",
    )
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=2.0,
        help="Seconds to wait after starting server before launching clients.",
    )
    args = parser.parse_args()
    server_address = args.server_address or _default_server_address()

    ensure_mnist_downloaded()

    server = mp.Process(
        target=_server_main,
        args=(server_address, args.num_rounds, args.num_clients),
        name="flower-server",
    )
    server.start()
    time.sleep(args.startup_delay)

    clients: list[mp.Process] = []
    for cid in range(args.num_clients):
        p = mp.Process(
            target=_client_main,
            args=(
                server_address,
                cid,
                args.num_clients,
                args.batch_size,
                args.local_epochs,
            ),
            name=f"flower-client-{cid}",
        )
        p.start()
        clients.append(p)

    server.join()
    for p in clients:
        p.join(timeout=120)
        if p.exitcode is None:
            p.terminate()
    if server.exitcode not in (0, None):
        raise SystemExit(server.exitcode)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
