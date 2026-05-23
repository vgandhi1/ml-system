"""Simulates real-time eCommerce transaction events to Kafka."""

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic


@dataclass
class SalesEvent:
    event_id: str
    customer_id: str
    stock_code: str
    quantity: int
    unit_price: float
    country: str
    timestamp: str

    @property
    def revenue(self) -> float:
        return self.quantity * self.unit_price


class SalesEventProducer:
    TOPIC = "ecommerce.sales.events"

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
        )
        self._ensure_topic(bootstrap_servers)

    def _ensure_topic(self, servers: str):
        admin = KafkaAdminClient(bootstrap_servers=servers)
        existing = admin.list_topics()
        if self.TOPIC not in existing:
            admin.create_topics([NewTopic(self.TOPIC, num_partitions=3, replication_factor=1)])

    def publish(self, event: SalesEvent):
        self.producer.send(
            self.TOPIC,
            key=event.customer_id,
            value=asdict(event),
        )

    def flush(self):
        self.producer.flush()


def main():
    import os

    servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    producer = SalesEventProducer(servers)

    for i in range(100):
        event = SalesEvent(
            event_id=f"evt-{i:05d}",
            customer_id=str(random.randint(10000, 99999)),
            stock_code=random.choice(["85123A", "71053", "84406B"]),
            quantity=random.randint(1, 50),
            unit_price=round(random.uniform(1.0, 99.0), 2),
            country="UNITED KINGDOM",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        producer.publish(event)
        time.sleep(0.05)

    producer.flush()
    print(f"Published 100 events to {SalesEventProducer.TOPIC}")


if __name__ == "__main__":
    main()
