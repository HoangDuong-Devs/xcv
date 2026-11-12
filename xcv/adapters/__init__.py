"""Model Adapter Registry for hot-swapping models in xcv."""

from typing import Dict, Type
from xcv.core.base_adapter import BaseModelAdapter
from .card_quality_2c import CardQuality2CAdapter
from .template_adapter import TemplateCustomAdapter

# Register available model adapters here. Keys correspond to adapter.id.
ADAPTER_REGISTRY: Dict[str, Type[BaseModelAdapter]] = {
    CardQuality2CAdapter.id: CardQuality2CAdapter,
    TemplateCustomAdapter.id: TemplateCustomAdapter,
}


def create_adapter(adapter_id: str) -> BaseModelAdapter:
    if adapter_id not in ADAPTER_REGISTRY:
        raise KeyError(f"Unknown adapter '{adapter_id}'. Registered: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[adapter_id]()
