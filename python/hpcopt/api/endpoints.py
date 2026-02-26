"""Route handler assembly for the HPC Workload Optimizer API."""

from __future__ import annotations

from typing import Any

from hpcopt.api.model_cache import get_runtime_predictor
from hpcopt.api.routes_admin import register_admin_routes
from hpcopt.api.routes_health import register_health_routes
from hpcopt.api.routes_predict import register_prediction_routes
from hpcopt.api.routes_recommendations import register_recommendation_routes
from hpcopt.utils.resilience import CircuitBreaker

_prediction_circuit = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)


def _runtime_predictor_loader() -> tuple[Any, Any]:
    # Compatibility hook: tests patch `hpcopt.api.endpoints.get_runtime_predictor`.
    return get_runtime_predictor()


def register_routes(app: Any) -> None:
    """Register all API route handlers on the FastAPI app instance."""
    register_health_routes(app)
    register_prediction_routes(
        app,
        prediction_circuit=_prediction_circuit,
        runtime_predictor_loader=_runtime_predictor_loader,
    )
    register_admin_routes(app)
    register_recommendation_routes(app)
