"""
Component Registry System

Provides dependency injection and component management for the modular architecture.
"""

from .component_registry import ComponentRegistry, register_component, get_component, get_registry

__all__ = [
    "ComponentRegistry",
    "register_component",
    "get_component",
    "get_registry"
]
