"""
Component Registry

Core component registry for dependency injection and component management.
"""

import asyncio
import inspect
from typing import Dict, Any, Type, Optional, List, TypeVar, Protocol, Union
from threading import Lock
import logging

T = TypeVar('T')

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Central registry for managing components and their dependencies.
    
    Provides dependency injection, component lifecycle management, and
    configuration-driven component creation.
    """
    
    def __init__(self):
        self._components: Dict[str, Dict[str, Type]] = {}
        self._instances: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._initialized = False
    
    def register(
        self, 
        interface: Union[Type, str], 
        implementation: Type, 
        name: str,
        config: Optional[Dict[str, Any]] = None,
        singleton: bool = False
    ) -> None:
        """
        Register a component implementation for an interface.
        
        Args:
            interface: Interface class or string name
            implementation: Implementation class
            name: Unique name for this implementation
            config: Optional configuration for the implementation
            singleton: Whether to create a singleton instance
        """
        with self._lock:
            interface_name = interface if isinstance(interface, str) else interface.__name__
            
            if interface_name not in self._components:
                self._components[interface_name] = {}
            
            self._components[interface_name][name] = implementation
            
            # Store configuration
            component_key = f"{interface_name}:{name}"
            if config:
                self._configs[component_key] = config
            
            # Mark as singleton if requested
            if singleton:
                self._singletons[component_key] = None
            
            logger.info(f"Registered {implementation.__name__} as '{name}' for {interface_name}")
    
    def create(
        self, 
        interface: Union[Type, str], 
        name: str,
        config_override: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Create an instance of a registered component.
        
        Args:
            interface: Interface class or string name
            name: Name of the implementation to create
            config_override: Optional configuration override
            **kwargs: Additional arguments for component creation
            
        Returns:
            Component instance
        """
        with self._lock:
            interface_name = interface if isinstance(interface, str) else interface.__name__
            component_key = f"{interface_name}:{name}"
            
            # Check if it's a singleton and already created
            if component_key in self._singletons and self._singletons[component_key] is not None:
                return self._singletons[component_key]
            
            # Get the implementation class
            if interface_name not in self._components:
                raise ValueError(f"No implementations registered for interface: {interface_name}")
            
            if name not in self._components[interface_name]:
                raise ValueError(f"No implementation '{name}' registered for interface: {interface_name}")
            
            implementation_class = self._components[interface_name][name]
            
            # Prepare configuration
            config = {}
            if component_key in self._configs:
                config.update(self._configs[component_key])
            if config_override:
                config.update(config_override)
            
            # Create the instance
            try:
                instance = self._create_instance(implementation_class, config, **kwargs)
                
                # Store singleton if needed
                if component_key in self._singletons:
                    self._singletons[component_key] = instance
                
                logger.info(f"Created instance of {implementation_class.__name__} ('{name}')")
                return instance
                
            except Exception as e:
                logger.error(f"Failed to create instance of {implementation_class.__name__}: {e}")
                raise
    
    async def create_async(
        self, 
        interface: Union[Type, str], 
        name: str,
        config_override: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Create an instance of a registered component asynchronously.
        
        Args:
            interface: Interface class or string name
            name: Name of the implementation to create
            config_override: Optional configuration override
            **kwargs: Additional arguments for component creation
            
        Returns:
            Component instance
        """
        # Use thread pool for CPU-bound instantiation
        loop = asyncio.get_event_loop()
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, 
                self.create, 
                interface, 
                name, 
                config_override,
                **kwargs
            )
    
    def list_implementations(self, interface: Union[Type, str]) -> List[str]:
        """
        List all registered implementations for an interface.
        
        Args:
            interface: Interface class or string name
            
        Returns:
            List of implementation names
        """
        interface_name = interface if isinstance(interface, str) else interface.__name__
        
        with self._lock:
            if interface_name not in self._components:
                return []
            return list(self._components[interface_name].keys())
    
    def list_interfaces(self) -> List[str]:
        """
        List all registered interfaces.
        
        Returns:
            List of interface names
        """
        with self._lock:
            return list(self._components.keys())
    
    def get_component_info(self, interface: Union[Type, str], name: str) -> Dict[str, Any]:
        """
        Get information about a registered component.
        
        Args:
            interface: Interface class or string name
            name: Name of the implementation
            
        Returns:
            Component information dictionary
        """
        interface_name = interface if isinstance(interface, str) else interface.__name__
        component_key = f"{interface_name}:{name}"
        
        with self._lock:
            if interface_name not in self._components or name not in self._components[interface_name]:
                raise ValueError(f"Component '{name}' not found for interface '{interface_name}'")
            
            implementation_class = self._components[interface_name][name]
            
            return {
                "interface": interface_name,
                "name": name,
                "implementation": implementation_class.__name__,
                "module": implementation_class.__module__,
                "is_singleton": component_key in self._singletons,
                "has_config": component_key in self._configs,
                "config": self._configs.get(component_key, {}),
                "is_instantiated": component_key in self._singletons and self._singletons[component_key] is not None
            }
    
    def configure_component(
        self, 
        interface: Union[Type, str], 
        name: str, 
        config: Dict[str, Any]
    ) -> None:
        """
        Configure a registered component.
        
        Args:
            interface: Interface class or string name
            name: Name of the implementation
            config: Configuration dictionary
        """
        interface_name = interface if isinstance(interface, str) else interface.__name__
        component_key = f"{interface_name}:{name}"
        
        with self._lock:
            if component_key not in self._configs:
                self._configs[component_key] = {}
            self._configs[component_key].update(config)
            
            logger.info(f"Updated configuration for {component_key}")
    
    def _create_instance(
        self, 
        implementation_class: Type, 
        config: Dict[str, Any], 
        **kwargs
    ) -> Any:
        """
        Create an instance of the implementation class with dependency injection.
        
        Args:
            implementation_class: Class to instantiate
            config: Configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Component instance
        """
        # Get constructor signature
        sig = inspect.signature(implementation_class.__init__)
        
        # Prepare constructor arguments
        init_args = {}
        
        # Add config if the constructor accepts it
        if 'config' in sig.parameters:
            init_args['config'] = config
        
        # Add any additional kwargs that match constructor parameters
        for param_name, param in sig.parameters.items():
            if param_name in kwargs:
                init_args[param_name] = kwargs[param_name]
        
        # Create the instance
        return implementation_class(**init_args)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                "interfaces": len(self._components),
                "total_implementations": sum(len(impls) for impls in self._components.values()),
                "singletons": len(self._singletons),
                "instantiated_singletons": len([s for s in self._singletons.values() if s is not None]),
                "configured_components": len(self._configs)
            }


# Global registry instance
_registry = None
_registry_lock = Lock()

def get_registry() -> ComponentRegistry:
    """Get the global component registry instance."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ComponentRegistry()
    return _registry


def register_component(
    interface: Union[Type, str], 
    implementation: Type, 
    name: str,
    config: Optional[Dict[str, Any]] = None,
    singleton: bool = False
) -> None:
    """
    Register a component implementation.
    
    Args:
        interface: Interface class or string name
        implementation: Implementation class
        name: Unique name for this implementation
        config: Optional configuration
        singleton: Whether to create a singleton instance
    """
    registry = get_registry()
    registry.register(interface, implementation, name, config, singleton)


def get_component(
    interface: Union[Type, str], 
    name: str,
    config_override: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Get a component instance from the registry.
    
    Args:
        interface: Interface class or string name
        name: Name of the implementation
        config_override: Optional configuration override
        **kwargs: Additional arguments
        
    Returns:
        Component instance
    """
    registry = get_registry()
    return registry.create(interface, name, config_override, **kwargs) 