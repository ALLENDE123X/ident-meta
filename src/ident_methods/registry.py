"""
Method registry for dynamically registering IDENT methods.
"""

from typing import Dict, List, Optional, Type
from .base import IdentMethodBase


class MethodRegistry:
    """
    Registry for IDENT method implementations.
    
    This allows new methods to be registered dynamically without
    modifying the core selector code.
    
    Usage:
        from src.ident_methods import METHOD_REGISTRY
        
        # Register a method instance
        METHOD_REGISTRY.register(MyIdentMethod())
        
        # Get a registered method
        method = METHOD_REGISTRY.get("MyIdentMethod")
        
        # List all registered methods
        names = METHOD_REGISTRY.list_methods()
    """
    
    def __init__(self):
        self._methods: Dict[str, IdentMethodBase] = {}
    
    def register(self, method: IdentMethodBase) -> None:
        """
        Register an IDENT method.
        
        Args:
            method: Instance of a class implementing IdentMethodBase
        
        Raises:
            TypeError: If method doesn't implement IdentMethodBase
            ValueError: If a method with this name is already registered
        """
        if not isinstance(method, IdentMethodBase):
            raise TypeError(
                f"Method must inherit from IdentMethodBase, got {type(method)}"
            )
        
        name = method.name
        if name in self._methods:
            raise ValueError(f"Method '{name}' is already registered")
        
        self._methods[name] = method
    
    def get(self, name: str) -> Optional[IdentMethodBase]:
        """
        Get a registered method by name.
        
        Args:
            name: Method name (case-sensitive)
        
        Returns:
            IdentMethodBase instance or None if not found
        """
        return self._methods.get(name)
    
    def get_or_raise(self, name: str) -> IdentMethodBase:
        """
        Get a registered method by name, raising if not found.
        
        Args:
            name: Method name (case-sensitive)
        
        Returns:
            IdentMethodBase instance
        
        Raises:
            KeyError: If method is not registered
        """
        method = self._methods.get(name)
        if method is None:
            available = ", ".join(self._methods.keys()) or "(none)"
            raise KeyError(
                f"Method '{name}' is not registered. Available: {available}"
            )
        return method
    
    def list_methods(self) -> List[str]:
        """
        List all registered method names.
        
        Returns:
            List of method names
        """
        return list(self._methods.keys())
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a method by name.
        
        Args:
            name: Method name to unregister
        
        Returns:
            True if method was unregistered, False if it wasn't registered
        """
        if name in self._methods:
            del self._methods[name]
            return True
        return False
    
    def clear(self) -> None:
        """Remove all registered methods."""
        self._methods.clear()
    
    def __len__(self) -> int:
        return len(self._methods)
    
    def __contains__(self, name: str) -> bool:
        return name in self._methods
    
    def __repr__(self) -> str:
        methods = ", ".join(self._methods.keys())
        return f"MethodRegistry([{methods}])"


# Global registry instance
METHOD_REGISTRY = MethodRegistry()
