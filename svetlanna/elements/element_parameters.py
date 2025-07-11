from abc import ABC
from dataclasses import dataclass, asdict
from typing import Any, TYPE_CHECKING, TypeVar, Callable, Union
import sys

if TYPE_CHECKING:
    from .element import Element

T = TypeVar('T', bound='ElementParameters')


def element_params(element_class: Union[type['Element'], str]) -> Callable[[type[T]], type[T]]:
    """
    Decorator to associate ElementParameters with their Element class.
    
    Parameters
    ----------
    element_class : type[Element] or str
        The Element class or its name as string.
        
    Returns
    -------
    Callable[[type[T]], type[T]]
        Decorator function.
        
    Examples
    --------
    >>> @element_params('FreeSpace')
    >>> @dataclass
    >>> class FreeSpaceParameters(ElementParameters):
    ...     distance: float
    ...     method: str = 'AS'
    """
    def decorator(params_class: type[T]) -> type[T]:
        if isinstance(element_class, str):
            # Create a property that resolves lazily
            def get_element_class(cls):
                # Check if already resolved
                if hasattr(cls, '_resolved_element_class'):
                    return cls._resolved_element_class
                
                # Try to resolve from module globals
                module = sys.modules[cls.__module__]
                if hasattr(module, element_class):
                    cls._resolved_element_class = getattr(module, element_class)
                    return cls._resolved_element_class
                else:
                    raise AttributeError(f"Element class '{element_class}' not found in module {cls.__module__}")
            
            # Set as class method
            setattr(params_class, 'element_class', classmethod(get_element_class))
        else:
            # Direct class reference
            params_class.element_class = element_class
        return params_class
    return decorator


@dataclass
class ElementParameters(ABC):
    """
    Base class for optical element parameters.
    
    Use @element_params decorator to associate with Element class.
    
    Examples
    --------
    >>> @element_params('MyElement')
    >>> @dataclass
    >>> class MyElementParameters(ElementParameters):
    ...     param1: float
    ...     param2: str = "default"
    """
    
    def to_kwargs(self) -> dict[str, Any]:
        """Convert parameters to keyword arguments dictionary."""
        return asdict(self)
    
    def validate(self) -> None:
        """Validate parameter values. Override in subclasses."""
        pass 
