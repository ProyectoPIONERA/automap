"""
Annotation Decorators

This module provides decorators for marking deprecated and unimplemented code.
"""

import functools
import warnings


def deprecated(reason=''):
    """
    Decorator to mark a function or class as deprecated.
    
    Args:
        reason: Optional explanation for why it's deprecated
    
    Example:
        @deprecated("Use new_function instead")
        def old_function():
            pass
    """
    def decorator(cls_or_func):
        msg = f"This call is deprecated and will be removed in a future version. Reason: {reason}"

        @functools.wraps(cls_or_func)
        def new_func(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return cls_or_func(*args, **kwargs)

        return new_func
    return decorator


def todo(reason=''):
    """
    Decorator to mark a function or class as not yet implemented.
    
    Args:
        reason: Optional explanation for the TODO
    
    Example:
        @todo("Needs implementation for fuzzy matching")
        def fuzzy_match():
            pass
    """
    def decorator(cls_or_func):
        msg = f"This call is not implemented yet. Reason: {reason}"

        @functools.wraps(cls_or_func)
        def new_func(*args, **kwargs):
            warnings.warn(msg, category=PendingDeprecationWarning, stacklevel=2)
            return cls_or_func(*args, **kwargs)

        return new_func
    return decorator
