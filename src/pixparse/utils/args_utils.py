import dataclasses
from dataclasses import fields

from typing import Optional

def get_selected_non_default_args(dataclass_instance, arg_names, rename_map: Optional[dict] = {}):
    """
    Extracts a subset of non-default arguments from a dataclass instance.

    This checks a specified list of argument names in a given instance. 
    It returns a dictionary of arguments that are not set to their default values.

    Parameters:
    - dataclass_instance: An instance of a dataclass from which to extract arguments.
    - arg_names: A list of strings representing the names of the arguments to be considered.
    - rename_map: Optional dictionary to rename argument keys.

    Returns:
    - A dictionary containing key-value pairs of argument names and their values,
      for those arguments that are not set to their default values.
    """
    selected_non_default_args = {}
    for field in fields(dataclass_instance.__class__):
        if field.name in arg_names:
            value = getattr(dataclass_instance, field.name)
            default_value = field.default
            if field.default_factory != dataclasses.MISSING:
                default_value = field.default_factory()

            if value != default_value:
                key = rename_map[field.name] if rename_map and field.name in rename_map else field.name
                selected_non_default_args[key] = value

    return selected_non_default_args