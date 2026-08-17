"""First-party Qwen3.8-27B Core AI authoring and export support for caix."""

from .contract import Qwen38Architecture, Qwen38ContractError, build_metadata

__all__ = ["Qwen38Architecture", "Qwen38ContractError", "build_metadata"]
