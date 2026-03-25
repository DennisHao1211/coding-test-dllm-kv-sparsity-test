from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelLoadInfo:
    model_path: str
    tokenizer_path: str
    code_path: str | None
    used_local_code: bool
    local_files_only: bool
    tokenizer_local_files_only: bool
    model_class_file: str | None


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean flag.")


def _existing_dir(path_like: str | os.PathLike[str] | None) -> Path | None:
    if path_like is None:
        return None
    candidate = Path(os.path.expanduser(str(path_like))).resolve()
    return candidate if candidate.is_dir() else None


def resolve_code_path(
    model_path: str | os.PathLike[str],
    code_path: str | os.PathLike[str] | None = None,
) -> Path | None:
    if code_path is None:
        code_path = os.getenv("FAST_DLLM_CODE_PATH")

    resolved_code_path = _existing_dir(code_path)
    if resolved_code_path is not None:
        return resolved_code_path

    return _existing_dir(model_path)


def _pick_local_module_file(code_dir: Path, module_name: str) -> tuple[Path, str]:
    module_rel = Path(*module_name.split("."))
    direct_file = (code_dir / module_rel).with_suffix(".py")
    if direct_file.is_file():
        return direct_file, direct_file.stem

    fallback_rel = module_rel.with_name(f"modified_{module_rel.name}")
    fallback_file = (code_dir / fallback_rel).with_suffix(".py")
    if fallback_file.is_file():
        return fallback_file, fallback_file.stem

    raise FileNotFoundError(
        f"Could not find a local module file for {module_name!r} under {code_dir}."
    )


def _ensure_import_package(code_dir: Path) -> str:
    package_name = f"_fast_dllm_local_{sha1(str(code_dir).encode()).hexdigest()[:12]}"
    if package_name in sys.modules:
        return package_name

    package = types.ModuleType(package_name)
    package.__path__ = [str(code_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    return package_name


def _load_local_module(code_dir: Path, module_name: str):
    module_file, module_basename = _pick_local_module_file(code_dir, module_name)
    package_name = _ensure_import_package(code_dir)
    qualified_name = f"{package_name}.{module_basename}"

    if qualified_name in sys.modules:
        return sys.modules[qualified_name]

    spec = importlib.util.spec_from_file_location(qualified_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {module_file}.")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _split_auto_map_target(target: Any, fallback: str) -> tuple[str, str]:
    if isinstance(target, (list, tuple)):
        target = next(
            (item for item in target if isinstance(item, str) and item),
            None,
        )
    if not target:
        target = fallback

    module_name, class_name = target.rsplit(".", 1)
    return module_name, class_name


def _load_local_model_classes(code_dir: Path):
    config_path = code_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json under {code_dir}.")

    config_data = json.loads(config_path.read_text())
    auto_map = config_data.get("auto_map") or {}
    architecture_name = config_data.get("architectures", ["Fast_dLLM_QwenForCausalLM"])[0]

    config_module_name, config_class_name = _split_auto_map_target(
        auto_map.get("AutoConfig"),
        "configuration.Fast_dLLM_QwenConfig",
    )
    model_module_name, model_class_name = _split_auto_map_target(
        auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModel"),
        f"modeling.{architecture_name}",
    )

    config_module = _load_local_module(code_dir, config_module_name)
    model_module = _load_local_module(code_dir, model_module_name)

    config_cls = getattr(config_module, config_class_name)
    model_cls = getattr(model_module, model_class_name)
    return config_cls, model_cls, model_module


def load_causal_lm_and_tokenizer(
    model_path: str | os.PathLike[str],
    *,
    code_path: str | os.PathLike[str] | None = None,
    tokenizer_path: str | os.PathLike[str] | None = None,
    local_files_only: bool | str | None = None,
    tokenizer_local_files_only: bool | str | None = None,
    print_load_source: bool = True,
    **model_kwargs,
):
    model_kwargs = dict(model_kwargs)
    model_kwargs.pop("trust_remote_code", None)
    model_kwargs.pop("local_files_only", None)

    model_path_str = str(model_path)
    tokenizer_path_str = str(tokenizer_path or model_path)
    resolved_code_path = resolve_code_path(model_path_str, code_path)

    explicit_local_files_only = _coerce_optional_bool(local_files_only)
    explicit_tokenizer_local_files_only = _coerce_optional_bool(tokenizer_local_files_only)

    model_local_files_only = (
        explicit_local_files_only
        if explicit_local_files_only is not None
        else _existing_dir(model_path_str) is not None
    )
    tokenizer_local_only = (
        explicit_tokenizer_local_files_only
        if explicit_tokenizer_local_files_only is not None
        else _existing_dir(tokenizer_path_str) is not None
    )

    if resolved_code_path is not None:
        config_cls, model_cls, model_module = _load_local_model_classes(resolved_code_path)
        config = config_cls.from_pretrained(
            model_path_str,
            local_files_only=model_local_files_only,
        )
        model = model_cls.from_pretrained(
            model_path_str,
            config=config,
            local_files_only=model_local_files_only,
            **model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path_str,
            trust_remote_code=False,
            local_files_only=tokenizer_local_only,
        )
        load_info = ModelLoadInfo(
            model_path=model_path_str,
            tokenizer_path=tokenizer_path_str,
            code_path=str(resolved_code_path),
            used_local_code=True,
            local_files_only=model_local_files_only,
            tokenizer_local_files_only=tokenizer_local_only,
            model_class_file=getattr(model_module, "__file__", None),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path_str,
            trust_remote_code=True,
            local_files_only=model_local_files_only,
            **model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path_str,
            trust_remote_code=True,
            local_files_only=tokenizer_local_only,
        )
        load_info = ModelLoadInfo(
            model_path=model_path_str,
            tokenizer_path=tokenizer_path_str,
            code_path=None,
            used_local_code=False,
            local_files_only=model_local_files_only,
            tokenizer_local_files_only=tokenizer_local_only,
            model_class_file=None,
        )

    if print_load_source:
        if load_info.used_local_code:
            print(f"[loader] model weights : {load_info.model_path}")
            print(f"[loader] tokenizer     : {load_info.tokenizer_path}")
            print(f"[loader] local code    : {load_info.code_path}")
            print(f"[loader] model class   : {load_info.model_class_file}")
        else:
            print(f"[loader] model weights : {load_info.model_path}")
            print(f"[loader] tokenizer     : {load_info.tokenizer_path}")
            print("[loader] local code    : <disabled>")
            print("[loader] model class   : Hugging Face dynamic module loader")

    return model, tokenizer, load_info
