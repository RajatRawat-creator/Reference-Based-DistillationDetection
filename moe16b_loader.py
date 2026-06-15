#!/usr/bin/env python3
"""
Shared loader for deepseek-ai/deepseek-moe-16b-base.

Importing this module (BEFORE importing run_wild / run_o1_ascii_unicode) sets the
hub-kernel env vars and applies the compatibility monkeypatches that the DeepSeek
remote modeling code needs. The patches are lifted verbatim from the proven
NewScripts/run_deepseekmoe16b_loss_collected.py (which ran this exact model); the
FP8 ones are no-ops for a bf16 load but are kept for safety.

Use:
    import moe16b_loader as moe          # sets env + applies patches, imports transformers
    import run_wild as rw                # reuses transformers already imported/patched
    model, tok = moe.load_moe_model(...)
"""

import os

# MUST run before transformers is imported anywhere (run_wild imports it on import).
os.environ.setdefault("DISABLE_KERNEL_MAPPING", "1")
os.environ.setdefault("HF_HUB_DISABLE_KERNELS", "1")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/deepseek-moe-16b-base"


# ---- compat patch: remote code expects is_torch_fx_available -----------------
import transformers.utils.import_utils as _tf_import_utils
if not hasattr(_tf_import_utils, "is_torch_fx_available"):
    def _is_torch_fx_available():
        return False
    _tf_import_utils.is_torch_fx_available = _is_torch_fx_available


# ---- compat patch: finegrained_fp8._first_attr (no-op for bf16) --------------
try:
    import transformers.integrations.finegrained_fp8 as _tf_fp8
    _orig_first_attr = _tf_fp8._first_attr

    def _patched_first_attr(obj, *names):
        extra = ("n_routed_experts",)
        for name in (*names, *extra):
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(f"{type(obj).__name__} has none of: {(*names, *extra)}")

    _tf_fp8._first_attr = _patched_first_attr
except Exception as _e:  # pragma: no cover
    print(f"[WARN] could not patch finegrained_fp8._first_attr: {_e}", flush=True)


# ---- compat patch: replace_with_fp8_linear experts iteration (no-op bf16) -----
try:
    import torch.nn as _nn
    import transformers.integrations.finegrained_fp8 as _tf_fp8m
    _orig_replace_with_fp8_linear = _tf_fp8m.replace_with_fp8_linear

    def _patched_replace_with_fp8_linear(model, modules_to_not_convert=None,
                                         quantization_config=None, pre_quantized=False):
        if quantization_config is None or getattr(quantization_config, "dequantize", False):
            import inspect as _inspect
            _orig_params = _inspect.signature(_orig_replace_with_fp8_linear).parameters
            _kwargs = {"modules_to_not_convert": modules_to_not_convert,
                       "quantization_config": quantization_config}
            if "pre_quantized" in _orig_params:
                _kwargs["pre_quantized"] = pre_quantized
            return _orig_replace_with_fp8_linear(model, **_kwargs)
        if not all(hasattr(_tf_fp8m, _n) for _n in (
                "FP8Experts", "ALL_FP8_EXPERTS_FUNCTIONS",
                "use_experts_implementation", "should_convert_module")):
            import inspect as _inspect
            _orig_params = _inspect.signature(_orig_replace_with_fp8_linear).parameters
            _kwargs = {"modules_to_not_convert": modules_to_not_convert,
                       "quantization_config": quantization_config}
            if "pre_quantized" in _orig_params:
                _kwargs["pre_quantized"] = pre_quantized
            return _orig_replace_with_fp8_linear(model, **_kwargs)

        FP8Linear = _tf_fp8m.FP8Linear
        FP8Experts = _tf_fp8m.FP8Experts
        ALL_FP8_EXPERTS_FUNCTIONS = _tf_fp8m.ALL_FP8_EXPERTS_FUNCTIONS
        use_experts_implementation = _tf_fp8m.use_experts_implementation
        should_convert_module = _tf_fp8m.should_convert_module

        snapshot = list(model.named_modules())
        pending = []
        replaced_prefixes = []
        for module_name, module in snapshot:
            if not should_convert_module(module_name, modules_to_not_convert):
                continue
            if any(module_name == p or module_name.startswith(p + ".") for p in replaced_prefixes):
                continue
            module_kwargs = {} if pre_quantized else {"dtype": None}
            new_module = None
            with torch.device("meta"):
                if module_name.endswith(".experts"):
                    has_gate = getattr(module, "has_gate", True)
                    has_bias = getattr(module, "has_bias", False)
                    config = getattr(module, "config", model.config.get_text_config())
                    new_class = use_experts_implementation(
                        experts_class=FP8Experts,
                        experts_interface=ALL_FP8_EXPERTS_FUNCTIONS,
                        has_bias=has_bias, has_gate=has_gate)
                    new_module = new_class(
                        config=config,
                        block_size=quantization_config.weight_block_size,
                        activation_scheme=quantization_config.activation_scheme,
                        has_bias=has_bias, has_gate=has_gate, **module_kwargs)
                    replaced_prefixes.append(module_name)
                elif isinstance(module, _nn.Linear):
                    new_module = FP8Linear(
                        in_features=module.in_features, out_features=module.out_features,
                        block_size=quantization_config.weight_block_size,
                        activation_scheme=quantization_config.activation_scheme,
                        has_bias=module.bias is not None, **module_kwargs)
            if new_module is not None:
                pending.append((module_name, new_module))
        for name, new_module in pending:
            model.set_submodule(name, new_module)
        if not pending:
            import logging
            logging.getLogger(__name__).warning("fp8 patch: no linear/experts modules were replaced.")
        return model

    _tf_fp8m.replace_with_fp8_linear = _patched_replace_with_fp8_linear
    try:
        import transformers.quantizers.quantizer_finegrained_fp8 as _tf_q_fp8
        _tf_q_fp8.replace_with_fp8_linear = _patched_replace_with_fp8_linear
    except Exception:
        pass
except Exception as _e:  # pragma: no cover
    print(f"[WARN] could not patch finegrained_fp8.replace_with_fp8_linear: {_e}", flush=True)


# ---- compat patch: FP8 weight init normal_ (no-op for bf16) ------------------
try:
    _FP8_DTYPES = tuple(dt for dt in (
        getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None), getattr(torch, "float8_e5m2fnuz", None),
    ) if dt is not None)
    _orig_tensor_normal_ = torch.Tensor.normal_

    def _safe_tensor_normal_(self, *args, **kwargs):
        if self.dtype in _FP8_DTYPES:
            return self
        return _orig_tensor_normal_(self, *args, **kwargs)

    torch.Tensor.normal_ = _safe_tensor_normal_
except Exception as _e:  # pragma: no cover
    print(f"[WARN] could not patch torch.Tensor.normal_ for FP8: {_e}", flush=True)


# ---- compat patch: torch 2.6 infer_schema PEP585 generics (no-op bf16) -------
try:
    import typing as _typing
    import torch._library.infer_schema as _torch_infer_schema
    _SPT = _torch_infer_schema.SUPPORTED_PARAM_TYPES
    _new_entries = {}
    for _typ, _schema in list(_SPT.items()):
        _origin = getattr(_typ, "__origin__", None)
        _args = getattr(_typ, "__args__", None)
        if _origin in (list, _typing.List) and _args:
            _lower = list[_args[0]]
            if _lower not in _SPT:
                _new_entries[_lower] = _schema
        if _origin is _typing.Union and _args:
            _non_none = [a for a in _args if a is not type(None)]
            if len(_non_none) == 1:
                _inner = _non_none[0]
                _inner_origin = getattr(_inner, "__origin__", None)
                _inner_args = getattr(_inner, "__args__", None)
                if _inner_origin in (list, _typing.List) and _inner_args:
                    _lower = _typing.Optional[list[_inner_args[0]]]
                    if _lower not in _SPT:
                        _new_entries[_lower] = _schema
    _SPT.update(_new_entries)
except Exception as _e:  # pragma: no cover
    print(f"[WARN] could not patch torch infer_schema for PEP 585 generics: {_e}", flush=True)


# =============================================================================
def dtype_from_string(dtype: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}[dtype]


def build_max_memory(per_gpu="130GiB", cpu="200GiB"):
    if per_gpu is None:
        return None
    if not torch.cuda.is_available():
        return {"cpu": cpu}
    mm = {i: per_gpu for i in range(torch.cuda.device_count())}
    mm["cpu"] = cpu
    return mm


def load_model_config(model_name=MODEL_NAME, trust_remote_code=True, local_files_only=False):
    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only)
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "type" not in rope_scaling:
        rope_type = rope_scaling.get("rope_type")
        if rope_type in (None, "default"):
            # Official config has rope_scaling=null; newer Transformers may
            # materialize {"rope_type": "default"}; remote code expects None.
            config.rope_scaling = None
        elif rope_type in ("linear", "dynamic"):
            rope_scaling = dict(rope_scaling)
            rope_scaling["type"] = rope_type
            config.rope_scaling = rope_scaling
        else:
            raise ValueError(f"Unsupported DeepSeek-MoE rope_scaling: {rope_scaling}")
    return config


def load_tokenizer(model_name=MODEL_NAME, trust_remote_code=True, local_files_only=False):
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
            local_files_only=local_files_only, fix_mistral_regex=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only)
    tok.truncation_side = "left"
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_moe_model(dtype="bfloat16", device_map="auto",
                   max_memory_per_gpu="130GiB", max_memory_cpu="200GiB",
                   model_name=MODEL_NAME):
    """Load deepseek-moe-16b-base (bf16, ~33GB) with eval/use_cache=False."""
    config = load_model_config(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map=device_map,
        torch_dtype=dtype_from_string(dtype),   # transformers 4.51.3 uses torch_dtype, not dtype
        trust_remote_code=True,
        local_files_only=False,
        low_cpu_mem_usage=True,
        max_memory=build_max_memory(max_memory_per_gpu, max_memory_cpu),
    )
    model.config.use_cache = False
    model.eval()
    return model
