# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Small scheduling/metadata regressions without Torch or model execution."""

import ast
import contextlib
import copy
import functools
import os
import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import Mock, patch

from test.engine.test_openwam_review import MODEL, load_definition
from turbo_physai.optimizations.models.openwam import settings


def nested_definition(relative, name, **namespace):
    path = MODEL / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


class CacheIdentityTest(unittest.TestCase):
    def setUp(self):
        self.validate, _ = load_definition("latent_cache.py", "validate_signature")
        self.meta = {
            "version": 1,
            "dtype": "float32",
            "vae": "same-weights",
            "readers": {
                "old-key": {
                    "data_root": "/old/dataset/task/clean_50/data",
                    "episodes": ["a"],
                    "window": 33,
                },
            },
        }
        self.expected = copy.deepcopy(self.meta)
        reader = self.expected["readers"].pop("old-key")
        reader["data_root"] = "/new/dataset/task/clean_50/data"
        self.expected["readers"]["new-key"] = reader

    def test_relocated_cache_uses_persisted_directory_key(self):
        self.assertEqual(
            self.validate(self.meta, self.expected), {"new-key": "old-key"}
        )

    def test_changed_window_and_ambiguous_matches_are_rejected(self):
        changed = copy.deepcopy(self.expected)
        changed["readers"]["new-key"]["window"] = 17
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.validate(self.meta, changed)
        self.meta["readers"]["duplicate"] = copy.deepcopy(
            self.meta["readers"]["old-key"]
        )
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.validate(self.meta, self.expected)

    def test_directory_key_cannot_escape_cache(self):
        self.meta["readers"]["../escape"] = self.meta["readers"].pop("old-key")
        with self.assertRaisesRegex(ValueError, "directory key"):
            self.validate(self.meta, self.expected)


class OwnerPlanTest(unittest.TestCase):
    def test_sparse_chunk_offsets_and_balanced_fallback(self):
        chunk, _ = load_definition("zero_fused.py", "Chunk", dataclass=dataclass)
        tiles, _ = load_definition("zero_fused.py", "tile_rows", BLOCK=4)
        plan, _ = load_definition(
            "zero_fused.py", "sparse_plan", Chunk=chunk, tile_rows=tiles
        )
        self.assertIsNone(plan([(0, 4, 0), (4, 4, 1)], 2, 4))
        self.assertIsNone(plan([(0, 0, 0)], 4, 4))
        result = plan([(0, 3, 0), (3, 1, 1), (4, 5, 0)], 4, 4)
        self.assertEqual(
            [(c.owner, c.count, c.direct) for c in result],
            [(0, 4, None), (0, 4, None), (1, 1, 3)],
        )
        self.assertEqual(result[0].rows, ((0, 0, 3), (4, 3, 1)))
        self.assertEqual(result[1].rows, ((5, 0, 4),))

    def test_bucket_copies_flush_before_storage_or_count_limit(self):
        torch = Mock()
        kind, _ = load_definition(
            "zero.py", "_BucketCopies", torch=torch, contextlib=contextlib
        )
        src = Mock(is_cuda=False)
        src.untyped_storage.return_value.nbytes.return_value = 8
        first, second, large_dst = Mock(), Mock(), Mock()
        copies = kind(max_bytes=12, max_tensors=2)
        copies.append(first, src)
        self.assertEqual(copies.bytes, 8)
        copies.append(second, src)
        first.copy_.assert_called_once_with(src.detach.return_value)
        self.assertEqual(copies.bytes, 8)
        copies.flush()
        self.assertEqual(copies.srcs, [])
        src.untyped_storage.return_value.nbytes.return_value = 20
        copies.append(large_dst, src)
        large_dst.copy_.assert_called_once_with(src)
        self.assertEqual(copies.bytes, 0)

    def test_parameter_rebinding_checks_later_offsets(self):
        original = Mock()
        update = nested_definition(
            "zero.py", "update_model_weights", orig_update_model_weights=original
        )

        def tensor(offset, count):
            value = Mock(dtype="bf16", device="gpu")
            value.storage_offset.return_value = offset
            value.numel.return_value = count
            value.untyped_storage.return_value.data_ptr.return_value = 100
            value.is_contiguous.return_value = True
            return value

        flat = tensor(0, 8)
        params = [
            types.SimpleNamespace(data=tensor(0, 4)),
            types.SimpleNamespace(data=tensor(4, 4)),
        ]
        optimizer = types.SimpleNamespace(
            bit16_groups_flat=[flat], bit16_groups=[params]
        )
        update(optimizer, 0)
        original.assert_not_called()
        params[1].data.storage_offset.return_value = 0
        update(optimizer, 0)
        original.assert_called_once_with(0)


class AttentionPlanTest(unittest.TestCase):
    def test_plan_rejects_mutation_and_cross_modality_query_groups(self):
        function, _ = load_definition(
            "attention.py", "separate_attention_plan", enabled=lambda name: True
        )
        mask = types.SimpleNamespace(
            ndim=2,
            shape=(5, 5),
            _version=1,
            _openwam_split_version=1,
            _openwam_split_plan=((0, 3, ((0, 3),)), (3, 5, ((0, 5),))),
        )
        self.assertEqual(
            function(mask, 3, 2), (((0, 3, ((0, 3),)),), ((0, 2, ((0, 5),)),))
        )
        mask._version = 2
        self.assertIsNone(function(mask, 3, 2))
        mask._version = 1
        mask._openwam_split_plan = ((0, 5, ((0, 5),)),)
        self.assertIsNone(function(mask, 3, 2))

    def test_separate_queries_reuse_packed_keys_without_joint_output(self):
        class Tensor:
            def __init__(self, rows):
                self.rows = list(rows)
                self.shape = (1, len(self.rows), 1)

            def __getitem__(self, key):
                return Tensor(self.rows[key[1]])

        torch = types.SimpleNamespace(
            cat=lambda tensors, dim: Tensor(
                [row for tensor in tensors for row in tensor.rows]
            )
        )
        function, _ = load_definition(
            "attention.py", "split_attention_separate", torch=torch
        )
        calls = []

        def attend(query, keys, values, num_heads):
            calls.append((keys.rows, values.rows))
            return query

        kv = (Tensor([10, 11, 12, 13, 14]), Tensor([20, 21, 22, 23, 24]))
        plan = (((0, 3, ((0, 3),)),), ((0, 2, ((0, 5),)),))
        video, action = function(
            Tensor([1, 2, 3]),
            None,
            None,
            Tensor([4, 5]),
            None,
            None,
            plan,
            1,
            attend,
            joint_kv=kv,
        )
        self.assertEqual((video.rows, action.rows), ([1, 2, 3], [4, 5]))
        self.assertEqual(
            calls, [([10, 11, 12], [20, 21, 22]), (kv[0].rows, kv[1].rows)]
        )


class LookaheadOrderTest(unittest.TestCase):
    def make_encoder(self):
        kind, _ = load_definition(
            "online_encode.py",
            "OnlineEncoder",
            torch=types.SimpleNamespace(no_grad=contextlib.nullcontext),
            PreparedInputs=lambda values: types.SimpleNamespace(values=values),
        )
        instance = kind.__new__(kind)
        backbone = types.SimpleNamespace()
        instance.architecture = types.SimpleNamespace(video_backbone=backbone)
        return instance, backbone

    def test_lookahead_flag_restores_missing_and_explicit_values_on_error(self):
        for previous in ("missing", None, False, True):
            for fail in (False, True):
                with self.subTest(previous=previous, fail=fail):
                    encoder, backbone = self.make_encoder()
                    if previous != "missing":
                        backbone._openwam_video_first = previous

                    def prepare(batch):
                        self.assertIs(backbone._openwam_video_first, True)
                        self.assertEqual(batch, ["batch"])
                        if fail:
                            raise RuntimeError("encode failed")
                        return {"prepared": True}

                    encoder.architecture.prepare_inputs = prepare
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "encode failed"):
                            encoder._prepare_lookahead("batch")
                    else:
                        self.assertEqual(
                            encoder._prepare_lookahead("batch").values,
                            {"prepared": True},
                        )
                    if previous == "missing":
                        self.assertNotIn("_openwam_video_first", vars(backbone))
                    else:
                        self.assertIs(backbone._openwam_video_first, previous)

    def test_gather_launch_precedes_encode_and_wait_runs_on_failure(self):
        for deferred in (False, True):
            for fail in (False, True):
                with self.subTest(deferred=deferred, fail=fail):
                    encoder, backbone = self.make_encoder()
                    events = []
                    dist = Mock()
                    dist.get_world_size.return_value = 2
                    dist.get_rank.return_value = 0
                    work = Mock()
                    work.wait.side_effect = lambda: events.append("wait")
                    dist.all_gather_into_tensor.side_effect = (
                        lambda *a, **kw: events.append("gather") or work
                    )
                    # Methods share the definition's stand-in globals.
                    encoder._gather.__func__.__globals__["dist"] = dist
                    encoder.deferred = (
                        types.SimpleNamespace(gather=lambda *a: events.append("gather"))
                        if deferred
                        else None
                    )
                    encoder.next_batch = lambda: "batch"
                    encoder.pending = None

                    def prepare(batch):
                        self.assertIs(backbone._openwam_video_first, True)
                        events.append("encode")
                        if fail:
                            raise RuntimeError("encode failed")
                        return {"done": True}

                    encoder.architecture.prepare_inputs = prepare
                    args = ([object()], [[object(), object()]], [object()], 1, 100)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "encode failed"):
                            encoder._gather(*args)
                        self.assertIsNone(encoder.pending)
                    else:
                        encoder._gather(*args)
                        self.assertEqual(encoder.pending.values, {"done": True})
                    self.assertEqual(
                        events, ["gather", "encode"] + ([] if deferred else ["wait"])
                    )
                    self.assertNotIn("_openwam_video_first", vars(backbone))
                    self.assertIsNone(encoder.next_batch)

    def test_no_lookahead_uses_native_gather_without_reordering(self):
        encoder, backbone = self.make_encoder()
        encoder.deferred = None
        encoder.next_batch = None
        encoder.original_gather = Mock(return_value="native")
        self.assertEqual(encoder._gather([], [], [], 1, 100), "native")
        encoder.original_gather.assert_called_once_with([], [], [], 1, 100)
        self.assertNotIn("_openwam_video_first", vars(backbone))

    def test_preprocess_order_preserves_text_first_for_cached_latents(self):
        for flag, cached, expected in [
            (False, False, ["text", "video"]),
            (True, False, ["video", "text"]),
            (True, True, ["text"]),
        ]:
            with self.subTest(flag=flag, cached=cached):
                events = []
                latent = Mock()
                latent.to.return_value = latent
                encode = types.SimpleNamespace(
                    encode_text=lambda *a, **kw: events.append("text")
                    or ("context", [1]),
                    encode_video=lambda *a, **kw: events.append("video") or latent,
                    preprocess_video=Mock(),
                )
                fn, _ = load_definition(
                    "ports/wan.py",
                    "WanBase_preprocess_input_for_train",
                    torch=types.SimpleNamespace(cat=Mock()),
                    wan_encode=encode,
                    check_resize_height_width=lambda h, w, n, **kw: (h, w, n),
                )
                backbone = types.SimpleNamespace(
                    device="gpu",
                    dtype="bf16",
                    _openwam_video_first=flag,
                    _height_division_factor=32,
                    _width_division_factor=32,
                    _time_division_factor=4,
                    _time_division_remainder=1,
                    _tokenizer=object(),
                    text_encoder=object(),
                    video_encoder=None,
                    vae=object(),
                    _variant=types.SimpleNamespace(
                        build_train_conditioning=Mock(return_value={})
                    ),
                )
                kw = (
                    {"video_latents": latent, "video_size": (1, 32, 32)}
                    if cached
                    else {}
                )
                result = fn(
                    backbone,
                    frames=[[types.SimpleNamespace(size=(32, 32))]],
                    text=["prompt"],
                    **kw,
                )
                self.assertEqual(events, expected)
                self.assertIs(result["input_latents"], latent)
                self.assertEqual(result["context"], "context")


class SchedulingTest(unittest.TestCase):
    def test_installed_options_follow_catalog_targets(self):
        module = types.ModuleType("openwam.model.action_backbone.components")

        def forward(self, value):
            return value

        forward.__openwam_group__ = "openwam.compute"
        forward.__openwam_options__ = {"rope_real": False}
        module.RMSNorm = type("RMSNorm", (), {"forward": forward})
        with patch.dict(sys.modules, {module.__name__: module}):
            self.assertEqual(
                settings._installed_groups(), {"openwam.compute": {"rope_real": False}}
            )

    def test_zero_bucket_default_preserves_explicit_override(self):
        configure, _ = load_definition(
            "__init__.py",
            "configure_zero",
            enabled=lambda name: name == "zero_overlap",
            os=os,
        )
        config = {"zero_optimization": {"stage": 2}}
        torch = types.SimpleNamespace(version=types.SimpleNamespace(hip=None))
        with patch.dict(sys.modules, {"torch": torch}):
            configure(config)
            self.assertEqual(
                config["zero_optimization"]["reduce_bucket_size"], 100_000_000
            )
            config["zero_optimization"]["reduce_bucket_size"] = 50_000_000
            configure(config)
            self.assertEqual(
                config["zero_optimization"]["reduce_bucket_size"], 50_000_000
            )

    def test_workspace_default_only_applies_to_hip_blaslt_zero2(self):
        configure, _ = load_definition(
            "__init__.py",
            "configure_zero",
            os=os,
            enabled=lambda name: name == "zero_overlap",
        )
        for hip, stage, prefer in [
            (None, 2, "1"),
            ("test", 1, "1"),
            ("test", 2, "0"),
            ("test", 2, "1"),
        ]:
            with self.subTest(hip=hip, stage=stage, prefer=prefer):
                torch = types.SimpleNamespace(version=types.SimpleNamespace(hip=hip))
                with patch.dict(sys.modules, {"torch": torch}), patch.dict(
                    os.environ, {"TORCH_BLAS_PREFER_HIPBLASLT": prefer}, clear=True
                ):
                    configure({"zero_optimization": {"stage": stage}})
                    self.assertEqual(
                        os.environ.get("HIPBLASLT_WORKSPACE_SIZE"),
                        "32768" if hip and stage == 2 and prefer == "1" else None,
                    )
        torch = types.SimpleNamespace(version=types.SimpleNamespace(hip="test"))
        for key in ("HIPBLASLT_WORKSPACE_SIZE", "CUBLASLT_WORKSPACE_SIZE"):
            env = {"TORCH_BLAS_PREFER_HIPBLASLT": "1", key: "0"}
            with self.subTest(override=key), patch.dict(
                sys.modules, {"torch": torch}
            ), patch.dict(os.environ, env, clear=True):
                configure({"zero_optimization": {"stage": 2}})
                self.assertEqual(dict(os.environ), env)

    def test_bias2d_uses_torch_backend_enum(self):
        tree = ast.parse((MODEL / "linear_bias2d.py").read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_CUBLAS_BACKEND" for t in n.targets
            )
        )
        expected = object()
        namespace = {
            "torch": types.SimpleNamespace(
                _C=types.SimpleNamespace(
                    _BlasBackend=types.SimpleNamespace(Cublas=expected)
                )
            )
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "backend", "exec"),
            namespace,
        )
        self.assertIs(namespace["_CUBLAS_BACKEND"], expected)

    def test_norm_take_waits_once_and_clear_drains_pending(self):
        torch = Mock()
        norm, _ = load_definition("zero_norm.py", "NormOverlap", torch=torch)
        instance = norm(object())
        instance.stream = types.SimpleNamespace(device="gpu")
        grads, params = [], []
        value, event = Mock(device="gpu"), object()
        instance.pending[id(grads)] = (id(grads), id(params), value, event)
        self.assertIsNone(instance.take(grads, []))
        self.assertIs(instance.take(grads, params), value)
        torch.cuda.current_stream.return_value.wait_event.assert_called_once_with(event)
        self.assertIsNone(instance.take(grads, params))
        instance.pending[0] = (0, 0, value, event)
        instance.clear()
        self.assertEqual(instance.pending, {})

    def test_fused_adamw_is_initialized_through_native_constructor(self):
        class AdamW:
            def __init__(self, groups, **defaults):
                self.param_groups, self.defaults = groups, defaults
                self.state = {}
                self.amp_support = defaults.get("fused", False)

        factory, _ = load_definition(
            "ports/trainer.py",
            "build_optimizer",
            functools=functools,
            torch=types.SimpleNamespace(optim=types.SimpleNamespace(AdamW=AdamW)),
            enabled=lambda name: True,
        )
        original_optimizer = AdamW([{"params": [object()]}], lr=0.1)
        original = Mock(return_value=original_optimizer)
        result = factory(original, {})(object())
        self.assertIsNot(result, original_optimizer)
        self.assertTrue(result.amp_support)
        self.assertFalse(result.defaults["foreach"])
        self.assertEqual(result.defaults["lr"], 0.1)

    def test_optional_zero_routes_require_owner_reduction(self):
        self.addCleanup(setattr, settings, "_values", None)
        for name in ("zero_scaled_pack", "zero_norm_overlap"):
            with self.subTest(name=name), patch.object(
                settings,
                "_installed_groups",
                return_value={"openwam.training": {name: True}},
            ):
                with self.assertRaisesRegex(ValueError, "require zero_reduce_scatter"):
                    settings.initialize(force=True)


if __name__ == "__main__":
    unittest.main()
