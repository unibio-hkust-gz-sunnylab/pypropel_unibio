from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pypropel import esm


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTokens:
    def __init__(self, sequences):
        self.sequences = sequences
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self


class FakeArray:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def __getitem__(self, item):
        return FakeArray(self.values[item])

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeModel:
    num_layers = 2

    def __init__(self, name="fake"):
        self.name = name
        self.device = "cpu"
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, batch_tokens, repr_layers):
        layer = repr_layers[0]
        dim = 3
        max_len = max(len(seq) for seq in batch_tokens.sequences) + 2
        values = np.zeros((len(batch_tokens.sequences), max_len, dim), dtype=np.float32)
        for batch_idx, sequence in enumerate(batch_tokens.sequences):
            for residue_idx, symbol in enumerate(sequence, start=1):
                values[batch_idx, residue_idx, :] = [
                    float(layer),
                    float(residue_idx),
                    float(ord(symbol)),
                ]
        return {"representations": {layer: FakeArray(values)}}


class FakeAlphabet:
    def get_batch_converter(self):
        def convert(data):
            sequences = [sequence for _, sequence in data]
            return None, None, FakeTokens(sequences)

        return convert


class FakePretrained:
    def __init__(self):
        self.loads = []

    def load_model_and_alphabet(self, model_name):
        self.loads.append(model_name)
        return FakeModel(model_name), FakeAlphabet()


class ESMTest(unittest.TestCase):
    def setUp(self):
        esm.clear_model_cache()

    def tearDown(self):
        esm.clear_model_cache()

    def test_is_available_false_when_optional_import_fails(self):
        with mock.patch("importlib.import_module", side_effect=ImportError("missing")):
            self.assertFalse(esm.is_available())

    def test_load_model_cache_is_keyed_by_model_and_device(self):
        with self.fake_dependencies() as fake_pretrained:
            model_a, _, _ = esm.load_model("esm2_a", device="cpu")
            model_b, _, _ = esm.load_model("esm2_b", device="cpu")
            model_a_again, _, _ = esm.load_model("esm2_a", device="cpu")

        self.assertIs(model_a, model_a_again)
        self.assertIsNot(model_a, model_b)
        self.assertEqual(fake_pretrained.loads, ["esm2_a", "esm2_b"])
        self.assertTrue(model_a.eval_called)

    def test_embed_sequence_uses_supplied_model_without_loading_default(self):
        with self.fake_dependencies() as fake_pretrained:
            model = FakeModel()
            converter = FakeAlphabet().get_batch_converter()
            embedding = esm.embed_sequence(
                "ACD",
                model=model,
                batch_converter=converter,
                layer=-1,
            )

        self.assertEqual(fake_pretrained.loads, [])
        self.assertEqual(embedding.shape, (3, 3))
        self.assertTrue(np.all(embedding[:, 0] == 2.0))

    def test_embed_batch_returns_one_array_per_sequence(self):
        with self.fake_dependencies():
            embeddings = esm.embed_batch(["AC", "WXYZ"], model_name="esm2_a", batch_size=1)

        self.assertEqual([arr.shape for arr in embeddings], [(2, 3), (4, 3)])
        self.assertEqual(float(embeddings[1][0, 2]), float(ord("W")))

    def test_validation_rejects_bad_sequence_and_alignment(self):
        with self.assertRaises(ValueError):
            esm.normalize_sequence("ACD-")
        with self.assertRaises(ValueError):
            esm.validate_embedding_alignment(np.zeros((2, 3)), "ACD")

    def test_save_and_load_embeddings_with_metadata(self):
        embedding = np.ones((2, 3), dtype=np.float32)
        metadata = esm.build_metadata("AC", embedding, model_name="esm2_a", layer=-1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "embedding.npz"
            esm.save_embeddings(embedding, path, metadata=metadata)
            loaded, loaded_metadata = esm.load_embeddings(path, include_metadata=True)

        self.assertTrue(np.array_equal(loaded["embeddings"], embedding))
        self.assertEqual(loaded_metadata["model_name"], "esm2_a")
        self.assertEqual(loaded_metadata["sequence_length"], 2)

    def fake_dependencies(self):
        return FakeDependencyContext()


class FakeDependencyContext:
    def __enter__(self):
        self.old_torch = sys.modules.get("torch")
        self.old_esm = sys.modules.get("esm")
        self.fake_pretrained = FakePretrained()
        fake_torch = types.SimpleNamespace(no_grad=lambda: FakeNoGrad())
        fake_esm = types.SimpleNamespace(pretrained=self.fake_pretrained)
        sys.modules["torch"] = fake_torch
        sys.modules["esm"] = fake_esm
        importlib.invalidate_caches()
        return self.fake_pretrained

    def __exit__(self, exc_type, exc, tb):
        if self.old_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self.old_torch
        if self.old_esm is None:
            sys.modules.pop("esm", None)
        else:
            sys.modules["esm"] = self.old_esm
        importlib.invalidate_caches()
        return False


if __name__ == "__main__":
    unittest.main()
