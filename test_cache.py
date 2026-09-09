"""Check the trained checkpoint if present, otherwise a small random model."""
import hashlib
import json
from pathlib import Path
import unittest

import torch

from model import Config, TinyGPT, cache_bytes, load_checkpoint


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.path = Path("checkpoints/tiny_gpt.pt")
        if cls.path.exists():
            cls.model, _ = load_checkpoint(cls.path)
        else:
            cls.model = TinyGPT(Config(width=32)).eval()
        cls.max_error = 0.0

    @torch.inference_mode()
    def test_all_positions_and_chunked_cache(self):
        model = self.model
        tokens = torch.randint(model.config.vocab_size, (2, 128))
        full, _ = model(tokens)
        for chunk in (1, 7, 16):
            cache, pieces = None, []
            for start in range(0, 128, chunk):
                logits, cache = model(tokens[:, start:start + chunk], cache, True)
                pieces.append(logits)
                expected_length = min(start + chunk, 128)
                for layer in cache:
                    for k, v in layer:
                        self.assertEqual(k.size(1), expected_length)
                        self.assertEqual(k.shape, v.shape)
            incremental = torch.cat(pieces, dim=1)
            type(self).max_error = max(type(self).max_error, (full - incremental).abs().max().item())
            torch.testing.assert_close(full, incremental, atol=1e-5, rtol=1e-4)

    @torch.inference_mode()
    def test_greedy_generation(self):
        for prompt_length in (1, 16, 64):
            prompt = torch.randint(self.model.config.vocab_size, (1, prompt_length))
            plain = self.model.generate(prompt, 32, use_cache=False)
            cached = self.model.generate(prompt, 32, use_cache=True)
            self.assertTrue(torch.equal(plain, cached))

    @torch.inference_mode()
    def test_cache_bytes(self):
        config = self.model.config
        _, cache = self.model(torch.zeros((2, 128), dtype=torch.long), use_cache=True)
        self.assertEqual(cache_bytes(cache), 2 * config.layers * 2 * 128 * config.width * 4)

    @torch.inference_mode()
    def test_causality_and_new_requests(self):
        tokens = torch.randint(self.model.config.vocab_size, (1, 32))
        before, cache = self.model(tokens, use_cache=True)
        changed = tokens.clone(); changed[:, 16:] = (changed[:, 16:] + 1) % self.model.config.vocab_size
        after, _ = self.model(changed)
        torch.testing.assert_close(before[:, :16], after[:, :16])
        self.model(tokens[:, :1], cache, True)
        fresh, _ = self.model(tokens, use_cache=True)
        torch.testing.assert_close(before, fresh)
        self.assertEqual(cache[0][0][0].size(1), 32)  # Appending does not mutate the input cache.

    @torch.inference_mode()
    def test_window_boundaries(self):
        tokens = torch.zeros((1, 128), dtype=torch.long)
        _, cache = self.model(tokens, use_cache=True)
        with self.assertRaises(ValueError):
            self.model(tokens[:, :1], cache, True)
        with self.assertRaises(ValueError):
            self.model.generate(tokens, 1)
        with self.assertRaises(ValueError):
            self.model(tokens[:, :0])
        self.assertTrue(torch.equal(self.model.generate(tokens, 0), tokens))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CacheTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        path = CacheTests.path
        report = {"tests_passed": result.testsRun, "max_abs_logit_error": CacheTests.max_error,
                  "atol": 1e-5, "rtol": 1e-4,
                  "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
        Path("results").mkdir(exist_ok=True)
        Path("results/correctness.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)

