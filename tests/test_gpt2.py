import unittest

import torch

from inference_lab.gpt2 import GPT, GPTConfig


def tiny_model() -> GPT:
    torch.manual_seed(0)
    config = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=8,
    )
    return GPT(config).eval()


class GPTTest(unittest.TestCase):
    def test_forward_shape(self) -> None:
        model = tiny_model()
        token_ids = torch.randint(0, model.config.vocab_size, (2, 4))

        logits = model(token_ids)

        self.assertEqual(logits.shape, (2, 4, model.config.vocab_size))

    def test_attention_is_causal(self) -> None:
        model = tiny_model()
        first = torch.tensor([[1, 2, 3, 4]])
        changed_future = torch.tensor([[1, 2, 3, 9]])

        with torch.inference_mode():
            first_logits = model(first)
            changed_logits = model(changed_future)

        # Changing token 3 cannot affect outputs at positions 0, 1, or 2.
        torch.testing.assert_close(first_logits[:, :3], changed_logits[:, :3])

    def test_input_and_output_embeddings_share_weights(self) -> None:
        model = tiny_model()

        self.assertIs(model.transformer.wte.weight, model.lm_head.weight)


if __name__ == "__main__":
    unittest.main()
