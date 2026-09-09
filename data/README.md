# Tiny Shakespeare

The experiment uses the character-level text distributed with Andrej Karpathy's
char-rnn project. Download it with `python prepare_data.py`.

- [Pinned text](https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt)
- Upstream commit: `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`
- SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
- Size: 1,115,394 bytes, with 65 distinct characters.

Use the first 90% of characters for training and the final 10% for validation.
Training windows stay inside the training split. Benchmark continuations come
from the validation split. The vocabulary comes from the complete text, as in
the lecture, but no validation targets participate in gradient updates.

The downloaded text is ignored by Git. The original Shakespeare works are public
domain. This project does not claim ownership of the dataset or its compilation.
