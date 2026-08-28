# WF5 steganography with logistic-map encryption

The **main contribution of this project is WF5** (`wf5.py`): weighted spatial embedding using **9-byte LSB groups**, a linear **Hamming-style** matrix `Hw` (4 payload bits per group via syndrome minimization), and **Sobel-gradient weights** so distortion is steered toward less sensitive regions. Payloads are XOR-encrypted with a **logistic map** keystream (`μ`, `x0`, optional `warmup`), with a shared wire format: a **32-bit header** (ciphertext length in bytes) plus **ciphertext**.

**LSB**, **DCT**, and **F5** are included **primarily as comparison baselines** under the same chaotic payload format, so you can benchmark capacity, image quality (PSNR/SSIM), and blind-detection scores (`stego_scores.py`) on equal footing.

A **desktop GUI** (`stego_app.py`) wraps all four embedders and prints quality + steganalysis scores after hiding a message. Extract is still done via the CLI scripts.

## Repository layout

```
Code/
├── README.md
├── requirements.txt
├── .gitignore
├── stego_app.py                  # GUI: embed + PSNR/SSIM/ER + RS/Chi²/SP
├── docs/                         # WF5 notes and diagrams (e.g. wf5.txt, drawio)
└── src/
    ├── algorithms/
    │   ├── wf5.py                # Primary method: weighted F5 + chaos (CLI)
    │   ├── lsb.py                # Baseline: spatial LSB + chaos
    │   ├── dct.py                # Baseline: DCT (8×8, Y channel) + chaos
    │   ├── f5.py                 # Baseline: F5-like DCT + JPEG-like quant + chaos + --passphrase
    │   └── stego_scores.py       # Blind steganalysis (RS, Chi-square, Sample Pair, LSB entropy)
    ├── input/                    # Example secrets (test1.txt, test2.txt, test4.txt)
    ├── cover_images/             # Sample covers (Barbara, ColorImage, Document, Fruit, Lenna, Logo, Mountain, VanGogh)
    ├── stego_image/              # All stego outputs (by method)
    │   ├── lsb/
    │   ├── dct/
    │   ├── wf5/
    │   └── f5/
    ├── secret.txt                # Typical secret file for CLI demos
    └── …                         # Optional local artifacts (recovered.*, payload_bits.txt, …)
```

Run all CLI examples and the GUI from the **repository root** so paths resolve correctly.

## Requirements

- Python 3.10+ (3.11 recommended)
- Tkinter (standard library) for the GUI — on some Linux installs: `sudo apt install python3-tk`

```bash
pip install -r requirements.txt
```

## Graphical interface — `stego_app.py`

Embed only: pick algorithm (WF5 / LSB / DCT / F5), cover image, secret file, output folder, and stego filename. Logistic parameters (`μ`, `x0`, `warmup`) are shared; extra fields apply per method (WF5 `threshold`, DCT `τ` / `pos_u` / `pos_v`, F5 passphrase / quality / `k`).

```bash
python stego_app.py
```

After embedding, the window reports:

- **PSNR / SSIM** vs. the cover
- **ER_img / ER_cap** (and F5 **ER_message** if applicable)
- **RS, Chi-square, Sample Pair, LSB entropy**, weighted `avg_score`, and a CLEAN / LOW / MODERATE / HIGH verdict

If the secret exceeds capacity, the payload is truncated and a warning is shown. To recover the message, use the matching **extract** CLI below with the same chaos (and F5) parameters.

## Command-line usage

Chaos parameters: `--mu` (e.g. `3.99`), `--x0` in `(0, 1)`, `--warmup` (default `1000`). Adjust `--in`, `--out`, and `--msgfile` as needed.

### WF5 (primary) — `src/algorithms/wf5.py`

```bash
python src/algorithms/wf5.py embed  --in src/cover_images/Lenna.png --out src/stego_image/wf5/Lenna_stego.png --msgfile src/secret.txt --mu 3.99 --x0 0.6 --threshold 30.0
python src/algorithms/wf5.py extract --in src/stego_image/wf5/Lenna_stego.png --out src/recovered.bin --mu 3.99 --x0 0.6 --verify src/secret.txt
```

- **9-byte LSB groups** with matrix `Hw` and **Sobel-based weights** (`--threshold` controls the smooth vs. edge split).
- **Pillow-first** I/O to limit LSB drift after save; the script runs a self-check after embedding.

### Baseline methods (for comparison)

#### LSB — `src/algorithms/lsb.py`

```bash
python src/algorithms/lsb.py embed  --in src/cover_images/Lenna.png --out src/stego_image/lsb/Lenna_stego.png --msgfile src/secret.txt --mu 3.99 --x0 0.6
python src/algorithms/lsb.py extract --in src/stego_image/lsb/Lenna_stego.png --out src/recovered.bin --mu 3.99 --x0 0.6 --verify src/secret.txt
```

~1 bit per image byte (BGR); truncates ciphertext if the cover is too small.

#### DCT — `src/algorithms/dct.py`

```bash
python src/algorithms/dct.py embed  --in src/cover_images/Lenna.png --out src/stego_image/dct/Lenna_stego.png --msgfile src/secret.txt --mu 3.99 --x0 0.6 --tau 5.0 --pos_u 4 --pos_v 3
python src/algorithms/dct.py extract --in src/stego_image/dct/Lenna_stego.png --out src/recovered.bin --mu 3.99 --x0 0.6 --pos_u 4 --pos_v 3 --verify src/secret.txt
```

1 bit per 8×8 **Y** block; sign of coefficient `(pos_u, pos_v)` with floor `tau`.

#### F5 — `src/algorithms/f5.py`

F5-like embedding on the **Y** channel: OpenCV **DCT**, **JPEG-like quantization** (`--quality`, default 85), matrix encoding with **`k`** bits per **2^k − 1** nonzero AC coefficients (default **`k=3`**, 7 coefficients per group). Coefficient order is shuffled with a **`--passphrase`**-seeded permutation. **No `jpegio`**—covers and outputs can be **PNG or JPEG** (whatever OpenCV accepts).

From the repo root:

```bash
python src/algorithms/f5.py embed --in src/cover_images/Fruit.png --out src/stego_image/f5/Fruit_f5_4.png --msgfile src/input/test1.txt --passphrase mypass --mu 3.99 --x0 0.4132 --quality 85 --k 3
python src/algorithms/f5.py extract --in src/stego_image/f5/Fruit_f5_4.png --out src/recovered.bin --passphrase mypass --mu 3.99 --x0 0.4132 --quality 85 --k 3 --verify src/input/test1.txt
```

Use the **same** `--passphrase`, `--quality`, `--k`, `--mu`, `--x0`, and `--warmup` for extract as for embed.

### Blind steganalysis — `src/algorithms/stego_scores.py`

```bash
python src/algorithms/stego_scores.py src/stego_image/wf5/Lenna_wf5_1.png
```

Use the same tool on **WF5** and **baseline** stego images to compare RS, Chi-square, Sample Pair, and LSB entropy–based signals. Outputs a weighted composite and a verdict (CLEAN / LOW / MODERATE / HIGH). RS follows Fridrich, Goljan, Du (2001). The GUI runs this automatically after each embed.

## Metrics printed by the embedders

- **PSNR / SSIM**: visual quality vs. cover (useful when comparing WF5 to baselines).
- **ER_img / ER_cap**: embedded bits vs. total image bits vs. method capacity.
- **`--verify`**: optional SHA-256 check against the original secret file (CLI extract only).

## Main dependencies (`requirements.txt`)

Includes `numpy`, `opencv-python`, `scikit-image`, `scipy`, `tqdm`, `nltk`, `bitarray`, `matplotlib`, `Pillow`, `cryptography`.

The current F5 baseline does not use `cryptography` or `jpegio` (permutation is SHA-256 + NumPy). Some other listed packages may be unused by the embedders and kept for notebooks or extensions. The GUI uses **Tkinter** (stdlib) plus the same algorithm modules.

## Purpose

Educational and research use (WF5-focused steganography, with classical methods for comparison, a small GUI for embed + metrics, and blind steganalysis).
