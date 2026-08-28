import argparse
import os
import numpy as np

import cv2
from skimage.metrics import structural_similarity as ssim

# Image I/O (robust) 
def read_image_bgr(path: str) -> np.ndarray:

    # Try Pillow first (often avoids PNG gamma/profile surprises), fallback to OpenCV.
    # Always returns uint8 BGR (H,W,3).
    try:
        from PIL import Image 
        im = Image.open(path).convert("RGB")
        rgb = np.array(im, dtype=np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        return bgr
    except Exception:
        # cv2.imread breaks on non-ASCII paths on Windows; imdecode + binary read is safe.
        with open(path, "rb") as f:
            raw = f.read()
        if not raw:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")
        return img

def write_image_bgr(path: str, bgr: np.ndarray) -> None:
    # Try Pillow first; fallback imencode + binary write (Unicode-safe on Windows).
    bgr = np.ascontiguousarray(bgr)
    try:
        from PIL import Image  # type: ignore
        rgb = bgr[:, :, ::-1]
        im = Image.fromarray(rgb, mode="RGB")
        im.save(path)
    except Exception:
        ext = os.path.splitext(path)[1].lower() or ".png"
        if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            ext = ".png"
        ok, buf = cv2.imencode(ext, bgr)
        if not ok:
            raise RuntimeError(f"Failed to write image: {path}")
        with open(path, "wb") as f:
            f.write(buf.tobytes())

# Bit helpers
def bytes_to_bits(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = ((arr[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.uint8)
    return bits.reshape(-1)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.reshape(-1)
    bits = bits[: (bits.size // 8) * 8]
    return np.packbits(bits, bitorder="big").tobytes()

def u32_to_bits(n: int) -> np.ndarray:
    if not (0 <= n <= 0xFFFFFFFF):
        raise ValueError("length must fit in 32 bits")
    return np.array([(n >> i) & 1 for i in range(31, -1, -1)], dtype=np.uint8)

def bits_to_u32(bits: np.ndarray) -> int:
    bits = bits[:32]
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val

# Chaotic encryption
def logistic_keystream(n: int, mu: float, x0: float, warmup: int = 1000) -> np.ndarray:
    if not (0.0 < x0 < 1.0):
        raise ValueError("x0 must be in (0,1)")
    if not (3.57 < mu < 4.0):
        raise ValueError("mu should be in (3.57, 4.0) for chaos; e.g. 3.99")

    x = float(x0)
    for _ in range(int(warmup)):
        x = mu * x * (1.0 - x)

    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        x = mu * x * (1.0 - x)
        out[i] = int(x * 256.0) & 0xFF
    return out

def chaotic_xor(data: bytes, mu: float, x0: float, warmup: int = 1000) -> bytes:
    if not data:
        return b""
    ks = logistic_keystream(len(data), mu, x0, warmup)
    arr = np.frombuffer(data, dtype=np.uint8)
    return (arr ^ ks).tobytes()

# WF5 matrices 
Hw = np.array(
    [
        [1, 0, 0, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)
HwT = Hw.T

def bits4_to_int(b4: np.ndarray) -> int:
    v = 0
    for x in b4[:4]:
        v = (v << 1) | int(x & 1)
    return v

# Precompute all 512 error vectors and syndrome groups
E9 = ((np.arange(512)[:, None] >> np.arange(8, -1, -1)) & 1).astype(np.uint8)  # (512,9)
SYN4 = (E9 @ HwT) & 1  # (512,4)

CAND_BY_SYN = {s: [] for s in range(16)}
for idx in range(512):
    CAND_BY_SYN[bits4_to_int(SYN4[idx])].append(idx)
for s in range(16):
    CAND_BY_SYN[s] = np.array(CAND_BY_SYN[s], dtype=np.int32)

# Sobel weight mask
def compute_weight_mask_bytes(img_bgr: np.ndarray, codeword_size: int = 9, threshold: float = 30.0) -> np.ndarray:
    # w=1 if G<T, w=2 if G>=T
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    w_pix = np.where(mag.ravel() < threshold, 2, 1).astype(np.uint8)
    w_bytes = np.repeat(w_pix, 3)

    n_groups = w_bytes.size // codeword_size
    if n_groups <= 0:
        return np.empty((0, codeword_size), dtype=np.uint8)
    return w_bytes[: n_groups * codeword_size].reshape(n_groups, codeword_size)

def pick_best_e(weights9: np.ndarray, syn_target_int: int) -> np.ndarray:
    idxs = CAND_BY_SYN[int(syn_target_int)]
    cand = E9[idxs]  # (m,9)
    costs = cand.astype(np.uint16) @ weights9.astype(np.uint16)
    return cand[np.argmin(costs)].astype(np.uint8)

# Payload
def build_payload_bits_from_ciphertext(ct: bytes) -> np.ndarray:
    return np.concatenate([u32_to_bits(len(ct)), bytes_to_bits(ct)])

def extract_ciphertext_only(bits: np.ndarray) -> bytes:
    # Extract ciphertext bytes from bitstream (no decrypt).
    if bits.size < 32:
        raise ValueError("Not enough bits for header")
    ct_len = bits_to_u32(bits[:32])
    need_bits = 32 + ct_len * 8
    if bits.size < need_bits:
        # take full bytes available
        avail_ct_bits = max(0, bits.size - 32)
        ct_len = avail_ct_bits // 8
        need_bits = 32 + ct_len * 8
    return bits_to_bytes(bits[32:need_bits])

def parse_payload_bits(bits: np.ndarray, mu: float, x0: float, warmup: int = 1000) -> bytes:
    ct = extract_ciphertext_only(bits)
    return chaotic_xor(ct, mu, x0, warmup)

# Capacity
def max_capacity_bits(img: np.ndarray) -> int:
    return (img.size // 9) * 4

# Embed/Extract
def embed_wf5(cover_bgr: np.ndarray, payload_bits: np.ndarray, threshold: float = 30.0) -> tuple[np.ndarray, int]:
    flat = cover_bgr.ravel()
    n_groups = flat.size // 9
    if n_groups <= 0:
        raise ValueError("Image too small for WF5 (no 9-byte groups).")

    groups = flat[: n_groups * 9].reshape(n_groups, 9)
    weights = compute_weight_mask_bytes(cover_bgr, 9, threshold)
    if weights.shape[0] != n_groups:
        m = min(weights.shape[0], n_groups)
        groups = groups[:m]
        weights = weights[:m]
        n_groups = m

    out = flat.copy()
    idx = 0

    for i in range(n_groups):
        remain = int(min(4, payload_bits.size - idx))
        if remain <= 0:
            break

        group = groups[i]
        B = (group & 1).astype(np.uint8)
        S = (B @ HwT) & 1

        I = np.zeros(4, dtype=np.uint8)
        I[:remain] = payload_bits[idx: idx + remain]

        A = (S ^ I).astype(np.uint8)
        e = pick_best_e(weights[i], bits4_to_int(A))

        B2 = B ^ e
        out[i * 9: (i + 1) * 9] = (group & 0xFE) | B2
        idx += remain

    stego = out.reshape(cover_bgr.shape)
    return stego, idx

def extract_wf5_bits(stego_bgr: np.ndarray) -> np.ndarray:
    flat = stego_bgr.ravel()
    n_groups = flat.size // 9
    if n_groups <= 0:
        return np.empty((0,), dtype=np.uint8)

    groups = flat[: n_groups * 9].reshape(n_groups, 9)
    bits_out = np.empty((n_groups * 4,), dtype=np.uint8)

    k = 0
    for g in groups:
        B = (g & 1).astype(np.uint8)
        S = (B @ HwT) & 1
        bits_out[k: k + 4] = S
        k += 4

    return bits_out

# Metrics
def compute_metrics(img1: np.ndarray, img2: np.ndarray):
    psnr_val = cv2.PSNR(img1, img2)
    try:
        ssim_val = ssim(img1, img2, channel_axis=2, data_range=255)
    except TypeError:
        ssim_val = ssim(img1, img2, multichannel=True, data_range=255)
    return float(psnr_val), float(ssim_val)

def first_diff(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n

# CLI
def main():
    ap = argparse.ArgumentParser(description="WF5 + Sobel weights + Chaotic XOR (Logistic)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("embed", help="Encrypt then embed")
    pe.add_argument("--in", dest="inp", required=True)
    pe.add_argument("--out", dest="out", required=True)
    pe.add_argument("--msgfile", required=True)
    pe.add_argument("--mu", type=float, required=True)
    pe.add_argument("--x0", type=float, required=True)
    pe.add_argument("--warmup", type=int, default=1000)
    pe.add_argument("--threshold", type=float, default=30.0)

    px = sub.add_parser("extract", help="Extract then decrypt")
    px.add_argument("--in", dest="inp", required=True)
    px.add_argument("--out", dest="out", required=True)
    px.add_argument("--mu", type=float, required=True)
    px.add_argument("--x0", type=float, required=True)
    px.add_argument("--warmup", type=int, default=1000)
    px.add_argument("--verify", help="optional original msgfile")

    args = ap.parse_args()

    if args.cmd == "embed":
        cover = read_image_bgr(args.inp)
        with open(args.msgfile, "rb") as f:
            pt = f.read()

        ct_full = chaotic_xor(pt, args.mu, args.x0, args.warmup)

        cap_bits = int(max_capacity_bits(cover))
        total_img_bits = int(cover.size * 8)
        if cap_bits < 32:
            raise ValueError("Image too small: capacity < 32 bits")

        max_ct_bytes = max(0, (cap_bits - 32) // 8)
        need_bits_full = 32 + len(ct_full) * 8
        too_large = need_bits_full > cap_bits

        ct_use = ct_full if not too_large else ct_full[:max_ct_bytes]
        payload_bits = build_payload_bits_from_ciphertext(ct_use)

        stego, nbits = embed_wf5(cover, payload_bits, args.threshold)
        if nbits != payload_bits.size:
            raise RuntimeError(f"Embed short: embedded={nbits}, expected={payload_bits.size}")

        write_image_bgr(args.out, stego)

        # Metrics
        stego2 = read_image_bgr(args.out)
        psnr_val, ssim_val = compute_metrics(cover, stego2)

        ER_img = nbits / total_img_bits if total_img_bits else 0.0
        ER_cap = nbits / cap_bits if cap_bits else 0.0
        coverage = (len(ct_use) / len(ct_full)) if len(ct_full) else 1.0

        print(f"Plaintext:  {len(pt)} bytes")
        print(f"Ciphertext: {len(ct_full)} bytes")
        print(f"WF5 cap:    {cap_bits} bits (~{max_ct_bytes} ct-bytes + 4B header)")
        print(f"Embedded:   {nbits} bits (payload bytes={4 + len(ct_use)})")
        print(f"Coverage:   {coverage*100:.2f}% (embedded {len(ct_use)}/{len(ct_full)} bytes)")
        print(f"ER_img:     {ER_img:.8f} (S / total image bits)")
        print(f"ER_cap:     {ER_cap:.6f} (S / WF5 capacity)")
        print(f"PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}")
        if too_large:
            print("WARNING: message > capacity -> embedded prefix only (header adjusted).")

        # SELF-CHECK (this is the key)
        bits_back = extract_wf5_bits(stego2)
        ct_back = extract_ciphertext_only(bits_back)

        if ct_back != ct_use:
            di = first_diff(ct_back, ct_use)
            print("SELF-CHECK: FAIL (ciphertext extracted from saved stego != ciphertext embedded)")
            if di >= 0:
                print(f"First diff at byte {di}: embedded={ct_use[di]:02x} extracted={ct_back[di]:02x}")
            print("=> This means some LSBs changed between write/read of the image file.")
            print("   Fix: try output .bmp/.tiff, or keep using Pillow I/O as in this script.")
        else:
            pt_back = chaotic_xor(ct_back, args.mu, args.x0, args.warmup)
            if pt_back == pt[:len(pt_back)]:
                print("SELF-CHECK: OK (file saved/read back correctly)")
            else:
                di = first_diff(pt_back, pt)
                print("SELF-CHECK: FAIL after decrypt (keystream/key mismatch?)")
                if di >= 0:
                    print(f"First diff at byte {di}: orig={pt[di]:02x} got={pt_back[di]:02x}")

    else:
        stego = read_image_bgr(args.inp)
        bits = extract_wf5_bits(stego)
        if bits.size < 32:
            raise ValueError("Not enough bits for header")

        ct_len_hdr = bits_to_u32(bits[:32])
        print(f"Header ciphertext length: {ct_len_hdr} bytes")

        pt = parse_payload_bits(bits, args.mu, args.x0, args.warmup)
        with open(args.out, "wb") as f:
            f.write(pt)
        print(f"Recovered plaintext: {len(pt)} bytes -> {args.out}")

        if args.verify and os.path.isfile(args.verify):
            with open(args.verify, "rb") as f:
                orig = f.read()

            if pt == orig:
                print("VERIFY: OK_FULL (full match)")
            elif len(pt) <= len(orig) and pt == orig[:len(pt)]:
                print("VERIFY: OK_PREFIX (correct prefix; message > capacity)")
                print(f"Prefix length: {len(pt)}/{len(orig)} bytes = {len(pt)/len(orig)*100:.2f}%")
            else:
                print("VERIFY: MISMATCH")
                di = first_diff(pt, orig)
                if di >= 0 and di < min(len(pt), len(orig)):
                    print(f"First diff at byte {di}: orig={orig[di]:02x} got={pt[di]:02x}")

if __name__ == "__main__":
    main()