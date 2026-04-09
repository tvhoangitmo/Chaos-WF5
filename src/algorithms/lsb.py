#!/usr/bin/env python3
# coding: utf-8

import argparse, os, hashlib
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# ----------------------------- Bit helpers ---------------------------------
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

# -------------------------- Chaotic encryption -----------------------------
def logistic_keystream(n: int, mu: float, x0: float, warmup: int = 1000) -> np.ndarray:
    if not (0 < x0 < 1):
        raise ValueError("x0 must be in (0,1)")
    if not (3.57 < mu < 4.0):
        raise ValueError("mu should be in (3.57, 4.0) for chaos; try 3.99")

    x = float(x0)
    for _ in range(warmup):
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

# ----------------------------- Payload -------------------------------------
def build_payload_bits_from_ct(ct: bytes) -> np.ndarray:
    # Header 32-bit = ciphertext length in BYTES
    return np.concatenate([u32_to_bits(len(ct)), bytes_to_bits(ct)])

def parse_payload_bits(bits: np.ndarray, mu: float, x0: float, warmup: int = 1000) -> bytes:
    ct_len = bits_to_u32(bits[:32])
    ct_bits = bits[32: 32 + ct_len * 8]
    ct = bits_to_bytes(ct_bits)
    return chaotic_xor(ct, mu, x0, warmup)

# ----------------------------- LSB Embed/Extract ---------------------------
def max_capacity_bits_lsb(img: np.ndarray) -> int:
    # 1 bit per image byte (LSB of each byte)
    return int(img.size)

def embed_lsb(img: np.ndarray, payload_bits: np.ndarray) -> np.ndarray:
    flat = img.reshape(-1).copy()
    n = min(payload_bits.size, flat.size)
    flat[:n] = (flat[:n] & 0xFE) | payload_bits[:n].astype(np.uint8)
    return flat.reshape(img.shape), n

def extract_lsb_bits(img: np.ndarray, nbits: int) -> np.ndarray:
    flat = img.reshape(-1)
    n = min(nbits, flat.size)
    return (flat[:n] & 1).astype(np.uint8)

# ----------------------------- Metrics -------------------------------------
def compute_metrics(img1: np.ndarray, img2: np.ndarray):
    if img1 is None or img2 is None:
        raise ValueError("Failed to read image(s). Check paths.")
    if img1.shape != img2.shape:
        raise ValueError(f"images must have same shape, got {img1.shape} vs {img2.shape}")

    psnr_val = cv2.PSNR(img1, img2)
    try:
        ssim_val = ssim(img1, img2, channel_axis=2, data_range=255)
    except TypeError:
        ssim_val = ssim(img1, img2, multichannel=True, data_range=255)
    return psnr_val, ssim_val

def file_sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ----------------------------- CLI -----------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LSB + Chaotic XOR (logistic)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("embed", help="Chaotic-encrypt then embed via LSB")
    pe.add_argument("--in", dest="inp", required=True)
    pe.add_argument("--out", dest="out", required=True)
    pe.add_argument("--msgfile", required=True, help="path to secret file (read as bytes)")
    pe.add_argument("--mu", type=float, required=True, help="logistic map mu (e.g., 3.99)")
    pe.add_argument("--x0", type=float, required=True, help="initial x0 in (0,1)")
    pe.add_argument("--warmup", type=int, default=1000)

    px = sub.add_parser("extract", help="Extract via LSB then chaotic-decrypt")
    px.add_argument("--in", dest="inp", required=True)
    px.add_argument("--out", dest="out", required=True)
    px.add_argument("--mu", type=float, required=True)
    px.add_argument("--x0", type=float, required=True)
    px.add_argument("--warmup", type=int, default=1000)
    px.add_argument("--verify", help="optional: original msg file for SHA-256 check")

    args = ap.parse_args()

    if args.cmd == "embed":
        cover = cv2.imread(args.inp)
        if cover is None:
            raise FileNotFoundError(f"Cover image not found or unreadable: {args.inp}")

        with open(args.msgfile, "rb") as f:
            pt = f.read()

        print(f"Plaintext length: {len(pt)} bytes = {len(pt)*8} bits")

        # Encrypt
        ct_full = chaotic_xor(pt, args.mu, args.x0, args.warmup)

        cap_bits = max_capacity_bits_lsb(cover)
        total_container_bits = int(cover.size * 8)

        # Ensure decodable truncation:
        # Need 32 header bits + 8*k ciphertext bits
        if cap_bits < 32:
            raise RuntimeError(f"Image too small: capacity={cap_bits} bits, need at least 32 bits for header.")

        max_ct_bytes = max(0, (cap_bits - 32) // 8)
        truncated = len(ct_full) > max_ct_bytes
        ct = ct_full[:max_ct_bytes] if truncated else ct_full

        payload_bits = build_payload_bits_from_ct(ct)
        total_payload_bits = int(payload_bits.size)

        # Embed
        stego_img, nbits = embed_lsb(cover, payload_bits)
        ok = cv2.imwrite(args.out, stego_img)
        if not ok:
            raise RuntimeError(f"Failed to write output image: {args.out}")

        # Completion vs ORIGINAL (không truncate) payload lý thuyết
        payload_bits_full = build_payload_bits_from_ct(ct_full)
        total_payload_bits_full = int(payload_bits_full.size)

        completion_ratio = (nbits / total_payload_bits_full) if total_payload_bits_full else 1.0
        completion_flag = 1 if (not truncated and nbits == total_payload_bits_full) else 0

        print(f"Max capacity (LSB): {cap_bits} bits")
        print(f"Total container bits: {total_container_bits} bits")
        print(f"Embedded: {nbits} bits")

        if truncated:
            lost_bytes = len(ct_full) - len(ct)
            print(f"WARNING: payload > capacity, ciphertext truncated by {lost_bytes} bytes.")
            print(f"Kept ciphertext: {len(ct)} bytes -> header updated to remain decodable.")

        print(f"Embed completion ratio: {completion_ratio:.6f}")
        print(f"Embed completed (1=full, 0=partial): {completion_flag}")

        stego = cv2.imread(args.out)
        psnr, ss = compute_metrics(cover, stego)

        ER_img = nbits / total_container_bits if total_container_bits else 0.0
        ER_cap = nbits / cap_bits if cap_bits else 0.0

        print(f"ER_img (S / total image bits): {ER_img:.8f}")
        print(f"ER_cap (S / LSB capacity):     {ER_cap:.6f}")
        print(f"PSNR={psnr:.2f} SSIM={ss:.4f}")

    else:
        stego = cv2.imread(args.inp)
        if stego is None:
            raise FileNotFoundError(f"Stego image not found or unreadable: {args.inp}")

        # Always extract header first (32 bits), then extract exact ciphertext length
        header_bits = extract_lsb_bits(stego, 32)
        ct_len = bits_to_u32(header_bits)

        needed_bits = 32 + ct_len * 8
        bits = extract_lsb_bits(stego, needed_bits)

        print(f"Ciphertext length from header: {ct_len} bytes = {ct_len*8} bits")

        pt = parse_payload_bits(bits, args.mu, args.x0, args.warmup)

        with open(args.out, "wb") as f:
            f.write(pt)

        print(f"Extracted plaintext: {len(pt)} bytes -> {args.out}")

        if args.verify and os.path.isfile(args.verify):
            h1 = file_sha256(args.verify)
            h2 = file_sha256(args.out)
            ok = (h1 == h2)
            print("VERIFY:", "OK" if ok else "MISMATCH")
            print("SHA256 orig:", h1)
            print("SHA256 recv:", h2)

if __name__ == "__main__":
    main()
