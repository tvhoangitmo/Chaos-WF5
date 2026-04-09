#!/usr/bin/env python3
# coding: utf-8
import argparse, os, hashlib
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# ============================= Bit helpers ==============================
def bytes_to_bits(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = ((arr[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.uint8)
    return bits.reshape(-1)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.reshape(-1)
    bits = bits[: (bits.size // 8) * 8]
    return np.packbits(bits, bitorder="big").tobytes()

def u32_to_bits(n: int) -> np.ndarray:
    return np.array([(n >> i) & 1 for i in range(31, -1, -1)], dtype=np.uint8)

def bits_to_u32(bits: np.ndarray) -> int:
    bits = bits[:32]
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v

# ======================= Chaotic encryption (logistic) =====================
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

# ============================== Metrics ===================================
def compute_metrics(img1: np.ndarray, img2: np.ndarray):
    if img1 is None or img2 is None:
        raise ValueError("Failed to read image(s).")
    if img1.shape != img2.shape:
        raise ValueError(f"shape mismatch: {img1.shape} vs {img2.shape}")
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

# ============================= DCT helpers ================================
# Standard JPEG luminance quant table (Q50 baseline)
Q50 = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

# Zigzag order for 8x8 (index list of (u,v))
def zigzag_indices():
    idx = []
    for s in range(0, 15):
        if s % 2 == 0:
            # even: go down-left
            for u in range(s, -1, -1):
                v = s - u
                if u < 8 and v < 8:
                    idx.append((u, v))
        else:
            # odd: go up-right
            for v in range(s, -1, -1):
                u = s - v
                if u < 8 and v < 8:
                    idx.append((u, v))
    return idx

ZZ = zigzag_indices()
# AC positions in zigzag order exclude first (0,0)
ZZ_AC = ZZ[1:]  # 63 entries

def block_view(img_y: np.ndarray):
    """Return 8x8 blocks view: (H/8, W/8, 8, 8)"""
    h, w = img_y.shape
    assert h % 8 == 0 and w % 8 == 0
    return img_y.reshape(h//8, 8, w//8, 8).transpose(0, 2, 1, 3)

def idct2(a):
    return cv2.idct(a.astype(np.float32))

def dct2(a):
    return cv2.dct(a.astype(np.float32))

def jpeg_like_quantize(dct_block, quality=50):
    # simple quality scaling
    if quality < 1: quality = 1
    if quality > 100: quality = 100
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    Q = np.floor((Q50 * scale + 50) / 100).astype(np.float32)
    Q[Q < 1] = 1
    return np.round(dct_block / Q).astype(np.int16), Q

def jpeg_like_dequantize(qblock, Q):
    return (qblock.astype(np.float32) * Q).astype(np.float32)

# ============================== F5-like ===================================
# We do matrix encoding with (1, n=2^k-1): here k=3 -> n=7 embed 3 bits per 7 coeffs
# Parity-check matrix for Hamming(7,4): 3x7 where columns are 1..7 in binary
H7 = np.array([
    [1,0,1,0,1,0,1],  # bit0 of column index (LSB)
    [0,1,1,0,0,1,1],  # bit1
    [0,0,0,1,1,1,1],  # bit2
], dtype=np.uint8)

def syndrome7(lsb7: np.ndarray) -> np.ndarray:
    # s = H * x  (mod 2), x shape (7,)
    return (H7 @ (lsb7 & 1)) & 1  # shape (3,)

def bits3_to_int(b3: np.ndarray) -> int:
    return (int(b3[0]) << 2) | (int(b3[1]) << 1) | int(b3[2])

def int_to_bits3(v: int) -> np.ndarray:
    return np.array([(v >> 2) & 1, (v >> 1) & 1, v & 1], dtype=np.uint8)

def pick_flip_index_for_delta(delta3: np.ndarray) -> int:
    """
    Need find column j where H[:,j] == delta (mod2), then flip x_j.
    Columns represent binary 1..7:
      j=0 col=001, j=1 col=010, ... j=6 col=111 in (bit2..bit0) depending convention.
    Our H7 rows are (b0,b1,b2). So column j equals bits of (j+1).
    """
    target = delta3.reshape(3,)
    for j in range(7):
        col = H7[:, j]
        if np.all(col == target):
            return j
    return -1

def coeff_lsb(c: int) -> int:
    return abs(int(c)) & 1

def dec_abs_by_one(c: int) -> int:
    # F5 typically changes coefficient by +/-1 towards zero (to avoid too much distortion).
    if c > 0:
        return c - 1
    elif c < 0:
        return c + 1
    else:
        return 0

def build_payload_bits_trunc(pt: bytes, mu: float, x0: float, warmup: int, cap_bits: int):
    """
    Create ciphertext ct = chaotic_xor(pt).
    Payload: [u32(ct_len_bytes)] + ct_bits.
    If capacity not enough => truncate ct and update header accordingly.
    """
    ct_full = chaotic_xor(pt, mu, x0, warmup)
    need_ct_bytes = len(ct_full)
    if cap_bits < 32:
        use_ct_bytes = 0
    else:
        avail_ct_bits = cap_bits - 32
        use_ct_bytes = min(need_ct_bytes, avail_ct_bits // 8)
    ct_use = ct_full[:use_ct_bytes]
    payload_bits = np.concatenate([u32_to_bits(use_ct_bytes), bytes_to_bits(ct_use)])

    embedded_msg_bits = use_ct_bytes * 8
    total_msg_bits = need_ct_bytes * 8
    er_msg = (embedded_msg_bits / total_msg_bits) if total_msg_bits else 1.0
    complete_flag = 1 if use_ct_bytes == need_ct_bytes else 0

    stats = {
        "need_ct_bytes": need_ct_bytes,
        "use_ct_bytes": use_ct_bytes,
        "embedded_msg_bits": embedded_msg_bits,
        "total_msg_bits": total_msg_bits,
        "ER_message": er_msg,
        "complete_flag": complete_flag,
        "payload_bits_used": int(payload_bits.size),
    }
    return payload_bits, stats

def parse_payload_bits(bits: np.ndarray, mu: float, x0: float, warmup: int):
    ct_len_bytes = bits_to_u32(bits[:32])
    ct_bits = bits[32:32 + ct_len_bytes * 8]
    ct = bits_to_bytes(ct_bits)
    pt = chaotic_xor(ct, mu, x0, warmup)
    return pt, ct_len_bytes

def collect_ac_coeffs(qblocks: np.ndarray):
    """
    qblocks: (bh, bw, 8, 8) int16 quantized DCT blocks
    Return a list of references (block_y, block_x, u, v) for AC coeffs != 0.
    """
    bh, bw = qblocks.shape[:2]
    refs = []
    for by in range(bh):
        for bx in range(bw):
            blk = qblocks[by, bx]
            for (u, v) in ZZ_AC:
                c = int(blk[u, v])
                if c != 0:
                    refs.append((by, bx, u, v))
    return refs

def capacity_bits_f5_like(qblocks: np.ndarray, k=3):
    """
    F5 matrix encoding with k bits per group of (2^k - 1) coefficients.
    cap = floor(num_nonzero_AC / (2^k - 1)) * k
    """
    refs = collect_ac_coeffs(qblocks)
    n = (2**k - 1)
    groups = len(refs) // n
    return groups * k, len(refs)

def permute_indices(n: int, passphrase: str):
    # deterministic permutation using numpy RNG seeded by hash(passphrase)
    seed = int(hashlib.sha256(passphrase.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx

def embed_f5_like_dct(cover_bgr: np.ndarray, payload_bits: np.ndarray, passphrase: str,
                      quality=85, k=3):
    """
    Return stego_bgr, embedded_bits_count, capacity_bits, nonzero_ac_count
    """
    # work on luminance channel only (simplified)
    ycrcb = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)

    # pad to multiple of 8
    h, w = Y.shape
    hp = (h + 7) // 8 * 8
    wp = (w + 7) // 8 * 8
    Yp = np.zeros((hp, wp), dtype=np.float32)
    Yp[:h, :w] = Y
    Yp -= 128.0

    blocks = block_view(Yp)  # (bh, bw, 8, 8)
    bh, bw = blocks.shape[:2]

    # DCT + quantize
    qblocks = np.empty((bh, bw, 8, 8), dtype=np.int16)
    Qlast = None
    for by in range(bh):
        for bx in range(bw):
            d = dct2(blocks[by, bx])
            qb, Q = jpeg_like_quantize(d, quality=quality)
            qblocks[by, bx] = qb
            Qlast = Q

    cap_bits, nonzero_ac = capacity_bits_f5_like(qblocks, k=k)

    # truncate payload to capacity
    use_bits = payload_bits[:cap_bits].astype(np.uint8, copy=False)
    total_bits = int(use_bits.size)

    # permutation of coefficient references
    refs = collect_ac_coeffs(qblocks)
    perm = permute_indices(len(refs), passphrase)
    refs = [refs[i] for i in perm]

    n = (2**k - 1)
    idx_bit = 0
    idx_ref = 0
    embedded = 0

    while idx_bit < total_bits and (idx_ref + n) <= len(refs):
        group = refs[idx_ref: idx_ref + n]

        # get current LSBs of |coeff|
        x = np.array([coeff_lsb(qblocks[by, bx, u, v]) for (by, bx, u, v) in group], dtype=np.uint8)
        s = syndrome7(x)  # 3 bits

        remain = min(k, total_bits - idx_bit)
        msg_bits = np.zeros(k, dtype=np.uint8)
        msg_bits[:remain] = use_bits[idx_bit: idx_bit + remain]

        # target syndrome = msg_bits (k=3)
        delta = s ^ msg_bits
        if np.all(delta == 0):
            # no change needed
            pass
        else:
            j = pick_flip_index_for_delta(delta)
            if j >= 0:
                by, bx, u, v = group[j]
                c = int(qblocks[by, bx, u, v])
                c2 = dec_abs_by_one(c)
                # shrinkage: if becomes 0, we "lose" coefficient; simplest handling: accept it (reduces future capacity)
                qblocks[by, bx, u, v] = np.int16(c2)

        idx_bit += remain
        embedded += remain
        idx_ref += n

    # dequant + IDCT -> rebuild Y
    Yrec = np.empty((hp, wp), dtype=np.float32)
    for by in range(bh):
        for bx in range(bw):
            d = jpeg_like_dequantize(qblocks[by, bx], Qlast)
            blk = idct2(d) + 128.0
            Yrec[by*8:(by+1)*8, bx*8:(bx+1)*8] = blk

    # crop, put back to image
    Yrec = np.clip(Yrec[:h, :w], 0, 255).astype(np.uint8)
    ycrcb[:h, :w, 0] = Yrec
    stego_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return stego_bgr, embedded, cap_bits, nonzero_ac

def extract_f5_like_dct(stego_bgr: np.ndarray, passphrase: str, quality=85, k=3):
    """
    Extract full capacity bits from stego (F5-like), return bits np.uint8.
    """
    ycrcb = cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)
    h, w = Y.shape
    hp = (h + 7) // 8 * 8
    wp = (w + 7) // 8 * 8
    Yp = np.zeros((hp, wp), dtype=np.float32)
    Yp[:h, :w] = Y
    Yp -= 128.0

    blocks = block_view(Yp)
    bh, bw = blocks.shape[:2]

    qblocks = np.empty((bh, bw, 8, 8), dtype=np.int16)
    Qlast = None
    for by in range(bh):
        for bx in range(bw):
            d = dct2(blocks[by, bx])
            qb, Q = jpeg_like_quantize(d, quality=quality)
            qblocks[by, bx] = qb
            Qlast = Q

    cap_bits, _ = capacity_bits_f5_like(qblocks, k=k)

    refs = collect_ac_coeffs(qblocks)
    perm = permute_indices(len(refs), passphrase)
    refs = [refs[i] for i in perm]

    n = (2**k - 1)
    out_bits = np.empty(cap_bits, dtype=np.uint8)

    idx_ref = 0
    idx_out = 0
    while (idx_ref + n) <= len(refs) and (idx_out + k) <= cap_bits:
        group = refs[idx_ref: idx_ref + n]
        x = np.array([coeff_lsb(qblocks[by, bx, u, v]) for (by, bx, u, v) in group], dtype=np.uint8)
        s = syndrome7(x)  # k bits
        out_bits[idx_out:idx_out+k] = s[:k]
        idx_out += k
        idx_ref += n

    return out_bits

# ============================== CLI =======================================
def main():
    ap = argparse.ArgumentParser(description="Single-file Chaotic XOR + F5-like DCT steganography (no repo)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("embed", help="encrypt then embed")
    pe.add_argument("--in", dest="inp", required=True)
    pe.add_argument("--out", dest="out", required=True)
    pe.add_argument("--msgfile", required=True, help="secret .txt (read as bytes)")
    pe.add_argument("--passphrase", default="abc123")
    pe.add_argument("--mu", type=float, required=True)
    pe.add_argument("--x0", type=float, required=True)
    pe.add_argument("--warmup", type=int, default=1000)
    pe.add_argument("--quality", type=int, default=85, help="JPEG-like quant quality (lower => less capacity, more distortion)")
    pe.add_argument("--k", type=int, default=3, help="matrix encoding parameter (k=3 => groups of 7 coeffs, 3 bits payload)")

    px = sub.add_parser("extract", help="extract then decrypt")
    px.add_argument("--in", dest="inp", required=True)
    px.add_argument("--out", dest="out", required=True)
    px.add_argument("--passphrase", default="abc123")
    px.add_argument("--mu", type=float, required=True)
    px.add_argument("--x0", type=float, required=True)
    px.add_argument("--warmup", type=int, default=1000)
    px.add_argument("--quality", type=int, default=85)
    px.add_argument("--k", type=int, default=3)
    px.add_argument("--verify", help="optional plaintext file to compare SHA-256")

    args = ap.parse_args()

    if args.cmd == "embed":
        cover = cv2.imread(args.inp)
        if cover is None:
            raise FileNotFoundError(args.inp)

        with open(args.msgfile, "rb") as f:
            pt = f.read()

        # estimate capacity by running DCT once
        # build payload with truncation AFTER cap known
        # we do a quick cap by embedding empty payload to query cap
        empty_payload = np.zeros(0, dtype=np.uint8)
        stego_tmp, _, cap_bits, nonzero_ac = embed_f5_like_dct(
            cover, empty_payload, args.passphrase, quality=args.quality, k=args.k
        )

        payload_bits, st = build_payload_bits_trunc(pt, args.mu, args.x0, args.warmup, cap_bits)
        # embed real
        stego, embedded_bits, cap_bits2, _ = embed_f5_like_dct(
            cover, payload_bits, args.passphrase, quality=args.quality, k=args.k
        )
        cv2.imwrite(args.out, stego)

        psnr_val, ssim_val = compute_metrics(cover, stego)

        # ER theo định nghĩa của bạn: embedded message bits / total message bits (ciphertext bits)
        ER = st["ER_message"]
        completion_flag = st["complete_flag"]

        print(f"Cover: {cover.shape[1]}x{cover.shape[0]}, nonzero AC≈{nonzero_ac}")
        print(f"Capacity (F5-like, k={args.k}): {cap_bits} bits")
        print(f"Ciphertext needed: {st['need_ct_bytes']} bytes = {st['total_msg_bits']} bits")
        print(f"Ciphertext embedded: {st['use_ct_bytes']} bytes = {st['embedded_msg_bits']} bits")
        print(f"Embedded payload bits (header+ct): {st['payload_bits_used']} bits")
        print(f"Embedded bits (actually written): {embedded_bits} bits")
        print(f"ER (embedded_msg_bits / total_msg_bits): {ER:.6f}")
        print(f"Completion flag (1=full,0=partial): {completion_flag}")
        print(f"PSNR={psnr_val:.2f} SSIM={ssim_val:.4f}")

    else:
        stego = cv2.imread(args.inp)
        if stego is None:
            raise FileNotFoundError(args.inp)

        bits = extract_f5_like_dct(stego, args.passphrase, quality=args.quality, k=args.k)
        pt, ct_len_bytes = parse_payload_bits(bits, args.mu, args.x0, args.warmup)

        with open(args.out, "wb") as f:
            f.write(pt)

        print(f"Ciphertext length from header: {ct_len_bytes} bytes")
        print(f"Recovered plaintext: {len(pt)} bytes -> {args.out}")

        if args.verify and os.path.isfile(args.verify):
            h1 = file_sha256(args.verify)
            h2 = file_sha256(args.out)
            print("VERIFY:", "OK" if h1 == h2 else "MISMATCH")
            print("SHA256 orig:", h1)
            print("SHA256 recv:", h2)

if __name__ == "__main__":
    main()