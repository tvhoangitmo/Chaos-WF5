import argparse, os, hashlib
import numpy as np, cv2
from skimage.metrics import structural_similarity as ssim


def _imread_bgr_unicode(path: str):
    """cv2.imread often fails on Windows when path contains non-ASCII (e.g. Cyrillic)."""
    with open(path, "rb") as f:
        raw = f.read()
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _imwrite_bgr_unicode(path: str, img: np.ndarray) -> bool:
    ext = os.path.splitext(path)[1].lower() or ".png"
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        ext = ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


# ----------------------------- Bit helpers ---------------------------------
def bytes_to_bits(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = ((arr[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.uint8)
    return bits.reshape(-1)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.reshape(-1)
    bits = bits[:(bits.size // 8) * 8]
    return np.packbits(bits, bitorder='big').tobytes()

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

# ---------------------- Payload build/parse (NEW) --------------------------
def build_payload_bits_truncate(pt: bytes, mu: float, x0: float, warmup: int, cap_bits: int):
    """
    Build payload as: [32-bit length in BYTES of ciphertext actually embedded] + ciphertext bits.
    If cap insufficient => truncate ciphertext to fit cap. Always tries to keep header.
    Returns: payload_bits_use, stats dict
    """
    ct_full = chaotic_xor(pt, mu, x0, warmup)
    need_ct_bytes = len(ct_full)

    if cap_bits < 32:
        # cannot even store header
        use_ct_bytes = 0
        payload_bits_use = np.zeros(0, dtype=np.uint8)
    else:
        avail_ct_bits = cap_bits - 32
        use_ct_bytes = min(need_ct_bytes, avail_ct_bits // 8)  # only full bytes
        ct_use = ct_full[:use_ct_bytes]
        payload_bits_use = np.concatenate([u32_to_bits(use_ct_bytes), bytes_to_bits(ct_use)])

    embedded_ct_bits = use_ct_bytes * 8
    total_ct_bits = need_ct_bytes * 8
    embed_ratio = (embedded_ct_bits / total_ct_bits) if total_ct_bits else 1.0
    completion_flag = 1 if use_ct_bytes == need_ct_bytes else 0

    stats = {
        "need_ct_bytes": need_ct_bytes,
        "use_ct_bytes": use_ct_bytes,
        "embedded_ct_bits": embedded_ct_bits,
        "total_ct_bits": total_ct_bits,
        "embed_ratio": embed_ratio,             # 0..1
        "completion_flag": completion_flag,     # 1/0
        "payload_bits_used": int(payload_bits_use.size),
    }
    return payload_bits_use, stats

def parse_payload_bits(bits: np.ndarray, mu: float, x0: float, warmup: int = 1000) -> bytes:
    length = bits_to_u32(bits[:32])
    blob = bits_to_bytes(bits[32:32 + length * 8])
    return chaotic_xor(blob, mu, x0, warmup)

# ----------------------------- DCT embed/extract ---------------------------
def max_capacity_bits(img: np.ndarray, block_size: int = 8) -> int:
    h, w, _ = img.shape
    H = (h // block_size) * block_size
    W = (w // block_size) * block_size
    return (H // block_size) * (W // block_size)  # 1 bit / block

def _embed_dct_array(img: np.ndarray, bits: np.ndarray,
                     block_size: int = 8, pos=(4, 3), tau: float = 5.0):
    h, w, _ = img.shape
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    Y = Y.astype(np.float32)

    bs = block_size
    H = (h // bs) * bs
    W = (w // bs) * bs
    outY = Y.copy()

    idx = 0
    for by in range(0, H, bs):
        for bx in range(0, W, bs):
            if idx >= bits.size:
                break
            block = outY[by:by + bs, bx:bx + bs]
            dct = cv2.dct(block)
            bit = int(bits[idx])
            idx += 1

            u, v = pos
            C = float(dct[u, v])

            # ensure magnitude
            if abs(C) < tau:
                C = tau if C >= 0 else -tau

            # encode bit by sign (0 => positive, 1 => negative)
            if bit == 0:
                C = abs(C)
            else:
                C = -abs(C)

            if abs(C) < tau:
                C = tau if C > 0 else -tau

            dct[u, v] = C
            outY[by:by + bs, bx:bx + bs] = cv2.idct(dct)

        if idx >= bits.size:
            break

    outY = np.clip(outY, 0, 255).astype(np.uint8)
    ycrcb_out = cv2.merge([outY, Cr, Cb])
    bgr_out = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)
    return bgr_out, idx  # idx = number of bits embedded

def _extract_dct_array(img: np.ndarray, block_size: int = 8, pos=(4, 3)) -> np.ndarray:
    h, w, _ = img.shape
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, _, _ = cv2.split(ycrcb)
    Y = Y.astype(np.float32)

    bs = block_size
    H = (h // bs) * bs
    W = (w // bs) * bs

    bits = []
    for by in range(0, H, bs):
        for bx in range(0, W, bs):
            block = Y[by:by + bs, bx:bx + bs]
            dct = cv2.dct(block)
            C = float(dct[pos[0], pos[1]])
            bits.append(1 if C < 0 else 0)
    return np.array(bits, dtype=np.uint8)

def embed_dct_full(img_path: str, bits: np.ndarray, out_path: str,
                   block_size: int = 8, pos=(4, 3), tau: float = 5.0) -> int:
    img = _imread_bgr_unicode(img_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {img_path}")
    stego, used = _embed_dct_array(img, bits, block_size, pos, tau)
    if not _imwrite_bgr_unicode(out_path, stego):
        raise RuntimeError(f"Failed to write stego image: {out_path}")
    return used

def extract_dct_full(path: str, block_size: int = 8, pos=(4, 3)) -> np.ndarray:
    img = _imread_bgr_unicode(path)
    if img is None:
        raise FileNotFoundError(f"Stego image not found: {path}")
    return _extract_dct_array(img, block_size, pos)

# ----------------------------- Metrics / hash -------------------------------
def compute_metrics(img1, img2):
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
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

# ----------------------------- CLI -----------------------------------------
def main():
    ap = argparse.ArgumentParser(description='DCT + Chaotic XOR (logistic)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    pe = sub.add_parser('embed', help='Chaotic-encrypt then embed in DCT domain (embed max possible)')
    pe.add_argument('--in', dest='inp', required=True)
    pe.add_argument('--out', dest='out', required=True)
    pe.add_argument('--msgfile', required=True, help='path to file containing secret bytes')
    pe.add_argument('--mu', type=float, required=True)
    pe.add_argument('--x0', type=float, required=True)
    pe.add_argument('--warmup', type=int, default=1000)
    pe.add_argument('--tau', type=float, default=5.0)
    pe.add_argument('--pos_u', type=int, default=4)
    pe.add_argument('--pos_v', type=int, default=3)

    px = sub.add_parser('extract', help='Extract then chaotic-decrypt')
    px.add_argument('--in', dest='inp', required=True)
    px.add_argument('--out', dest='out', required=True)
    px.add_argument('--mu', type=float, required=True)
    px.add_argument('--x0', type=float, required=True)
    px.add_argument('--warmup', type=int, default=1000)
    px.add_argument('--verify', dest='verify', help='optional: path to original message file for SHA-256 check')
    px.add_argument('--pos_u', type=int, default=4)
    px.add_argument('--pos_v', type=int, default=3)

    args = ap.parse_args()

    pos = (args.pos_u, args.pos_v)

    if args.cmd == 'embed':
        img = _imread_bgr_unicode(args.inp)
        if img is None:
            raise FileNotFoundError(f"Input image not found: {args.inp}")

        with open(args.msgfile, 'rb') as f:
            pt = f.read()

        cap = int(max_capacity_bits(img))
        payload_bits_use, st = build_payload_bits_truncate(pt, args.mu, args.x0, args.warmup, cap)

        if payload_bits_use.size == 0 and cap < 32:
            print(f"WARNING: capacity={cap} bits < 32 bits (header). Nothing embedded.")
            # still write a copy image (optional)
            _imwrite_bgr_unicode(args.out, img)
            return

        nbits = embed_dct_full(args.inp, payload_bits_use, args.out, pos=pos, tau=args.tau)

        stego = _imread_bgr_unicode(args.out)
        psnr, ss = compute_metrics(img, stego)

        total_image_bits = int(img.size * 8)
        ER_img = nbits / total_image_bits if total_image_bits else 0.0
        ER_cap = nbits / cap if cap else 0.0

        print(f"Capacity (DCT): {cap} bits (1 bit / 8x8 block)")
        print(f"Ciphertext needed: {st['need_ct_bytes']} bytes = {st['total_ct_bits']} bits")
        print(f"Ciphertext embedded: {st['use_ct_bytes']} bytes = {st['embedded_ct_bits']} bits")
        print(f"Message embedded ratio: {st['embed_ratio']:.6f}  ({st['embed_ratio']*100:.2f}%)")
        print(f"Completion flag (1=full,0=partial): {st['completion_flag']}")
        print(f"Embedded payload bits (header+ct): {st['payload_bits_used']} bits")
        print(f"Actually embedded bits: {nbits} bits")
        print(f"ER_img (embedded / total image bits): {ER_img:.8f}")
        print(f"ER_cap (embedded / capacity bits):    {ER_cap:.6f}")
        print(f"PSNR={psnr:.2f} SSIM={ss:.4f}")

    else:
        bits = extract_dct_full(args.inp, pos=pos)
        pt = parse_payload_bits(bits, args.mu, args.x0, args.warmup)
        with open(args.out, 'wb') as f:
            f.write(pt)
        print(f"Extracted {len(pt)} bytes -> {args.out}")

        if args.verify and os.path.isfile(args.verify):
            h1 = file_sha256(args.verify)
            h2 = file_sha256(args.out)
            ok = (h1 == h2)
            print("VERIFY:", "OK" if ok else "MISMATCH")
            print("SHA256 orig:", h1)
            print("SHA256 recv:", h2)

if __name__ == '__main__':
    main()
