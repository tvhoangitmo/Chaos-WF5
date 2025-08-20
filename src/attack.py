import numpy as np
import cv2
from skimage.util import view_as_blocks
from scipy.stats import chisquare
from tqdm import tqdm
from nltk.corpus import words
import nltk
from bitarray import bitarray

# Tải bộ từ tiếng Anh
nltk.download('words')
english_words = set(words.words())

# Hamming(9,4) parity-check matrix (transposed) - 5x9 matrix
H = np.array([
    [1, 1, 1, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
], dtype=np.uint8)

# Bước 1: Trích hệ số DCT
def jpeg_dct_coefficients(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Pad image to be divisible by 8
    h, w = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    
    img = img.astype(np.float32) - 128
    blocks = view_as_blocks(img, block_shape=(8, 8))
    h, w, _, _ = blocks.shape
    dcts = np.zeros_like(blocks)
    for i in range(h):
        for j in range(w):
            dcts[i, j] = cv2.dct(blocks[i, j])
    coeffs = dcts.reshape(-1, 8, 8)[:, 1:, 1:].reshape(-1)
    coeffs = np.round(coeffs).astype(int)
    return coeffs

def analyze_parity_and_chi2(coeffs):
    coeffs = coeffs[coeffs != 0]
    even = np.sum(coeffs % 2 == 0)
    odd = np.sum(coeffs % 2 != 0)
    total = even + odd
    ratio = odd / total if total > 0 else 0
    stat, p = chisquare([even, odd], f_exp=[total/2, total/2])
    print(f"[+] Parity: {even} even / {odd} odd ({ratio*100:.2f}% odd)")
    print(f"[+] Chi-square: {stat:.2f}, p-value: {p:.4e}")
    return coeffs

# Bước 2: Giả định dùng WF5 với Hamming(9,4) → trích dữ liệu
def wf5_extract(cipher_coeffs):
    bits = []
    i = 0
    while i + 9 <= len(cipher_coeffs):
        block = cipher_coeffs[i:i+9]
        if np.count_nonzero(block) < 9:
            i += 1
            continue
        r = np.array([b % 2 for b in block])
        s = (H @ r) % 2
        bits.extend(s.tolist())
        i += 9
    return np.array(bits[:21600], dtype=np.uint8)

# Bước 3: Sinh chuỗi chaotic
def logistic_map(x0, r, size):
    seq = np.zeros(size)
    x = x0
    for i in range(size):
        x = r * x * (1 - x)
        seq[i] = x
    return seq

# Bước 4: Brute-force chaotic decryption (tối ưu)
def chaotic_brute_force(encrypted_bits, max_trials=100000):
    n_bits = len(encrypted_bits)
    n_bytes = n_bits // 8
    best_score = 0
    best_plaintext = None
    best_key = None

    for _ in tqdm(range(max_trials)):
        x0 = np.random.uniform(0.1, 0.9)
        r = np.random.uniform(3.57, 4.0)
        chaos = logistic_map(x0, r, n_bytes)
        chaos_bytes = (chaos * 256).astype(np.uint8)
        chaos_bits = np.unpackbits(chaos_bytes)[:n_bits]
        decrypted_bits = np.bitwise_xor(encrypted_bits, chaos_bits)
        decrypted_bytes = np.packbits(decrypted_bits)
        try:
            text = decrypted_bytes.tobytes().decode('utf-8')
        except:
            continue
        words_found = sum(1 for w in text.split() if w.lower() in english_words)
        score = words_found / max(len(text.split()), 1)
        if score > best_score:
            best_score = score
            best_plaintext = text
            best_key = (x0, r)
        if score > 0.7:
            print(f"[!] Có thể đúng khóa: x0={x0:.6f}, r={r:.6f}, score={score:.2f}")
            return best_key, score, text

    return best_key, best_score, best_plaintext

# ======= CHẠY TOÀN BỘ QUÁ TRÌNH =======

# Đọc ảnh và phân tích
coeffs = jpeg_dct_coefficients("../images/output/wf5_method/stego_wf5_4.png")
cipher_coeffs = analyze_parity_and_chi2(coeffs)

# Trích encrypted bits từ ảnh
encrypted = wf5_extract(cipher_coeffs)
print(f"[+] Đã trích {len(encrypted)} bits từ ảnh.")

# Brute-force chaotic key để giải mã
key, score, message = chaotic_brute_force(encrypted, max_trials=100000)

# In kết quả cuối cùng
if message:
    print("\n[✔] 🔑 Best key found:")
    print(f"   x0 = {key[0]:.6f}")
    print(f"   r  = {key[1]:.6f}")
    print(f"[✔] English score: {score:.2f}")
    print("🔓 Message preview:\n", message[:500])
else:
    print("\n[✘] Không tìm được khóa có điểm số cao (score > 0.7)")
    if score > 0.2:
        print(f"[~] Có bản giải mã gần đúng với score ≈ {score:.2f}.")
        print("[~] Bản giải mã gần đúng (preview):\n", message[:500] if message else "[empty]")
    else:
        print("[i] Tất cả kết quả đều không giống văn bản tiếng Anh.")
