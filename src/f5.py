import numpy as np
import cv2
from math import log10
from skimage.metrics import structural_similarity as ssim
from skimage.util import view_as_blocks
import random

# --------- CHAOTIC ENCRYPTION ---------
def logistic_map(x0, r, size):
    seq = np.zeros(size)
    x = x0
    for i in range(size):
        x = r * x * (1 - x)
        seq[i] = x
    return seq

def chaotic_encrypt(binary_data, x0=0.5, r=3.9):
    n_bits = len(binary_data)
    n_bytes = (n_bits + 7) // 8
    chaotic_seq = logistic_map(x0, r, n_bytes)
    chaotic_bytes = (chaotic_seq * 256).astype(np.uint8)
    chaotic_bits = np.unpackbits(chaotic_bytes)[:n_bits]
    encrypted = np.bitwise_xor(binary_data, chaotic_bits)
    return encrypted

def chaotic_decrypt(encrypted_data, x0=0.5, r=3.9):
    return chaotic_encrypt(encrypted_data, x0, r)  # XOR is symmetric

def text_to_bits(text):
    """Convert text to binary array"""
    return np.unpackbits(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))

def bits_to_text(bits):
    """Convert binary array back to text"""
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    bytes_arr = np.packbits(bits)
    try:
        return bytes_arr.tobytes().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"[Decode Error] {str(e)}"

def matrix_encode(data_bits, k=3):
    """
    Matrix encoding for F5 algorithm
    k: number of bits to encode in k+1 coefficients
    """
    n = 2**k - 1  # number of coefficients needed
    encoded_bits = []
    
    for i in range(0, len(data_bits), k):
        chunk = data_bits[i:i+k]
        if len(chunk) < k:
            # Pad with zeros if needed
            chunk = np.pad(chunk, (0, k - len(chunk)), 'constant')
        
        # Create syndrome
        syndrome = 0
        for j, bit in enumerate(chunk):
            if bit:
                syndrome ^= (j + 1)
        
        encoded_bits.extend(chunk)
        encoded_bits.append(syndrome)
    
    return np.array(encoded_bits)

def matrix_decode(coefficients, k=3):
    """
    Matrix decoding for F5 algorithm
    """
    n = 2**k - 1
    decoded_bits = []
    
    for i in range(0, len(coefficients), n):
        chunk = coefficients[i:i+n]
        if len(chunk) < n:
            break
        
        # Calculate syndrome
        syndrome = 0
        for j, coeff in enumerate(chunk):
            if coeff != 0:
                syndrome ^= (j + 1)
        
        # Extract data bits (first k bits)
        data_bits = chunk[:k]
        decoded_bits.extend(data_bits)
    
    return np.array(decoded_bits)

def permutative_straddling(coefficients, seed=42):
    """
    Permutative straddling to distribute embedding across coefficients
    """
    random.seed(seed)
    indices = list(range(len(coefficients)))
    random.shuffle(indices)
    return indices

def embed_f5(image_path, secret_message, k=3, output_path="../images/output/lsb_method/stego_f5_1.png", x0=0.5, r=3.9):
    """
    Embed secret message using F5 steganography with chaotic encryption
    
    Args:
        image_path: Path to cover image
        secret_message: Text message to hide
        k: Matrix encoding parameter (k=3 means 3 bits in 7 coefficients)
        output_path: Path to save stego image
        x0: Initial value for chaotic encryption
        r: Growth rate for chaotic encryption
    
    Returns:
        Path to stego image
    """
    # Read cover image
    cover_image = cv2.imread(image_path)
    if cover_image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Convert to YUV color space
    yuv_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2YUV)
    y_channel = yuv_image[:, :, 0].astype(np.float32)
    
    # Store original dimensions
    original_h, original_w = y_channel.shape
    
    # Convert message to binary and apply chaotic encryption
    binary_message = text_to_bits(secret_message)
    encrypted_message = chaotic_encrypt(binary_message, x0, r)
    message_length = len(encrypted_message)
    
    # Pad image to be divisible by 8
    h, w = y_channel.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        y_channel = cv2.copyMakeBorder(y_channel, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    
    # Subtract 128 for DCT
    y_channel = y_channel - 128
    
    # Divide into 8x8 blocks
    blocks = view_as_blocks(y_channel, block_shape=(8, 8))
    h_blocks, w_blocks, _, _ = blocks.shape
    
    # Collect all DCT coefficients (excluding DC)
    all_coeffs = []
    coeff_positions = []
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            dct_block = cv2.dct(blocks[i, j])
            # Skip DC coefficient (0,0) and collect AC coefficients
            for x in range(8):
                for y in range(8):
                    if x == 0 and y == 0:
                        continue
                    if dct_block[x, y] != 0:  # Only non-zero coefficients
                        all_coeffs.append(dct_block[x, y])
                        coeff_positions.append((i, j, x, y))
    
    # Check capacity
    n = 2**k - 1  # coefficients needed per k data bits
    max_capacity = (len(all_coeffs) // n) * k
    if message_length > max_capacity:
        raise ValueError(f"Message too large. Need {message_length} bits, but image can only hold {max_capacity} bits")
    
    # Matrix encode the encrypted message
    encoded_bits = matrix_encode(encrypted_message, k)
    
    # Permutative straddling
    indices = permutative_straddling(all_coeffs)
    
    # Embed encoded bits
    coeff_idx = 0
    for i in range(0, len(encoded_bits), n):
        if coeff_idx >= len(indices):
            break
            
        chunk = encoded_bits[i:i+n]
        if len(chunk) < n:
            chunk = np.pad(chunk, (0, n - len(chunk)), 'constant')
        
        # Embed chunk in coefficients
        for j, bit in enumerate(chunk):
            if coeff_idx + j >= len(indices):
                break
                
            pos = indices[coeff_idx + j]
            if pos < len(all_coeffs):
                # Modify coefficient based on bit
                if bit == 1:
                    # Ensure coefficient is odd
                    if int(all_coeffs[pos]) % 2 == 0:
                        all_coeffs[pos] += 1 if all_coeffs[pos] > 0 else -1
                else:
                    # Ensure coefficient is even
                    if int(all_coeffs[pos]) % 2 == 1:
                        all_coeffs[pos] += 1 if all_coeffs[pos] > 0 else -1
        
        coeff_idx += n
    
    # Reconstruct DCT blocks
    coeff_idx = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            dct_block = cv2.dct(blocks[i, j])
            # Update coefficients
            for x in range(8):
                for y in range(8):
                    if x == 0 and y == 0:
                        continue
                    if dct_block[x, y] != 0:
                        if coeff_idx < len(all_coeffs):
                            dct_block[x, y] = all_coeffs[coeff_idx]
                            coeff_idx += 1
            
            # Apply inverse DCT
            blocks[i, j] = cv2.idct(dct_block)
    
    # Reconstruct Y channel
    # Reshape blocks to the padded shape
    y_channel = blocks.reshape(y_channel.shape)
    y_channel = y_channel + 128
    
    # Remove padding to get back to original dimensions
    y_channel = y_channel[:original_h, :original_w]
    
    # Convert back to BGR
    # Create a new YUV image with the correct dimensions
    yuv_stego = np.zeros((original_h, original_w, 3), dtype=np.uint8)
    yuv_stego[:, :, 0] = np.clip(y_channel, 0, 255).astype(np.uint8)
    # Copy U and V channels from original (cropped to match)
    yuv_stego[:, :, 1] = yuv_image[:original_h, :original_w, 1]
    yuv_stego[:, :, 2] = yuv_image[:original_h, :original_w, 2]
    
    stego_image = cv2.cvtColor(yuv_stego, cv2.COLOR_YUV2BGR)
    
    # Save stego image
    cv2.imwrite(output_path, stego_image)
    
    return output_path

def extract_f5(stego_image_path, message_length, k=3, x0=0.5, r=3.9):
    """
    Extract secret message from stego image using F5 steganography with chaotic decryption
    
    Args:
        stego_image_path: Path to stego image
        message_length: Length of embedded message in bits
        k: Matrix encoding parameter
        x0: Initial value for chaotic decryption
        r: Growth rate for chaotic decryption
    
    Returns:
        Extracted text message
    """
    # Read stego image
    stego_image = cv2.imread(stego_image_path)
    if stego_image is None:
        raise ValueError(f"Cannot read image: {stego_image_path}")
    
    # Convert to YUV color space
    yuv_image = cv2.cvtColor(stego_image, cv2.COLOR_BGR2YUV)
    y_channel = yuv_image[:, :, 0].astype(np.float32)
    
    # Store original dimensions
    original_h, original_w = y_channel.shape
    
    # Pad image to be divisible by 8
    h, w = y_channel.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        y_channel = cv2.copyMakeBorder(y_channel, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    
    # Subtract 128 for DCT
    y_channel = y_channel - 128
    
    # Divide into 8x8 blocks
    blocks = view_as_blocks(y_channel, block_shape=(8, 8))
    h_blocks, w_blocks, _, _ = blocks.shape
    
    # Collect all DCT coefficients (excluding DC)
    all_coeffs = []
    
    for i in range(h_blocks):
        for j in range(w_blocks):
            dct_block = cv2.dct(blocks[i, j])
            # Skip DC coefficient (0,0) and collect AC coefficients
            for x in range(8):
                for y in range(8):
                    if x == 0 and y == 0:
                        continue
                    if dct_block[x, y] != 0:  # Only non-zero coefficients
                        all_coeffs.append(dct_block[x, y])
    
    # Permutative straddling (same seed as embedding)
    indices = permutative_straddling(all_coeffs, seed=42)
    
    # Extract bits from coefficients
    extracted_bits = []
    n = 2**k - 1
    
    for i in range(0, len(indices), n):
        chunk_indices = indices[i:i+n]
        chunk_bits = []
        
        for idx in chunk_indices:
            if idx < len(all_coeffs):
                # Extract bit from coefficient parity
                bit = 1 if int(all_coeffs[idx]) % 2 == 1 else 0
                chunk_bits.append(bit)
        
        if len(chunk_bits) == n:
            extracted_bits.extend(chunk_bits)
    
    # Matrix decode
    decoded_bits = matrix_decode(np.array(extracted_bits), k)
    
    # Truncate to original message length
    decoded_bits = decoded_bits[:message_length]
    
    # Decrypt and convert bits to text
    decrypted_bits = chaotic_decrypt(decoded_bits, x0, r)
    extracted_message = bits_to_text(decrypted_bits)
    
    return extracted_message

def psnr(original, modified):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / np.sqrt(mse))

def ssim_score(original, modified):
    """Calculate Structural Similarity Index"""
    return ssim(original, modified, channel_axis=2)

def compute_embedding_rate(message_length_bits, image_size_in_pixels):
    """Calculate embedding rate"""
    # F5 can embed k bits in 2^k - 1 coefficients
    k = 3
    n = 2**k - 1
    max_capacity = (image_size_in_pixels // 64) * k  # Approximate
    ER = message_length_bits / max_capacity
    return ER, max_capacity

def compute_detection_rate(original_bits, recovered_bits):
    """Calculate detection rate (bit accuracy)"""
    min_len = min(len(original_bits), len(recovered_bits))
    if min_len == 0:
        return 0
    D = np.sum(original_bits[:min_len] == recovered_bits[:min_len])
    T = len(original_bits)
    return D / T

def main():
    """Demo function for F5 steganography with chaotic encryption"""
    cover_image = "../images/input/cover_1.jpg"
    secret_message = "In a world that never stops moving, we often forget that we are allowed to slow down. The modern landscape is filled with signals telling us to go faster, achieve more, and stay constantly available. We are encouraged to maximize every moment, to make use of every second, to multitask, to push forward without pause. There is so much emphasis on growth and ambition that rest has become a foreign concept, and presence has become a rare experience. We live surrounded by screens and noise, and yet often feel lonelier than ever before. We communicate constantly, yet many of us rarely feel truly heard. We succeed publicly, but struggle privately. We are taught how to win, but not how to rest, how to perform, but not how to be. And yet, something inside us remembers. Some part of us, even when buried beneath years of pressure and expectation, still knows that life was never meant to be a race. It was meant to be a journey. A personal, intricate unfolding. A space for feeling, not just functioning. A rhythm, not a rush. That part of us is quiet. It speaks not in commands, but in questions. It wonders where the joy went. It asks when we last felt truly alive. It longs for space to breathe, to create, to simply exist without needing to prove anything. That voice can be hard to hear when we are caught in the machinery of modern life. But it never disappears. It waits. And it grows stronger when we stop and listen. Listening to ourselves has become a radical act. In a world that constantly pulls us outward, turning our attention inward is often misunderstood. People might say we are selfish, unmotivated, lazy, or too sensitive. But the truth is that self-awareness is not self-indulgence. It is a form of self-responsibility. When we begin to pay attention to what we are feeling, what we are avoiding, what we are carrying, we begin to move through life with more wisdom and care. We stop projecting our pain onto others. We stop chasing things we do not truly want. We start making choices that are aligned with our deeper values, not just with what is popular or expected. This kind of inner work does not happen quickly. It is not glamorous. It rarely brings applause. But it brings peace. And peace is something that cannot be bought, rushed, or faked. It is something that is cultivated through presence. Through honesty. Through the willingness to face discomfort rather than escape it. It comes when we stop numbing ourselves with busyness and begin to sit with the truth of our own lives. That truth might include grief, regret, longing, or confusion. But it also includes beauty, courage, clarity, and compassion. The full range of human experience lives inside us. And when we deny any part of it, we flatten our lives into something small and mechanical. Many people go through years of their lives feeling disconnected from themselves. They feel like they are living on autopilot, doing what they are supposed to do, checking all the boxes, but not really feeling much of anything. Sometimes it takes a breakdown, a loss, or a sudden change to wake them up. But sometimes, it just takes a moment of stillness. A moment when everything slows down, and they hear something inside them say, I want more than this. I want to feel again. I want to live differently. That moment is sacred. It is the beginning of something new. Not necessarily a change in circumstances, but a change in direction. A change in attention. A return to the center. The center is where life begins again. It is the space within us that is not defined by roles, titles, achievements, or appearance. It is the part of us that simply is. That watches. That feels. That remembers who we were before the world told us who to be. When we reconnect with that space, even for a few minutes a day, we begin to heal. We begin to understand ourselves more deeply. We begin to see through the noise and find what is essential. And from there, we can start to build lives that are not just successful by external standards, but meaningful by internal ones. To live this way does not mean escaping responsibility. It means approaching responsibility with intention. It means showing up for your life from a place of presence rather than pressure. You can still work, create, love, and contribute. But you do so from a place that is rooted in who you are, not who you think you have to be. You no longer chase approval because you are already grounded in your own integrity. You no longer need constant distraction because you are no longer afraid of your own thoughts. You become someone who moves through the world with a quiet strength, not needing to be the loudest voice, because your voice is connected to something real. This journey is not about perfection. It is about practice. There will be days when you forget, when you fall back into old habits, when the world feels louder than your own intuition. That is okay. This is not a path of endless progress. It is a spiral. You will return to the same lessons, the same questions, but each time from a slightly deeper place. You will grow not in a straight line, but in layers. And every time you return to yourself, even after drifting far away, you strengthen the trust you have in your own path. You are allowed to grow slowly. You are allowed to rest often. You are allowed to protect your energy. You are allowed to want a life that feels gentle instead of rushed. You are allowed to say no to what drains you and yes to what brings you peace. These are not signs of weakness. They are signs of wisdom. In a world that glorifies overwork, choosing rest is an act of rebellion. In a culture that prizes speed, choosing presence is a revolution. And in a time where everyone is expected to know everything, choosing humility and curiosity is a gift. Presence is not a skill you master once. It is a relationship you return to every day. Sometimes you will feel deeply connected to the moment. Other times you will feel scattered and restless. But the point is not to be perfect. The point is to notice. To notice when you are here. To notice when you are not. To keep coming back, gently, without judgment. Just as the ocean always returns to shore, you can always return to yourself. The way you treat yourself in these moments matters. Speak kindly. Move slowly. Let yourself be human. Let yourself be tired. Let yourself be joyful without explanation. Let yourself grieve without apology. This is how you build a life that honors the whole of who you are. Not just the parts that are easy to share, but the parts that are raw, vulnerable, and real. In this quiet, honest living, you may find something you did not know you were looking for. A deeper sense of belonging. A deeper connection to others. A greater appreciation for the little things. The way light moves across a room. The sound of your own laughter. The steadiness of your breath when you finally feel safe. These are the treasures that fast living cannot offer. These are the rewards of presence. And they are always available, if we are willing to slow down and receive them. You are not behind. You are not late. You are not broken. You are becoming. And the pace of becoming is not a race. It is a rhythm. One that is unique to you. Trust it. Trust the pauses. Trust the stillness. Trust that even when nothing seems to be happening on the outside, something meaningful is unfolding within. Growth is not always visible. Healing is not always loud. But both are sacred. And both are happening, even now. Keep returning. Keep listening. Keep softening. Keep showing up for your life, not just as something to survive, but as something to cherish. The world may never slow down, but you can. And in doing so, you remind others that they can too. Your presence is powerful. Your peace is revolutionary. Your life is yours to shape, one quiet, conscious moment at a time."
    x0, r = 0.6, 3.9
    
    print("=== F5 Steganography with Chaotic Encryption Demo ===")
    print(f"Cover image: {cover_image}")
    print(f"Secret message: {secret_message}")
    print(f"Message length: {len(secret_message)} characters")
    print(f"Chaotic parameters: x0={x0}, r={r}")
    
    # Convert message to bits for analysis
    binary_message = text_to_bits(secret_message)
    print(f"Binary length: {len(binary_message)} bits")
    
    # Embed message
    try:
        stego_path = embed_f5(cover_image, secret_message, k=3, x0=x0, r=r)
        print(f"Stego image saved: {stego_path}")
        
        # Extract message
        extracted_message = extract_f5(stego_path, len(binary_message), k=3, x0=x0, r=r)
        print(f"Extracted message: {extracted_message}")
        print(f"Recovery success: {extracted_message == secret_message}")
        
        # Calculate metrics
        original = cv2.imread(cover_image)
        stego = cv2.imread(stego_path)
        
        print("\n=== Performance Metrics ===")
        print(f"PSNR: {psnr(original, stego):.2f} dB")
        print(f"SSIM: {ssim_score(original, stego):.4f}")
        
        dr = compute_detection_rate(binary_message, text_to_bits(extracted_message))
        print(f"Detection Rate (DR): {dr:.4f}")
        
        er, max_capacity = compute_embedding_rate(len(binary_message), len(original.flatten()))
        print(f"Embedding Rate (ER): {er:.4f} ({len(binary_message)} / {max_capacity} bits)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

