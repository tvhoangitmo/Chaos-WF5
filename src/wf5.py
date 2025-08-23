import numpy as np
import cv2
from math import log10
from skimage.metrics import structural_similarity as ssim

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

# --------- WF5 EMBEDDING (WITH WEIGHTED HAMMING CODE) ---------
Hw = np.array([
    [1, 0, 0, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 1, 0, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 1, 1]
], dtype=np.uint8)

def compute_weight_mask(image, codeword_size=9, threshold=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_flat = magnitude.flatten()

    weights = np.where(grad_flat < threshold, 1, 2)
    n_groups = len(weights) // codeword_size
    weights = weights[:n_groups * codeword_size]
    return weights.reshape(n_groups, codeword_size)

# weights_mask = np.array([1]*4 + [2]*5)

# def embed_wf5_full(image_path, encrypted_data, output_path="stego_wf5.png"):
#     img = cv2.imread(image_path)
#     flat = img.flatten()
#     result = flat.copy()
#     data_idx = 0

#     codeword_size = 9
#     group_count = len(flat) // codeword_size
#     flat = flat[:group_count * codeword_size]
#     groups = flat.reshape(-1, codeword_size)

#     for i, group in enumerate(groups):
#         if data_idx + 4 > len(encrypted_data):
#             break

#         # Lấy 9 bit đầu tiên từ 9 byte (mỗi byte lấy bit thấp nhất)
#         bits = np.array([(b & 1) for b in group], dtype=np.uint8)
#         B = bits.copy()
#         S = np.dot(B, Hw.T) % 2
#         I = encrypted_data[data_idx:data_idx + 4]
#         A = (S ^ I) % 2

#         best_e = None
#         min_weight = float('inf')

#         for e_idx in range(codeword_size):
#             e = np.zeros(codeword_size, dtype=np.uint8)
#             e[e_idx] = 1
#             if np.all((np.dot(e, Hw.T) % 2) == A):
#                 weight = weights_mask[e_idx]
#                 if weight < min_weight:
#                     min_weight = weight
#                     best_e = e

#         if best_e is not None:
#             B = (B ^ best_e) % 2
#             # Gán lại bit thấp nhất của từng byte trong group
#             for j in range(codeword_size):
#                 result[i * codeword_size + j] = (result[i * codeword_size + j] & 0xFE) | B[j]

#         data_idx += 4

#     stego = result.reshape(img.shape)
#     cv2.imwrite(output_path, stego)
#     return output_path

def hamming_encode_4bits(data4):
    """
    Encode 4 bits (array of 4 uint8) to 9 bits using Hamming (9,4)
    """
    # Generator matrix G (4x9)
    G = np.array([
        [1,0,0,0,1,1,1,0,1],
        [0,1,0,0,1,1,0,1,1],
        [0,0,1,0,1,0,1,1,1],
        [0,0,0,1,0,1,1,1,1]
    ], dtype=np.uint8)
    codeword = np.dot(data4, G) % 2
    return codeword

def hamming_decode_9bits(code9):
    """
    Decode 9 bits (array of 9 uint8) to 4 bits using Hamming (9,4)
    """
    # Parity-check matrix H (5x9)
    H = np.array([
        [1,1,1,0,1,0,0,0,0],
        [1,1,0,1,0,1,0,0,0],
        [1,0,1,1,0,0,1,0,0],
        [0,1,1,1,0,0,0,1,0],
        [1,1,1,1,1,1,1,1,1]
    ], dtype=np.uint8)
    syndrome = np.dot(H, code9) % 2
    # Syndrome to error position (for single error correction)
    error_pos = None
    for i in range(9):
        if np.all(H[:, i] == syndrome):
            error_pos = i
            break
    code_corr = code9.copy()
    if error_pos is not None:
        code_corr[error_pos] ^= 1
    # Extract data bits (first 4 bits)
    data4 = code_corr[:4]
    return data4

def embed_wf5_full(image_path, binary_data, output_path="../images/output/wf5_method/stego_wf5.png", threshold=30):
    img = cv2.imread(image_path)
    flat = img.flatten()
    result = flat.copy()
    codeword_size = 9

    # Calculate number of groups (codewords)
    n_groups = len(flat) // codeword_size
    flat = flat[:n_groups * codeword_size]
    groups = flat.reshape(-1, codeword_size)

    # Calculate weights from Sobel gradient
    weights_mask = compute_weight_mask(img, codeword_size=codeword_size, threshold=threshold)

    data_idx = 0
    for i in range(n_groups):
        if data_idx + 4 > len(binary_data):
            break

        # 1. Get LSB bits of image group (original codeword)
        group = groups[i]
        B = np.array([(b & 1) for b in group], dtype=np.uint8)

        # 2. Current syndrome: S = B ⋅ Hw^T
        S = np.dot(B, Hw.T) % 2

        # 3. Get 4 data bits to embed
        I = np.zeros(4, dtype=np.uint8)
        remain = min(4, len(binary_data) - data_idx)
        I[:remain] = binary_data[data_idx:data_idx + remain]

        # 4. Calculate syndrome difference: A = S ⊕ I
        A = (S ^ I) % 2

        # 5. Find error vector e: e ⋅ Hw^T = A with minimum weight
        best_e = None
        min_weight = float('inf')
        weights = weights_mask[i]

        for n in range(512):  # Duyệt mọi vector nhị phân dài 9 bit
            e = np.array(list(np.binary_repr(n, width=9)), dtype=np.uint8)
            if np.all(np.dot(e, Hw.T) % 2 == A):
                weight = np.sum(e * weights)
                if weight < min_weight:
                    min_weight = weight
                    best_e = e

        # 6. If found e, modify bit
        if best_e is not None:
            B = (B ^ best_e) % 2
            for j in range(codeword_size):
                result[i * codeword_size + j] = (result[i * codeword_size + j] & 0xFE) | B[j]

        data_idx += 4

    # Save result image
    stego = result.reshape(img.shape)
    cv2.imwrite(output_path, stego)
    return output_path


def extract_wf5_full(stego_image_path, num_bits):
    img = cv2.imread(stego_image_path)
    flat = img.flatten()
    codeword_size = 9

    # Calculate number of groups
    n_groups = len(flat) // codeword_size
    flat = flat[:n_groups * codeword_size]
    groups = flat.reshape(-1, codeword_size)

    n_blocks = num_bits // 4
    if num_bits % 4 != 0:
        n_blocks += 1

    extracted_bits = []
    for i in range(min(n_blocks, n_groups)):
        group = groups[i]
        r_prime = np.array([(b & 1) for b in group], dtype=np.uint8)

        # Tính syndrome: I = r' ⋅ H_w^T
        I = np.dot(r_prime, Hw.T) % 2

        remain = min(4, num_bits - i * 4)
        extracted_bits.extend(I[:remain].tolist())

    return np.array(extracted_bits[:num_bits], dtype=np.uint8)


# --------- METRICS & UTILITIES ---------
def psnr(original, modified):
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / np.sqrt(mse))

def ssim_score(original, modified):
    return ssim(original, modified, channel_axis=2)

def compute_embedding_rate(encrypted_data_len, image_size_in_pixels):
    max_capacity = (image_size_in_pixels // 9) * 4
    ER = encrypted_data_len / max_capacity
    return ER, max_capacity

def compute_detection_rate(original_bits, recovered_bits):
    min_len = min(len(original_bits), len(recovered_bits))
    D = np.sum(original_bits[:min_len] == recovered_bits[:min_len])
    T = len(original_bits)
    return D / T

def text_to_bits(text):
    return np.unpackbits(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))

def bits_to_text(bits):
    n = (len(bits) // 8) * 8
    bits = bits[:n]
    bytes_arr = np.packbits(bits)
    try:
        return bytes_arr.tobytes().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"[Decode Error] {str(e)}"

# --------- WRAPPER FUNCTIONS FOR TEST SCRIPT ---------
def embed_wf5(image_path, secret_message, output_path="../images/output/wf5_method/stego_wf5.png", x0=0.6, r=3.9, threshold=30):
    """
    Wrapper function for WF5 embedding that matches the test script interface
    """
    # Convert text to bits
    binary = text_to_bits(secret_message)
    
    # Chaotic encryption
    encrypted = chaotic_encrypt(binary, x0, r)
    
    # Embed using the full WF5 algorithm
    return embed_wf5_full(image_path, encrypted, output_path, threshold)

def extract_wf5(stego_image_path, num_bits, x0=0.6, r=3.9):
    """
    Wrapper function for WF5 extraction that matches the test script interface
    """
    # Extract encrypted bits
    extracted = extract_wf5_full(stego_image_path, num_bits)
    
    # Chaotic decryption
    decrypted = chaotic_decrypt(extracted, x0, r)
    
    # Convert bits back to text
    return bits_to_text(decrypted)

def main():
    cover_image = "../images/input/cover_1.jpg"
    #secret_message = "In the modern world, the pace of life seems to accelerate with every passing year. Cities grow louder, schedules become tighter, and the space for silence slowly disappears from our daily routines. We wake up to the sound of alarms, check our phones before even rising from bed, and are pulled into a current of obligations, deadlines, messages, and expectations before we have a chance to ask ourselves how we truly feel. Day after day, we go through the motions, answering emails, attending meetings, scrolling through social media, yet often finish the day with a vague sense of emptiness, as if something essential is missing but we cannot quite name it. It is not that our lives are meaningless or that we are doing something wrong. Rather, it is that we have grown used to neglecting the quiet voice within us that longs for connection, rest, and reflection. The world teaches us to chase success, to collect accomplishments, and to measure worth by what can be seen, counted, and compared. But the heart measures differently. It asks, Are you at peace? Do you feel joy in what you do? Are you moving toward a life that feels honest and whole? These questions are easy to ignore, especially when we are surrounded by noise, both literal and metaphorical. But ignoring them does not make them disappear. Over time, the consequences surface in unexpected ways: a deep fatigue that sleep cannot fix, a sense of disconnection even in the company of others, a quiet grief for the life we imagined but never created. To reclaim ourselves, we do not need to abandon our responsibilities or retreat into isolation. What we need is to create room for stillness, even in the midst of our busy lives. We need to pause, not just when we are exhausted, but regularly, as a form of care, not collapse. Stillness is not idleness. It is a form of listening. It is the practice of tuning in to what truly matters, of letting go of what is urgent but unimportant, and of remembering that we are human beings, not just human doings. You can begin today. You do not need perfect conditions. You only need to notice. Watch the sky. Feel the texture of the air. Step outside and breathe. Write a few lines about how you really feel. Speak gently to yourself, especially in moments of frustration. These are small things, but they are not insignificant. They are the roots of a deeper life, one that is not just full but fulfilling. No one else can live your life for you. You have the responsibility and the privilege of shaping it from the inside out. And that begins not with doing more, but with noticing more. Listening more. Trusting more. Not the voices of the world, but the quiet truth inside your own heart."
    #secret_message = "In the modern world, time often feels like a shrinking resource. Hours disappear, days blur into each other, and weeks pass without us truly noticing where they went. From the moment we wake up to the moment we fall asleep, our attention is pulled in a hundred directions. We answer messages, check the news, reply to emails, fulfill responsibilities, try to stay connected, try to stay informed, try to stay ahead. The rhythm of daily life becomes so dense and fast that it leaves very little room for silence, for slowness, for simply being present with ourselves. We may not even notice this shift at first. It begins gradually. A missed sunset here. A skipped meal there. A constant urge to multitask, to optimize, to squeeze more productivity out of every hour. Over time, we forget how to sit still without reaching for a screen. We forget how to breathe deeply without rushing to the next obligation. We forget how to listen to the quiet voice within us that once guided us with clarity and calm. This voice does not disappear, but it becomes harder to hear in a world that never stops talking. The danger is not in being busy. Being busy is sometimes necessary. The danger lies in being busy without awareness, in moving so quickly that we lose sight of why we started moving in the first place. We begin to live reactively instead of intentionally. We wake up tired. We work without joy. We socialize without presence. We fill every gap with noise, afraid of what we might feel in the silence. And then, slowly, we begin to feel the weight of it all. Not all at once, but in small, subtle ways. A sense of emptiness. A low hum of anxiety. A disconnect between what we do and who we are. Sometimes we tell ourselves this is just how life is. Everyone is busy. Everyone is tired. Everyone is overwhelmed. But that explanation is not a solution. It is a warning. It tells us that something in our collective way of living is off-balance. And when something is off-balance, we must pause. We must step back. We must ask different questions. Not just what do I need to get done today, but what kind of life am I building through these days. Not just how can I be more efficient, but what am I sacrificing in the name of efficiency. Not just how can I do more, but what would happen if I did less and felt more. Slowing down does not mean abandoning our goals or escaping from our responsibilities. It means approaching life with more depth, more attention, and more heart. It means giving ourselves permission to rest without guilt. To reflect without rushing to conclusions. To enjoy something simply because it is beautiful, not because it is useful. This kind of living is not less productive. In fact, it is often the foundation of truly meaningful work and real creativity. A mind that is well-rested can solve problems more wisely. A heart that is nourished can connect more deeply. A life that is balanced can sustain itself with integrity over time. Stillness is not a luxury. It is a necessity. And it is available to us even in small, ordinary ways. You do not need to leave the city, change your job, or delete all your apps to begin. You can start right where you are. You can take a few minutes each day to do nothing except breathe. You can walk more slowly, eat more mindfully, speak more gently. You can put your phone away during a conversation and give someone your full attention. You can write down your thoughts at the end of the day, not to analyze them, but to meet yourself honestly on the page. These practices may seem small, but they have a powerful effect. They help you return to your own rhythm. They help you remember what it feels like to be fully alive, not just functioning. As you begin to reconnect with this slower rhythm, you may notice that some things begin to shift. You become more aware of what drains your energy and what restores it. You notice how some people uplift you and others leave you feeling heavy. You start to care less about what looks good from the outside and more about what feels true on the inside. You begin to make choices that are less about impressing others and more about respecting yourself. This is not always easy. It may require saying no when you are used to saying yes. It may require walking away from habits that kept you comfortable but unfulfilled. It may mean facing emotions you once buried beneath distraction. But it is worth it, because the life that comes out of this process is one that belongs fully to you. There is something deeply healing about living in alignment with your inner truth. You stop needing to explain yourself so much. You stop comparing your path to others. You begin to trust that what is right for you will unfold in its own time, and that you do not need to force it. You begin to take deeper breaths, not just physically but emotionally. You begin to move through the world with a quieter confidence, not based on perfection or performance, but based on presence. Presence is not flashy. It does not demand attention. But it transforms everything it touches. A present parent raises a secure child. A present friend offers real comfort. A present life does not need to be extraordinary to be meaningful. There will still be challenges. Life will still ask a lot of you. The difference is that now you are no longer running on empty. You are no longer hiding from yourself. You are rooted in something real. And that root can hold you through whatever storms may come. You may still get tired, but you will know how to rest. You may still feel lost at times, but you will know how to listen. You may still stumble, but you will rise with more grace and less fear. This is not a one-time transformation. It is a daily practice. A moment-by-moment choice to return, to reconnect, to remember. Remember that your worth is not measured by your output. Remember that your body needs care, not constant pressure. Remember that your heart deserves kindness, even when it is hurting. Remember that your soul speaks in quiet ways, and that it is always trying to guide you home. No one else can make these choices for you. No one else can feel what you feel, dream what you dream, or become who you are meant to become. That is your path. And it begins not with doing more, but with being more. Being more present. Being more honest. Being more compassionate with yourself and with others. The world does not need more speed. It needs more soul. And you, simply by slowing down and being real, can offer exactly that."
    secret_message = "In a world that never stops moving, we often forget that we are allowed to slow down. The modern landscape is filled with signals telling us to go faster, achieve more, and stay constantly available. We are encouraged to maximize every moment, to make use of every second, to multitask, to push forward without pause. There is so much emphasis on growth and ambition that rest has become a foreign concept, and presence has become a rare experience. We live surrounded by screens and noise, and yet often feel lonelier than ever before. We communicate constantly, yet many of us rarely feel truly heard. We succeed publicly, but struggle privately. We are taught how to win, but not how to rest, how to perform, but not how to be. And yet, something inside us remembers. Some part of us, even when buried beneath years of pressure and expectation, still knows that life was never meant to be a race. It was meant to be a journey. A personal, intricate unfolding. A space for feeling, not just functioning. A rhythm, not a rush. That part of us is quiet. It speaks not in commands, but in questions. It wonders where the joy went. It asks when we last felt truly alive. It longs for space to breathe, to create, to simply exist without needing to prove anything. That voice can be hard to hear when we are caught in the machinery of modern life. But it never disappears. It waits. And it grows stronger when we stop and listen. Listening to ourselves has become a radical act. In a world that constantly pulls us outward, turning our attention inward is often misunderstood. People might say we are selfish, unmotivated, lazy, or too sensitive. But the truth is that self-awareness is not self-indulgence. It is a form of self-responsibility. When we begin to pay attention to what we are feeling, what we are avoiding, what we are carrying, we begin to move through life with more wisdom and care. We stop projecting our pain onto others. We stop chasing things we do not truly want. We start making choices that are aligned with our deeper values, not just with what is popular or expected. This kind of inner work does not happen quickly. It is not glamorous. It rarely brings applause. But it brings peace. And peace is something that cannot be bought, rushed, or faked. It is something that is cultivated through presence. Through honesty. Through the willingness to face discomfort rather than escape it. It comes when we stop numbing ourselves with busyness and begin to sit with the truth of our own lives. That truth might include grief, regret, longing, or confusion. But it also includes beauty, courage, clarity, and compassion. The full range of human experience lives inside us. And when we deny any part of it, we flatten our lives into something small and mechanical. Many people go through years of their lives feeling disconnected from themselves. They feel like they are living on autopilot, doing what they are supposed to do, checking all the boxes, but not really feeling much of anything. Sometimes it takes a breakdown, a loss, or a sudden change to wake them up. But sometimes, it just takes a moment of stillness. A moment when everything slows down, and they hear something inside them say, I want more than this. I want to feel again. I want to live differently. That moment is sacred. It is the beginning of something new. Not necessarily a change in circumstances, but a change in direction. A change in attention. A return to the center. The center is where life begins again. It is the space within us that is not defined by roles, titles, achievements, or appearance. It is the part of us that simply is. That watches. That feels. That remembers who we were before the world told us who to be. When we reconnect with that space, even for a few minutes a day, we begin to heal. We begin to understand ourselves more deeply. We begin to see through the noise and find what is essential. And from there, we can start to build lives that are not just successful by external standards, but meaningful by internal ones. To live this way does not mean escaping responsibility. It means approaching responsibility with intention. It means showing up for your life from a place of presence rather than pressure. You can still work, create, love, and contribute. But you do so from a place that is rooted in who you are, not who you think you have to be. You no longer chase approval because you are already grounded in your own integrity. You no longer need constant distraction because you are no longer afraid of your own thoughts. You become someone who moves through the world with a quiet strength, not needing to be the loudest voice, because your voice is connected to something real. This journey is not about perfection. It is about practice. There will be days when you forget, when you fall back into old habits, when the world feels louder than your own intuition. That is okay. This is not a path of endless progress. It is a spiral. You will return to the same lessons, the same questions, but each time from a slightly deeper place. You will grow not in a straight line, but in layers. And every time you return to yourself, even after drifting far away, you strengthen the trust you have in your own path. You are allowed to grow slowly. You are allowed to rest often. You are allowed to protect your energy. You are allowed to want a life that feels gentle instead of rushed. You are allowed to say no to what drains you and yes to what brings you peace. These are not signs of weakness. They are signs of wisdom. In a world that glorifies overwork, choosing rest is an act of rebellion. In a culture that prizes speed, choosing presence is a revolution. And in a time where everyone is expected to know everything, choosing humility and curiosity is a gift. Presence is not a skill you master once. It is a relationship you return to every day. Sometimes you will feel deeply connected to the moment. Other times you will feel scattered and restless. But the point is not to be perfect. The point is to notice. To notice when you are here. To notice when you are not. To keep coming back, gently, without judgment. Just as the ocean always returns to shore, you can always return to yourself. The way you treat yourself in these moments matters. Speak kindly. Move slowly. Let yourself be human. Let yourself be tired. Let yourself be joyful without explanation. Let yourself grieve without apology. This is how you build a life that honors the whole of who you are. Not just the parts that are easy to share, but the parts that are raw, vulnerable, and real. In this quiet, honest living, you may find something you did not know you were looking for. A deeper sense of belonging. A deeper connection to others. A greater appreciation for the little things. The way light moves across a room. The sound of your own laughter. The steadiness of your breath when you finally feel safe. These are the treasures that fast living cannot offer. These are the rewards of presence. And they are always available, if we are willing to slow down and receive them. You are not behind. You are not late. You are not broken. You are becoming. And the pace of becoming is not a race. It is a rhythm. One that is unique to you. Trust it. Trust the pauses. Trust the stillness. Trust that even when nothing seems to be happening on the outside, something meaningful is unfolding within. Growth is not always visible. Healing is not always loud. But both are sacred. And both are happening, even now. Keep returning. Keep listening. Keep softening. Keep showing up for your life, not just as something to survive, but as something to cherish. The world may never slow down, but you can. And in doing so, you remind others that they can too. Your presence is powerful. Your peace is revolutionary. Your life is yours to shape, one quiet, conscious moment at a time."
    x0, r = 0.6, 3.9

    # 1. Convert text to bits
    binary = text_to_bits(secret_message)
    print("binary[:32]:", binary[:32])
    print("len(binary):", len(binary))

    # 2. Chaotic encryption
    encrypted = chaotic_encrypt(binary, x0, r)
    print("encrypted[:32]:", encrypted[:32])

    # 3. Embed bits into image
    stego_path = embed_wf5_full(cover_image, encrypted, output_path="../images/output/wf5_method/stego_wf5_4.png")

    # 4. Extract bits from stego image
    extracted = extract_wf5_full(stego_path, len(binary))
    print("extracted[:32]:", extracted[:32])
    print("len(extracted):", len(extracted))

    # 5. Chaotic decryption
    decrypted = chaotic_decrypt(extracted, x0, r)
    print("decrypted[:32]:", decrypted[:32])

    # 6. Convert bits to text
    recovered = bits_to_text(decrypted)
    print("1. Recovered message:", recovered)
    print("2. Recovery success:", recovered == secret_message)

    # 7. Evaluate
    original = cv2.imread(cover_image)
    stego = cv2.imread(stego_path)
    print("3. Metrics:")
    print("- PSNR:", psnr(original, stego))
    print("- SSIM:", ssim_score(original, stego))
    dr = compute_detection_rate(binary, decrypted)
    print(f"- Detection Rate (DR): {dr:.4f}")

    er, max_capacity = compute_embedding_rate(len(binary), len(original.flatten()))
    print(f"- Embedding Rate (ER): {er:.4f} ({len(binary)} / {max_capacity} bits)")

if __name__ == "__main__":
    main()
