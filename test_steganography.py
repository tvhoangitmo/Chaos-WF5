#!/usr/bin/env python3
"""
Comprehensive test script for LSB, DCT, and F5 steganography algorithms
"""

import os
import sys
import time
import numpy as np
import cv2
from math import log10
from skimage.metrics import structural_similarity as ssim

# Add src directory to path
sys.path.append('src')

# Import steganography modules
from lsb import embed_lsb, extract_lsb, text_to_bits, bits_to_text
from dct import embed_dct, extract_dct
from f5 import embed_f5, extract_f5
from wf5 import embed_wf5, extract_wf5

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
    max_capacity = image_size_in_pixels - 32  # Subtract 32 bits for length
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

def test_algorithm(algorithm_name, embed_func, extract_func, cover_image, secret_message, **kwargs):
    """Test a steganography algorithm and return performance metrics"""
    print(f"\n{'='*60}")
    print(f"Testing {algorithm_name} Algorithm with Chaotic Encryption")
    print(f"{'='*60}")
    
    # Convert message to bits for analysis
    binary_message = text_to_bits(secret_message)
    print(f"Cover image: {cover_image}")
    print(f"Secret message: {secret_message}")
    print(f"Message length: {len(secret_message)} characters")
    print(f"Binary length: {len(binary_message)} bits")
    
    # Extract chaotic parameters if provided
    x0 = kwargs.get('x0', 0.6)
    r = kwargs.get('r', 3.9)
    print(f"Chaotic parameters: x0={x0}, r={r}")
    
    try:
        # Measure embedding time
        start_time = time.time()
        stego_path = embed_func(cover_image, secret_message, **kwargs)
        embed_time = time.time() - start_time
        
        print(f"Stego image saved: {stego_path}")
        print(f"Embedding time: {embed_time:.4f} seconds")
        
        # Measure extraction time
        start_time = time.time()
        if algorithm_name == "DCT":
            extracted_message = extract_func(stego_path, len(binary_message), x0=x0, r=r)
        elif algorithm_name == "F5":
            extracted_message = extract_func(stego_path, len(binary_message), k=3, x0=x0, r=r)
        elif algorithm_name == "WF5":
            extracted_message = extract_func(stego_path, len(binary_message), x0=x0, r=r)
        else:  # LSB
            extracted_message = extract_func(stego_path, x0=x0, r=r)
        extract_time = time.time() - start_time
        
        print(f"Extracted message: {extracted_message}")
        print(f"Recovery success: {extracted_message == secret_message}")
        print(f"Extraction time: {extract_time:.4f} seconds")
        
        # Calculate performance metrics
        original = cv2.imread(cover_image)
        stego = cv2.imread(stego_path)
        
        psnr_value = psnr(original, stego)
        ssim_value = ssim_score(original, stego)
        dr = compute_detection_rate(binary_message, text_to_bits(extracted_message))
        er, max_capacity = compute_embedding_rate(len(binary_message), len(original.flatten()))
        
        print(f"\nPerformance Metrics:")
        print(f"  PSNR: {psnr_value:.2f} dB")
        print(f"  SSIM: {ssim_value:.4f}")
        print(f"  Detection Rate (DR): {dr:.4f}")
        print(f"  Embedding Rate (ER): {er:.4f} ({len(binary_message)} / {max_capacity} bits)")
        
        return {
            'algorithm': algorithm_name,
            'success': extracted_message == secret_message,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'detection_rate': dr,
            'embedding_rate': er,
            'embed_time': embed_time,
            'extract_time': extract_time,
            'message_length': len(secret_message),
            'binary_length': len(binary_message)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'algorithm': algorithm_name,
            'success': False,
            'error': str(e)
        }

def compare_algorithms():
    """Compare all three steganography algorithms"""
    print("STEGANOGRAPHY ALGORITHMS COMPARISON")
    print("="*60)
    
    # Test parameters
    cover_images = [
        "images/input/cover_1.jpg",
        "images/input/cover_2.png", 
        "images/input/cover_3.jpg",
        "images/input/cover_4.png"
    ]
    
    # secret_messages = [
    #     "Hello, this is a test message for steganography!",
    #     "Steganography is the art of hiding information in plain sight.",
    #     "This message demonstrates the effectiveness of different embedding techniques.",
    #     "The quick brown fox jumps over the lazy dog. 1234567890!@#$%^&*()"
    # ]
    secret_messages = [
    "In a world that never stops moving, we often forget that we are allowed to slow down. The modern landscape is filled with signals telling us to go faster, achieve more, and stay constantly available. We are encouraged to maximize every moment, to make use of every second, to multitask, to push forward without pause. There is so much emphasis on growth and ambition that rest has become a foreign concept, and presence has become a rare experience. We live surrounded by screens and noise, and yet often feel lonelier than ever before. We communicate constantly, yet many of us rarely feel truly heard. We succeed publicly, but struggle privately. We are taught how to win, but not how to rest, how to perform, but not how to be. And yet, something inside us remembers. Some part of us, even when buried beneath years of pressure and expectation, still knows that life was never meant to be a race. It was meant to be a journey. A personal, intricate unfolding. A space for feeling, not just functioning. A rhythm, not a rush. That part of us is quiet. It speaks not in commands, but in questions. It wonders where the joy went. It asks when we last felt truly alive. It longs for space to breathe, to create, to simply exist without needing to prove anything. That voice can be hard to hear when we are caught in the machinery of modern life. But it never disappears. It waits. And it grows stronger when we stop and listen. Listening to ourselves has become a radical act. In a world that constantly pulls us outward, turning our attention inward is often misunderstood. People might say we are selfish, unmotivated, lazy, or too sensitive. But the truth is that self-awareness is not self-indulgence. It is a form of self-responsibility. When we begin to pay attention to what we are feeling, what we are avoiding, what we are carrying, we begin to move through life with more wisdom and care. We stop projecting our pain onto others. We stop chasing things we do not truly want. We start making choices that are aligned with our deeper values, not just with what is popular or expected. This kind of inner work does not happen quickly. It is not glamorous. It rarely brings applause. But it brings peace. And peace is something that cannot be bought, rushed, or faked. It is something that is cultivated through presence. Through honesty. Through the willingness to face discomfort rather than escape it. It comes when we stop numbing ourselves with busyness and begin to sit with the truth of our own lives. That truth might include grief, regret, longing, or confusion. But it also includes beauty, courage, clarity, and compassion. The full range of human experience lives inside us. And when we deny any part of it, we flatten our lives into something small and mechanical. Many people go through years of their lives feeling disconnected from themselves. They feel like they are living on autopilot, doing what they are supposed to do, checking all the boxes, but not really feeling much of anything. Sometimes it takes a breakdown, a loss, or a sudden change to wake them up. But sometimes, it just takes a moment of stillness. A moment when everything slows down, and they hear something inside them say, I want more than this. I want to feel again. I want to live differently. That moment is sacred. It is the beginning of something new. Not necessarily a change in circumstances, but a change in direction. A change in attention. A return to the center. The center is where life begins again. It is the space within us that is not defined by roles, titles, achievements, or appearance. It is the part of us that simply is. That watches. That feels. That remembers who we were before the world told us who to be. When we reconnect with that space, even for a few minutes a day, we begin to heal. We begin to understand ourselves more deeply. We begin to see through the noise and find what is essential. And from there, we can start to build lives that are not just successful by external standards, but meaningful by internal ones. To live this way does not mean escaping responsibility. It means approaching responsibility with intention. It means showing up for your life from a place of presence rather than pressure. You can still work, create, love, and contribute. But you do so from a place that is rooted in who you are, not who you think you have to be. You no longer chase approval because you are already grounded in your own integrity. You no longer need constant distraction because you are no longer afraid of your own thoughts. You become someone who moves through the world with a quiet strength, not needing to be the loudest voice, because your voice is connected to something real. This journey is not about perfection. It is about practice. There will be days when you forget, when you fall back into old habits, when the world feels louder than your own intuition. That is okay. This is not a path of endless progress. It is a spiral. You will return to the same lessons, the same questions, but each time from a slightly deeper place. You will grow not in a straight line, but in layers. And every time you return to yourself, even after drifting far away, you strengthen the trust you have in your own path. You are allowed to grow slowly. You are allowed to rest often. You are allowed to protect your energy. You are allowed to want a life that feels gentle instead of rushed. You are allowed to say no to what drains you and yes to what brings you peace. These are not signs of weakness. They are signs of wisdom. In a world that glorifies overwork, choosing rest is an act of rebellion. In a culture that prizes speed, choosing presence is a revolution. And in a time where everyone is expected to know everything, choosing humility and curiosity is a gift. Presence is not a skill you master once. It is a relationship you return to every day. Sometimes you will feel deeply connected to the moment. Other times you will feel scattered and restless. But the point is not to be perfect. The point is to notice. To notice when you are here. To notice when you are not. To keep coming back, gently, without judgment. Just as the ocean always returns to shore, you can always return to yourself. The way you treat yourself in these moments matters. Speak kindly. Move slowly. Let yourself be human. Let yourself be tired. Let yourself be joyful without explanation. Let yourself grieve without apology. This is how you build a life that honors the whole of who you are. Not just the parts that are easy to share, but the parts that are raw, vulnerable, and real. In this quiet, honest living, you may find something you did not know you were looking for. A deeper sense of belonging. A deeper connection to others. A greater appreciation for the little things. The way light moves across a room. The sound of your own laughter. The steadiness of your breath when you finally feel safe. These are the treasures that fast living cannot offer. These are the rewards of presence. And they are always available, if we are willing to slow down and receive them. You are not behind. You are not late. You are not broken. You are becoming. And the pace of becoming is not a race. It is a rhythm. One that is unique to you. Trust it. Trust the pauses. Trust the stillness. Trust that even when nothing seems to be happening on the outside, something meaningful is unfolding within. Growth is not always visible. Healing is not always loud. But both are sacred. And both are happening, even now. Keep returning. Keep listening. Keep softening. Keep showing up for your life, not just as something to survive, but as something to cherish. The world may never slow down, but you can. And in doing so, you remind others that they can too. Your presence is powerful. Your peace is revolutionary. Your life is yours to shape, one quiet, conscious moment at a time."
    ]
    
    results = []
    
    for i, (cover_image, secret_message) in enumerate(zip(cover_images, secret_messages)):
        print(f"\n\n{'#'*80}")
        print(f"TEST CASE {i+1}: {os.path.basename(cover_image)}")
        print(f"{'#'*80}")
        
        # Test LSB
        lsb_result = test_algorithm(
            "LSB", 
            embed_lsb, 
            extract_lsb, 
            cover_image, 
            secret_message,
            x0=0.6, r=3.9,
            output_path=f"images/output/lsb_method/stego_lsb_{i+1}.png"
        )
        results.append(lsb_result)
        
        # Test DCT
        dct_result = test_algorithm(
            "DCT", 
            embed_dct, 
            extract_dct, 
            cover_image, 
            secret_message,
            alpha=0.1, x0=0.6, r=3.9,
            output_path=f"images/output/dct_method/stego_dct_{i+1}.png"
        )
        results.append(dct_result)
        
        # Test F5
        f5_result = test_algorithm(
            "F5", 
            embed_f5, 
            extract_f5, 
            cover_image, 
            secret_message,
            k=3, x0=0.6, r=3.9,
            output_path=f"images/output/f5_method/stego_f5_{i+1}.png"
        )
        results.append(f5_result)
        
        # Test WF5
        wf5_result = test_algorithm(
            "WF5", 
            embed_wf5, 
            extract_wf5, 
            cover_image, 
            secret_message,
            x0=0.6, r=3.9,
            output_path=f"images/output/wf5_method/stego_wf5_{i+1}.png"
        )
        results.append(wf5_result)
    
    # Summary comparison
    print_summary(results)

def print_summary(results):
    """Print summary comparison of all algorithms"""
    print(f"\n\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    # Group results by algorithm
    algorithms = {}
    for result in results:
        if 'error' in result:
            continue
        alg = result['algorithm']
        if alg not in algorithms:
            algorithms[alg] = []
        algorithms[alg].append(result)
    
    # Calculate averages for each algorithm
    summary = {}
    for alg, alg_results in algorithms.items():
        if not alg_results:
            continue
            
        summary[alg] = {
            'success_rate': sum(1 for r in alg_results if r['success']) / len(alg_results),
            'avg_psnr': np.mean([r['psnr'] for r in alg_results]),
            'avg_ssim': np.mean([r['ssim'] for r in alg_results]),
            'avg_dr': np.mean([r['detection_rate'] for r in alg_results]),
            'avg_er': np.mean([r['embedding_rate'] for r in alg_results]),
            'avg_embed_time': np.mean([r['embed_time'] for r in alg_results]),
            'avg_extract_time': np.mean([r['extract_time'] for r in alg_results]),
            'total_tests': len(alg_results)
        }
    
    # Print comparison table
    print(f"{'Algorithm':<10} {'Success':<8} {'PSNR':<8} {'SSIM':<8} {'DR':<8} {'ER':<8} {'Embed':<8} {'Extract':<8}")
    print("-" * 80)
    
    for alg, stats in summary.items():
        print(f"{alg:<10} {stats['success_rate']:<8.2%} {stats['avg_psnr']:<8.1f} "
              f"{stats['avg_ssim']:<8.3f} {stats['avg_dr']:<8.3f} {stats['avg_er']:<8.3f} "
              f"{stats['avg_embed_time']:<8.3f} {stats['avg_extract_time']:<8.3f}")
    
    # Find best algorithm for each metric
    print(f"\n{'='*80}")
    print("BEST PERFORMANCE BY METRIC")
    print(f"{'='*80}")
    
    if summary:
        best_psnr = max(summary.items(), key=lambda x: x[1]['avg_psnr'])
        best_ssim = max(summary.items(), key=lambda x: x[1]['avg_ssim'])
        best_dr = max(summary.items(), key=lambda x: x[1]['avg_dr'])
        best_er = max(summary.items(), key=lambda x: x[1]['avg_er'])
        fastest_embed = min(summary.items(), key=lambda x: x[1]['avg_embed_time'])
        fastest_extract = min(summary.items(), key=lambda x: x[1]['avg_extract_time'])
        
        print(f"Best PSNR: {best_psnr[0]} ({best_psnr[1]['avg_psnr']:.1f} dB)")
        print(f"Best SSIM: {best_ssim[0]} ({best_ssim[1]['avg_ssim']:.3f})")
        print(f"Best Detection Rate: {best_dr[0]} ({best_dr[1]['avg_dr']:.3f})")
        print(f"Best Embedding Rate: {best_er[0]} ({best_er[1]['avg_er']:.3f})")
        print(f"Fastest Embedding: {fastest_embed[0]} ({fastest_embed[1]['avg_embed_time']:.3f}s)")
        print(f"Fastest Extraction: {fastest_extract[0]} ({fastest_extract[1]['avg_extract_time']:.3f}s)")

def create_output_directories():
    """Create output directories if they don't exist"""
    directories = [
        "images/output/lsb_method",
        "images/output/dct_method", 
        "images/output/f5_method",
        "images/output/wf5_method"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main function"""
    print("Steganography Algorithms Implementation with Chaotic Encryption")
    print("="*60)
    print("This script tests LSB, DCT, F5, and WF5 steganography algorithms")
    print("with chaotic encryption and compares their performance metrics.")
    
    # Create output directories
    create_output_directories()
    
    # Run comparison
    compare_algorithms()
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETED")
    print(f"{'='*80}")
    print("Check the output directories for generated stego images:")
    print("- images/output/lsb_method/")
    print("- images/output/dct_method/")
    print("- images/output/f5_method/")
    print("- images/output/wf5_method/")

if __name__ == "__main__":
    main()

