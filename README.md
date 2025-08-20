# Steganography Algorithms Implementation with Chaotic Encryption

This project implements three popular steganography algorithms with chaotic encryption for hiding secret messages in digital images:

1. **LSB (Least Significant Bit)** - Spatial domain steganography with chaotic encryption
2. **DCT (Discrete Cosine Transform)** - Frequency domain steganography with chaotic encryption
3. **F5** - Advanced JPEG steganography with matrix encoding and chaotic encryption
4. **WF5** - Weighted F5 steganography with chaotic encryption (existing)

## Features

- **LSB Steganography**: Embeds secret data in the least significant bits of image pixels with chaotic encryption
- **DCT Steganography**: Embeds data in DCT coefficients of 8x8 image blocks with chaotic encryption
- **F5 Steganography**: Uses matrix encoding and permutative straddling with chaotic encryption
- **WF5 Steganography**: Advanced weighted F5 with chaotic encryption and Hamming codes
- **Chaotic Encryption**: Logistic map-based encryption for enhanced security
- **Performance Metrics**: PSNR, SSIM, Detection Rate, Embedding Rate
- **Comprehensive Testing**: Automated comparison of all algorithms

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Individual Algorithm Testing

#### LSB Steganography with Chaotic Encryption
```python
from src.lsb import embed_lsb, extract_lsb

# Embed secret message with chaotic encryption
stego_path = embed_lsb("cover_image.jpg", "Secret message here", x0=0.6, r=3.9)

# Extract secret message with chaotic decryption
extracted_message = extract_lsb(stego_path, x0=0.6, r=3.9)
```

#### DCT Steganography with Chaotic Encryption
```python
from src.dct import embed_dct, extract_dct

# Embed secret message with chaotic encryption
stego_path = embed_dct("cover_image.jpg", "Secret message here", alpha=0.1, x0=0.6, r=3.9)

# Extract secret message with chaotic decryption (need to know message length in bits)
binary_message = text_to_bits("Secret message here")
extracted_message = extract_dct(stego_path, len(binary_message), x0=0.6, r=3.9)
```

#### F5 Steganography with Chaotic Encryption
```python
from src.f5 import embed_f5, extract_f5

# Embed secret message with chaotic encryption
stego_path = embed_f5("cover_image.jpg", "Secret message here", k=3, x0=0.6, r=3.9)

# Extract secret message with chaotic decryption (need to know message length in bits)
binary_message = text_to_bits("Secret message here")
extracted_message = extract_f5(stego_path, len(binary_message), k=3, x0=0.6, r=3.9)
```

### Comprehensive Testing

Run the complete test suite to compare all algorithms:

```bash
python test_steganography.py
```

This will:
- Test all three algorithms on multiple cover images
- Generate stego images in respective output directories
- Calculate and compare performance metrics
- Provide a summary comparison table

## Algorithm Details

### LSB (Least Significant Bit) with Chaotic Encryption
- **Domain**: Spatial domain
- **Method**: Replaces least significant bits of pixel values with chaotic encryption
- **Capacity**: High (1 bit per pixel)
- **Robustness**: Low (sensitive to image modifications)
- **Visibility**: Very low (minimal visual distortion)
- **Security**: Enhanced with logistic map chaotic encryption

### DCT (Discrete Cosine Transform) with Chaotic Encryption
- **Domain**: Frequency domain
- **Method**: Modifies DCT coefficients in 8x8 blocks with chaotic encryption
- **Capacity**: Medium (1 bit per 8x8 block)
- **Robustness**: Medium (resistant to some image processing)
- **Visibility**: Low (good visual quality)
- **Security**: Enhanced with logistic map chaotic encryption

### F5 with Chaotic Encryption
- **Domain**: Frequency domain (JPEG coefficients)
- **Method**: Matrix encoding with permutative straddling and chaotic encryption
- **Capacity**: Medium (k bits in 2^k-1 coefficients)
- **Robustness**: High (resistant to JPEG compression)
- **Visibility**: Low (optimized for minimal distortion)
- **Security**: Enhanced with logistic map chaotic encryption

### WF5 with Chaotic Encryption (Existing)
- **Domain**: Frequency domain with weighted embedding
- **Method**: Advanced F5 with Hamming codes and chaotic encryption
- **Capacity**: Medium (4 bits per 9 coefficients)
- **Robustness**: Very high (error correction + chaotic encryption)
- **Visibility**: Low (weighted embedding for minimal distortion)
- **Security**: Enhanced with logistic map chaotic encryption

## Performance Metrics

The implementation calculates and compares:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality
- **SSIM (Structural Similarity Index)**: Measures perceptual similarity
- **Detection Rate (DR)**: Bit accuracy of extracted message
- **Embedding Rate (ER)**: Ratio of embedded bits to maximum capacity
- **Processing Time**: Embedding and extraction speed
- **Security Level**: Enhanced by chaotic encryption parameters

## File Structure

```
Code/
├── src/
│   ├── lsb.py          # LSB steganography implementation
│   ├── dct.py          # DCT steganography implementation
│   ├── f5.py           # F5 steganography implementation
│   ├── wf5.py          # WF5 steganography (existing)
│   └── attack.py       # Steganalysis attack (existing)
├── images/
│   ├── input/          # Cover images
│   └── output/         # Generated stego images
│       ├── lsb_method/
│       ├── dct_method/
│       └── f5_method/
├── test_steganography.py  # Comprehensive test script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Output

The test script generates:
- Stego images for each algorithm and test case
- Performance comparison tables
- Summary statistics
- Best algorithm recommendations for different metrics

## Dependencies

- numpy >= 1.21.0
- opencv-python >= 4.5.0
- scikit-image >= 0.18.0
- scipy >= 1.7.0
- tqdm >= 4.62.0
- nltk >= 3.6.0
- bitarray >= 2.3.0
- matplotlib >= 3.5.0
- Pillow >= 8.3.0

## Example Output

```
STEGANOGRAPHY ALGORITHMS COMPARISON WITH CHAOTIC ENCRYPTION
============================================================

Testing LSB Algorithm with Chaotic Encryption
============================================================
Cover image: ../images/input/cover_1.jpg
Secret message: Hello, this is a test message for steganography with chaotic encryption!
Message length: 65 characters
Binary length: 520 bits
Chaotic parameters: x0=0.6, r=3.9
Stego image saved: ../images/output/lsb_method/stego_lsb_1.png
Embedding time: 0.0156 seconds
Extracted message: Hello, this is a test message for steganography with chaotic encryption!
Recovery success: True
Extraction time: 0.0102 seconds

Performance Metrics:
  PSNR: 51.23 dB
  SSIM: 0.9998
  Detection Rate (DR): 1.0000
  Embedding Rate (ER): 0.0017 (520 / 307200 bits)
```

## Contributing

Feel free to contribute improvements, bug fixes, or additional steganography algorithms.

## License

This project is for educational and research purposes. 