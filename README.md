# WF5 + Chaotic Encryption Steganography

A novel steganography implementation combining WF5 watermarking algorithm with chaotic encryption for enhanced security and data hiding capabilities.

## Project Structure

```
WF5 Method Steganography/
├── src/                    # Main source code
│   ├── wf5.py            # WF5 + Chaotic encryption core
│   ├── attack.py          # Attack analysis (secondary)
│   ├── dct.py             # DCT implementation (empty)
│   ├── lsb.py             # LSB implementation (empty)
│   └── f5.py              # F5 implementation (empty)
├── images/                # Input and output images
│   ├── input/             # Original images
│   └── output/            # Result images
├── docs/                  # Documentation
│   ├── chaos_stego_uml   # UML diagram
│   ├── Flowchar.drawio    # Flowchart diagram
│   ├── wf5.txt           # WF5 documentation
│   └── *.pdf             # Research papers
├── requirements.txt       # Dependencies
└── README.md             # This guide
```

## Installation

1. **Clone repository:**
```bash
git clone <repository-url>
cd Code
```

2. **Create virtual environment:**
```bash
python -m venv .venv
```

3. **Activate virtual environment:**
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Core Implementation

### WF5 + Chaotic Encryption Method

This project implements a hybrid approach combining:

#### 1. WF5 Watermarking Algorithm
- **DCT-based embedding**: Uses Discrete Cosine Transform for frequency domain embedding
- **Error correction**: Hamming(9,4) code for robust data transmission
- **Matrix encoding**: Efficient bit embedding with minimal distortion

#### 2. Chaotic Encryption Layer
- **Logistic Map**: x(n+1) = r × x(n) × (1 - x(n))
- **Parameter Space**: x0 ∈ [0.1, 0.9], r ∈ [3.57, 4.0]
- **Key Generation**: Chaotic sequences for data scrambling
- **Security Enhancement**: Large key space with high sensitivity

### Implementation Details

#### Embedding Process:
1. **Data Preparation**: Convert message to binary and apply chaotic encryption
2. **DCT Transformation**: Convert image to frequency domain
3. **WF5 Embedding**: Apply WF5 algorithm with Hamming(9,4) codes
4. **Inverse DCT**: Convert back to spatial domain

#### Extraction Process:
1. **DCT Analysis**: Extract frequency coefficients
2. **WF5 Decoding**: Apply WF5 extraction with Hamming(9,4) error correction
3. **Chaotic Decryption**: Use chaotic key to decrypt data
4. **Message Recovery**: Convert binary back to original message

## Main Features

### Primary Features (Core Implementation):
- **WF5 Algorithm**: Advanced watermarking with error correction
- **Chaotic Encryption**: Logistic map-based data scrambling
- **Hybrid Security**: Combined approach for enhanced protection
- **DCT Processing**: Frequency domain information hiding

### Secondary Features (Analysis Tools):
- **Attack Analysis**: Statistical detection methods
- **Brute-force Testing**: Chaotic key recovery attempts
- **Performance Evaluation**: Quality and security metrics

## Usage

### Main Implementation (WF5 + Chaotic):
```bash
cd src
python wf5.py
```

### Attack Analysis (Secondary):
```bash
cd src
python attack.py
```

### Future Implementations:
```bash
# DCT-based steganography (planned)
python dct.py

# LSB steganography (planned)
python lsb.py

# F5 steganography (planned)
python f5.py
```

## Configuration

### Core Parameters:
- **Chaotic Parameters**: x0 (initial value), r (growth rate)
- **WF5 Settings**: Hamming(9,4) code configuration, embedding strength
- **DCT Parameters**: Block size, coefficient selection

### Supported Formats:
- **Input**: PNG, JPG, JPEG images
- **Output**: Steganographic images with hidden data
- **Data**: Text messages, binary files

## Technical Architecture

### WF5 Algorithm Components:
- **Matrix Encoding**: Efficient bit embedding
- **Hamming(9,4) Codes**: Error detection and correction
- **DCT Coefficients**: Frequency domain manipulation

### Chaotic Encryption Features:
- **Logistic Map**: Nonlinear chaotic system
- **Parameter Sensitivity**: High dependence on initial conditions
- **Key Space**: Large parameter space for security
- **Randomness**: True chaotic behavior for encryption

### Integration Strategy:
- **Pre-processing**: Chaotic encryption before embedding
- **Embedding**: WF5 algorithm with DCT
- **Post-processing**: Quality preservation and validation

## Implementation Results

### Core Performance:
- **Embedding Success**: Reliable data hiding with WF5
- **Security Enhancement**: Chaotic encryption adds protection layer
- **Quality Preservation**: Minimal visual distortion
- **Error Correction**: Hamming codes ensure data integrity

### Secondary Analysis:
- **Attack Resistance**: Statistical analysis capabilities
- **Key Recovery**: Brute-force testing framework
- **Performance Metrics**: Quality and security evaluation

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError:**
```bash
pip install -r requirements.txt
```

2. **File not found:**
- Check image paths in `images/`
- Ensure files exist and have read permissions

3. **Chaotic parameters:**
- Verify parameter ranges (x0: 0.1-0.9, r: 3.57-4.0)
- Check for chaotic behavior validation

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Steganography Techniques](https://en.wikipedia.org/wiki/Steganography)
- [Chaotic Encryption](https://en.wikipedia.org/wiki/Chaos_theory)
- [DCT Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform)
- [WF5 Algorithm](https://en.wikipedia.org/wiki/F5_steganography)

## Authors

Project developed for Cryptography course at ITMO University.

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For questions and support, please contact the development team or create an issue in the repository. 