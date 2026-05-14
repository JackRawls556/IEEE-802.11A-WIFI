# IEEE-802.11A-WIFI
Python simulation of an IEEE 802.11a Wi-Fi link replacing standard QAM and channel equalization with deep learning.
An end-to-end Python simulation of a modified IEEE 802.11a physical layer. This architecture replaces traditional textbook components with a deep learning autoencoder framework designed to reduce multipath fading and AWGN.
## Architecture Pipeline

1. **TX Mapper (Alice):** A neural network takes in 16 raw bits and maps them onto 48 active subcarriers.The 4 standard IEEE 802.11a pilots are then inserted to create a 52-subcarrier frame. Standard 802.11a Guard Bands, the DC Null, and the IFFT are applied.
2. **The RF Channel:** A 16-sample Cyclic Prefix is attached to reduce/overcome Inter-Symbol Interference before the signal passes through a simulated multipath fading and standard noisy environment.
3. **RX Estimator (Bob):** The receiver removes the Cyclic Prefix and runs the FFT. The Digital Signal Processing (DSP) block then strips out the 4 standard pilots, feeding only the remaining 48 data subcarriers into the neural network to estimate and decode the original 16 raw bits.
