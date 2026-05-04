# IEEE-802.11A-WIFI
Python simulation of an IEEE 802.11a Wi-Fi link replacing standard QAM and channel equalization with deep learning.
An end-to-end Python simulation of a modified IEEE 802.11a physical layer. This architecture replaces traditional textbook components with a deep learning autoencoder framework designed to reduce multipath fading and AWGN.
## Architecture Pipeline

1. **TX Mapper (Alice):** A neural network takes in 16 raw bits and maps them onto 52 active subcarriers. Standard 802.11a Guard Bands, DC Null, and IFFT are then applied.
2. **The RF Channel:** A 16-sample Cyclic Prefix is attached to reduce/overcome Inter-Symbol Interference before the signal passes through a simulated multipath fading and standard noisy environment.
3. **RX Estimator (Bob):** The receiver removes the Cyclic Prefix, runs the FFT, and feeds the 52 frequencies into a neural network to estimate and decode the original 16 raw bits.
