#FZB297@mocs.utc.edu
#UTC Wireless Sensing Group
#UTC Department of Electrical Engineering
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. SETUP PARAMETERS
# ==========================================
K = 64              # 64 Subcarriers (IEEE 802.11a standard)
active_sc = 52      # 52 Usable Subcarriers
cp_length = 16      # 16-sample Cyclic Prefix
m_bits = 16         # 16-bit blocks
n_samples = 100000  #Number of Samples
epochs = 100        # Number of training epochs
base_path = '/home/coolp/OFDM_Simulation_Pipeline/' #Might have to change to your project file 

if not os.path.exists(base_path):
    os.makedirs(base_path)

# ==========================================
# 2. THE MODELS (Alice & Bob)
# ==========================================
alice = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(m_bits,)),
    tf.keras.layers.Dense(128, activation='tanh'),
    # Output is now 104 (52 Real + 52 Imaginary)
    tf.keras.layers.Dense(active_sc * 2, activation='tanh', name="Alice_Mapper")
], name="Alice_TX")

bob = tf.keras.Sequential([
    # Input is now 104 (52 Real + 52 Imaginary)
    tf.keras.layers.Input(shape=(active_sc * 2,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(m_bits, activation='sigmoid', name="Bob_Estimator")
], name="Bob_RX")

# ==========================================
# 3. LINK THE SYSTEM (TF Graph)
# ==========================================
input_bits = tf.keras.Input(shape=(m_bits,))

# Alice maps the bits to 104 flat values
tx_iq_flat = alice(input_bits) 

# Convert 104 floats to 52 Complex Active Subcarriers
real_part = tx_iq_flat[:, :active_sc]
imag_part = tx_iq_flat[:, active_sc:]
active_complex = tf.complex(real_part, imag_part)

# --- IEEE 802.11a SUBCARRIER ALLOCATION ---
left_active = active_complex[:, :26]
right_active = active_complex[:, 26:]
batch_size = tf.shape(active_complex)[0]
dc_null = tf.zeros((batch_size, 1), dtype=tf.complex64)
guard_bands = tf.zeros((batch_size, 11), dtype=tf.complex64)

# Assembling the 64 OFDM bins: [DC(1) + Left(26) + Guards(11) + Right(26)]
complex_subcarriers = tf.concat([dc_null, left_active, guard_bands, right_active], axis=1)

# --- IFFT & CYCLIC PREFIX ---
ofdm_time_signal = tf.signal.ifft(complex_subcarriers)
cp = ofdm_time_signal[:, -cp_length:]
tx_time_cp = tf.concat([cp, ofdm_time_signal], axis=1)

# --- THE AIR (Adding Gaussian Noise) ---
noise_std = 0.18
noise_r = tf.keras.layers.GaussianNoise(noise_std)(tf.math.real(tx_time_cp))
noise_i = tf.keras.layers.GaussianNoise(noise_std)(tf.math.imag(tx_time_cp))
rx_time_signal = tf.complex(noise_r, noise_i)

# --- RECEIVER: REMOVE CP & FFT ---
rx_time = rx_time_signal[:, cp_length:]
rx_subcarriers = tf.signal.fft(rx_time)

# --- EXTRACT THE ACTIVE SUBCARRIERS ---
rx_left = rx_subcarriers[:, 1:27]
rx_right = rx_subcarriers[:, 38:64]
rx_active = tf.concat([rx_left, rx_right], axis=1)

# Flatten back to 104 for Bob (RX)
rx_flat = tf.concat([tf.math.real(rx_active), tf.math.imag(rx_active)], axis=1)
bit_estimate = bob(rx_flat)

poc_system = tf.keras.Model(inputs=input_bits, outputs=bit_estimate)
poc_system.compile(optimizer='adam', loss='binary_crossentropy')

# ==========================================
# 4. TRAIN 
# ==========================================
x_train = np.random.randint(0, 2, size=(n_samples, m_bits)).astype('float32')
print(f"\n--- Phase 1: Training NN-OFDM for {epochs} Epochs ---")
history = poc_system.fit(x_train, x_train, epochs=epochs, batch_size=128, verbose=1)

# ==========================================
#  POWER SPECTRAL DENSITY CHECK
# ==========================================
print("\n--- Generating PSD Plot (Close window to continue) ---")
test_bits_psd = np.random.randint(0, 2, size=(10000, m_bits)).astype('float32')
tx_iq_flat_psd = alice.predict(test_bits_psd, verbose=0)
active_comp_psd = tx_iq_flat_psd[:, :52] + 1j * tx_iq_flat_psd[:, 52:]
left_act_psd = active_comp_psd[:, :26]
right_act_psd = active_comp_psd[:, 26:]
dc_np_psd = np.zeros((10000, 1), dtype=complex)
guards_np_psd = np.zeros((10000, 11), dtype=complex)
complex_sc_psd = np.hstack([dc_np_psd, left_act_psd, guards_np_psd, right_act_psd])

ofdm_wave_psd = np.fft.ifft(complex_sc_psd, axis=1)
cp_np_psd = ofdm_wave_psd[:, -cp_length:]
tx_wave_cp_psd = np.hstack([cp_np_psd, ofdm_wave_psd])

plt.figure(figsize=(10, 5))
# Flattening the array to average it, and turning off the window to expose the DC null
plt.psd(tx_wave_cp_psd.flatten(), NFFT=64, Fs=64, color='cyan', window=plt.mlab.window_none) 
plt.title("Power Spectral Density of the OFDM Signal")
plt.xlabel("Subcarrier Index (Frequency)")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.grid(True)

# Save first (with underscores), then show
plt.savefig('Power_Spectral_Density_New_TX_RX.png')
plt.show()

# ==========================================
# GENERATE THE PREAMBLE (IEEE 802.11a LTF)
# ==========================================
print("\n--- Generating IEEE 802.11a LTF Preamble ---")
test_size = 10000

# The official IEEE 802.11a Long Training Field (LTF) sequence
ltf_sequence = np.array([
    1,  1, -1, -1,  1,  1, -1,  1, -1,  1,  1,  1, 
    1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1,  1,  1,  1, 
    1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1, 
   -1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1
])
preamble_bpsk = ltf_sequence.reshape(1, 52) + 0j

# Allocate the preamble to the 64-lane OFDM highway
left_preamble = preamble_bpsk[:, :26]
right_preamble = preamble_bpsk[:, 26:]
dc_pre = np.zeros((1, 1), dtype=complex)
guards_pre = np.zeros((1, 11), dtype=complex)

complex_preamble = np.hstack([dc_pre, left_preamble, guards_pre, right_preamble])

# Push the Preamble through IFFT and add its own Cyclic Prefix
ofdm_preamble = np.fft.ifft(complex_preamble, axis=1)
cp_preamble = ofdm_preamble[:, -cp_length:]
preamble_time_wave = np.hstack([cp_preamble, ofdm_preamble])

# Replicate the 80-sample preamble for the entire batch size
preamble_batch = np.repeat(preamble_time_wave, test_size, axis=0)

# ==========================================
# 5. THE RESULTS SWEEP ( With Echoes/Rayleigh Fading & Preamble)
# ==========================================
print("\n--- Phase 2: OFDM BER & BLER Sweep ---")
snr_range = np.arange(2, 25, 3)
test_bits = np.random.randint(0, 2, size=(test_size, m_bits)).astype('float32')

snr_results, ber_results, bler_results = [], [], []
print(f"{'SNR (dB)':<10} | {'BER':<10} | {'BLER':<10}")
print("-" * 35)

for snr_db in snr_range:
    # 1. Alice Mapper
    tx_iq_flat = alice.predict(test_bits, verbose=0)
    
    # 2. Subcarrier Allocation
    active_comp = tx_iq_flat[:, :52] + 1j * tx_iq_flat[:, 52:]
    left_act = active_comp[:, :26]
    right_act = active_comp[:, 26:]
    
    dc_np = np.zeros((test_size, 1), dtype=complex)
    guards_np = np.zeros((test_size, 11), dtype=complex)
    
    complex_sc = np.hstack([dc_np, left_act, guards_np, right_act])
    
    # 3. IFFT & Add CP
    ofdm_wave = np.fft.ifft(complex_sc, axis=1)
    cp_np = ofdm_wave[:, -cp_length:]
    tx_wave_cp = np.hstack([cp_np, ofdm_wave])
    
    # ADD THE PREAMBLE TO THE FRONT OF THE DATA (Time Domain Concatenation)
    # The physical frame is now 160 samples long (80 preamble + 80 data)
    full_tx_frame = np.hstack([preamble_batch, tx_wave_cp])
    
    # 4. Noise Channel (AWGN) attacks the entire 160-sample frame
    sigma = np.sqrt(1 / (10**(snr_db / 10)))
    noise = (np.random.normal(0, sigma, full_tx_frame.shape) + 
             1j * np.random.normal(0, sigma, full_tx_frame.shape)) / np.sqrt(2)
             
    # 5. Multipath Echo/Rayleigh Fading - attacks the entire 160-sample frame
    delay = 5 
    echo = np.zeros_like(full_tx_frame)
    echo[:, delay:] = full_tx_frame[:, :-delay] * 0.5
    
    # Add everything together in the air for RX
    rx_frame_cp = full_tx_frame + echo + noise
    
    # 6. Remove Preamble & Remove Data CP
    # SIMULATING AUTOCORRELATION: Slice off the first 80 samples (the preamble)
    # Then slice off the next 16 samples (the data's Cyclic Prefix)
    rx_wave = rx_frame_cp[:, 80 + cp_length:] 
    rx_sc = np.fft.fft(rx_wave, axis=1)
    
    # 7. Extract Active Bins
    rx_left = rx_sc[:, 1:27]
    rx_right = rx_sc[:, 38:64]
    rx_active = np.hstack([rx_left, rx_right])
    
    # 8. Bob Estimator
    rx_flat = np.hstack([np.real(rx_active), np.imag(rx_active)])
    preds = (bob.predict(rx_flat, verbose=0) > 0.5).astype(float)
    
    # Stats/Key Metrics
    ber = np.sum(test_bits != preds) / (test_size * m_bits) #Bit Error Rate
    bler = np.sum(np.any(test_bits != preds, axis=1)) / test_size #Block Error Rate
    
    snr_results.append(snr_db)
    ber_results.append(ber)
    bler_results.append(bler)
    print(f"{snr_db:<10} | {ber:<10.5f} | {bler:<10.5f}")

print("\nArchitecture Complete: Guard Bands, DC Null, PSD, and IEEE 802.11a Preamble Active.")