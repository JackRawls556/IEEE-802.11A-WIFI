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
active_sc = 52      # 52 Usable Subcarriers (Total)
data_sc = 48        # 48 Data Subcarriers (AI Controlled)
pilot_sc = 4        # 4 Pilot Subcarriers (Hardcoded)
cp_length = 16      # 16-sample Cyclic Prefix
m_bits = 16         # 16-bit blocks
n_samples = 100000  # Number of Samples
epochs = 40         # Number of training epochs
base_path = '/home/coolp/OFDM_Simulation_Pipeline/' #Might have to change path depending on specific system

if not os.path.exists(base_path):
    os.makedirs(base_path)

# ==========================================
# 2. THE MODELS (Alice & Bob)
# ==========================================
alice = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(m_bits,)),
    tf.keras.layers.Dense(128, activation='tanh'),
    # Output is now 96 (48 Real + 48 Imaginary) for Data only
    tf.keras.layers.Dense(data_sc * 2, activation='tanh', name="Alice_Mapper")
], name="Alice_TX")

bob = tf.keras.Sequential([
    # Input is now 96 (48 Real + 48 Imaginary) - Pilots are stripped before this
    tf.keras.layers.Input(shape=(data_sc * 2,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(m_bits, activation='sigmoid', name="Bob_Estimator")
], name="Bob_RX")

# ==========================================
# 3. LINK THE SYSTEM (TF Graph)
# ==========================================
input_bits = tf.keras.Input(shape=(m_bits,))

# Alice maps the bits to 96 flat values
tx_iq_flat = alice(input_bits)

# Convert 96 floats to 48 Complex Data Subcarriers
real_part = tx_iq_flat[:, :data_sc]
imag_part = tx_iq_flat[:, data_sc:]
data_complex = tf.complex(real_part, imag_part)

# --- IEEE 802.11a SUBCARRIER ALLOCATION (WITH PILOTS) ---
left_data = data_complex[:, :24]
right_data = data_complex[:, 24:]
batch_size = tf.shape(data_complex)[0]

# Define the IEEE Pilots (1+0j)
pilot_tf = tf.ones((batch_size, 1), dtype=tf.complex64)

# Weave Pilots into the Left Side (-26 to -1)
left_active = tf.concat([
    left_data[:, :5],   # -26 to -22
    pilot_tf,           # -21 (PILOT)
    left_data[:, 5:18], # -20 to -8
    pilot_tf,           # -7  (PILOT)
    left_data[:, 18:]   # -6 to -1
], axis=1)

# Weave Pilots into the Right Side (+1 to +26)
right_active = tf.concat([
    right_data[:, :6],   # +1 to +6
    pilot_tf,            # +7  (PILOT)
    right_data[:, 6:19], # +8 to +20
    pilot_tf,            # +21 (PILOT)
    right_data[:, 19:]   # +22 to +26
], axis=1)

dc_null = tf.zeros((batch_size, 1), dtype=tf.complex64)
guard_bands = tf.zeros((batch_size, 11), dtype=tf.complex64)

# Assembling the 64 OFDM bins
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

# --- EXTRACT THE ACTIVE SUBCARRIERS & STRIP PILOTS ---
rx_left = rx_subcarriers[:, 1:27]
rx_right = rx_subcarriers[:, 38:64]

# Throw away the pilot bins (indices 5, 19 on left, and 6, 20 on right)
rx_left_data = tf.concat([rx_left[:, :5], rx_left[:, 6:19], rx_left[:, 20:]], axis=1)
rx_right_data = tf.concat([rx_right[:, :6], rx_right[:, 7:20], rx_right[:, 21:]], axis=1)

rx_active_data = tf.concat([rx_left_data, rx_right_data], axis=1)

# Flatten back to 96 for Bob (RX)
rx_flat = tf.concat([tf.math.real(rx_active_data), tf.math.imag(rx_active_data)], axis=1)
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
data_comp_psd = tx_iq_flat_psd[:, :48] + 1j * tx_iq_flat_psd[:, 48:]

left_data_psd = data_comp_psd[:, :24]
right_data_psd = data_comp_psd[:, 24:]
pilot_np_psd = np.ones((10000, 1), dtype=complex)

left_act_psd = np.hstack([left_data_psd[:, :5], pilot_np_psd, left_data_psd[:, 5:18], pilot_np_psd, left_data_psd[:, 18:]])
right_act_psd = np.hstack([right_data_psd[:, :6], pilot_np_psd, right_data_psd[:, 6:19], pilot_np_psd, right_data_psd[:, 19:]])

dc_np_psd = np.zeros((10000, 1), dtype=complex)
guards_np_psd = np.zeros((10000, 11), dtype=complex)
complex_sc_psd = np.hstack([dc_np_psd, left_act_psd, guards_np_psd, right_act_psd])

ofdm_wave_psd = np.fft.ifft(complex_sc_psd, axis=1)
cp_np_psd = ofdm_wave_psd[:, -cp_length:]
tx_wave_cp_psd = np.hstack([cp_np_psd, ofdm_wave_psd])

plt.figure(figsize=(10, 5))
plt.psd(tx_wave_cp_psd.flatten(), NFFT=64, Fs=64, color='cyan', window=plt.mlab.window_none)
plt.title("Power Spectral Density of the OFDM Signal (With 4 Pilots)")
plt.xlabel("Subcarrier Index (Frequency)")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.grid(True)
plt.savefig('Power_Spectral_Density_New_TX_RX.png')
# plt.show() # Uncomment if you want the script to pause and display the plot

# ==========================================
# GENERATE THE PREAMBLE (IEEE 802.11a LTF)
# ==========================================
print("\n--- Generating IEEE 802.11a LTF Preamble ---")
test_size = 10000

ltf_sequence = np.array([
    1,  1, -1, -1,  1,  1, -1,  1, -1,  1,  1,  1,
    1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1,  1,  1,  1,
    1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1,
   -1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1
])
preamble_bpsk = ltf_sequence.reshape(1, 52) + 0j
#Allocate the preamble to the 64-lane OFDM Highway
left_preamble = preamble_bpsk[:, :26]
right_preamble = preamble_bpsk[:, 26:]
dc_pre = np.zeros((1, 1), dtype=complex)
guards_pre = np.zeros((1, 11), dtype=complex)

complex_preamble = np.hstack([dc_pre, left_preamble, guards_pre, right_preamble])
#Push the Preamble through IFFT and add its own Cyclic Prefix
ofdm_preamble = np.fft.ifft(complex_preamble, axis=1)
cp_preamble = ofdm_preamble[:, -cp_length:]
preamble_time_wave = np.hstack([cp_preamble, ofdm_preamble])
preamble_batch = np.repeat(preamble_time_wave, test_size, axis=0)

# ==========================================
# 5. THE RESULTS SWEEP
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
   
    # 2. Subcarrier Allocation (NumPy Array Weaving)
    data_comp = tx_iq_flat[:, :48] + 1j * tx_iq_flat[:, 48:]
    left_data = data_comp[:, :24]
    right_data = data_comp[:, 24:]
   
    pilot_np = np.ones((test_size, 1), dtype=complex)
    left_act = np.hstack([left_data[:, :5], pilot_np, left_data[:, 5:18], pilot_np, left_data[:, 18:]])
    right_act = np.hstack([right_data[:, :6], pilot_np, right_data[:, 6:19], pilot_np, right_data[:, 19:]])
   
    dc_np = np.zeros((test_size, 1), dtype=complex)
    guards_np = np.zeros((test_size, 11), dtype=complex)
    complex_sc = np.hstack([dc_np, left_act, guards_np, right_act])
   
    # 3. IFFT & Add CP
    ofdm_wave = np.fft.ifft(complex_sc, axis=1)
    cp_np = ofdm_wave[:, -cp_length:]
    tx_wave_cp = np.hstack([cp_np, ofdm_wave])
   
    # Add Preamble
    full_tx_frame = np.hstack([preamble_batch, tx_wave_cp])
   
    # 4. Noise Channel (AWGN)
    sigma = np.sqrt(1 / (10**(snr_db / 10)))
    noise = (np.random.normal(0, sigma, full_tx_frame.shape) +
             1j * np.random.normal(0, sigma, full_tx_frame.shape)) / np.sqrt(2)
             
    # 5. Multipath Echo/Rayleigh Fading - attacks the entire 160 sample frame
    delay = 5
    echo = np.zeros_like(full_tx_frame)
    echo[:, delay:] = full_tx_frame[:, :-delay] * 0.5
    
    # Add everything in the air for RX
    rx_frame_cp = full_tx_frame + echo + noise
   
    # ==========================================
    # NEW: OFFLINE BLIND BURST DETECTION
    # ==========================================
    # A. Create a random amount of "Empty Air" padding before and after the frame
    pad_left = np.random.randint(50, 300)
    pad_right = 400 - pad_left # Keep total width uniform for matrix math
    
    noise_left = (np.random.normal(0, sigma, (test_size, pad_left)) + 
                  1j * np.random.normal(0, sigma, (test_size, pad_left))) / np.sqrt(2)
                  
    noise_right = (np.random.normal(0, sigma, (test_size, pad_right)) + 
                   1j * np.random.normal(0, sigma, (test_size, pad_right))) / np.sqrt(2)
    
    # B. Hide the transmitted batch inside the empty air
    continuous_rx_stream = np.hstack([noise_left, rx_frame_cp, noise_right])
    
    # [NEW FIX] Pad the universe with a massive safety buffer.
    # If the detector false-alarms on a noise spike near the end of the array, 
    # this guarantees the numpy slice below will never truncate the output.
    safe_pad = np.zeros((test_size, 200), dtype=complex)
    continuous_rx_stream = np.hstack([continuous_rx_stream, safe_pad])
    
    # C. Run a sliding Cross-Correlator to hunt for the preamble in the noise
    # We test the first frame in the batch to calculate the start index for the whole matrix
    search_window = continuous_rx_stream[0]
    known_preamble_reference = preamble_time_wave[0]
    
    # Calculate correlation magnitude across the entire padded array
    correlation_metric = np.abs(np.correlate(search_window, known_preamble_reference, mode='valid'))
    
    # The detector 'triggers' at the peak of the correlation spike
    calculated_start_idx = np.argmax(correlation_metric)
    
    # ==========================================
    # END BURST DETECTION
    # ==========================================
   
    # 6. Remove Preamble & Remove Data CP (Updated for explicit bounds)
    data_start = calculated_start_idx + 80 + cp_length
    data_end = data_start + 64  # [NEW FIX] Explicitly force the FFT input to be exactly 64 samples
    
    rx_wave = continuous_rx_stream[:, data_start:data_end]
    
    rx_sc = np.fft.fft(rx_wave, axis=1)
   
    # 7. Extract Active Bins & Strip Pilots
    rx_left = rx_sc[:, 1:27]
    rx_right = rx_sc[:, 38:64]
   
    rx_left_data = np.hstack([rx_left[:, :5], rx_left[:, 6:19], rx_left[:, 20:]])
    rx_right_data = np.hstack([rx_right[:, :6], rx_right[:, 7:20], rx_right[:, 21:]])
    rx_active_data = np.hstack([rx_left_data, rx_right_data])
   
    # 8. Bob Estimator
    rx_flat = np.hstack([np.real(rx_active_data), np.imag(rx_active_data)])
    preds = (bob.predict(rx_flat, verbose=0) > 0.5).astype(float)
   
    # Stats
    ber = np.sum(test_bits != preds) / (test_size * m_bits)
    bler = np.sum(np.any(test_bits != preds, axis=1)) / test_size
   
    snr_results.append(snr_db)
    ber_results.append(ber)
    bler_results.append(bler)
    print(f"{snr_db:<10} | {ber:<10.5f} | {bler:<10.5f}")

print("\nArchitecture Complete: Guard Bands, DC Null, Pilots, PSD, and IEEE 802.11a Preamble Active.")

# 6. EXPORT THE TRAINED BRAINS
print("\n--- Saving the Neural Network Weights ---")
alice.save('alicetx_model.h5')
bob.save('bobrx_model.h5')
print("[System Success!] alicetx_model.h5 and bobrx_model.h5 are saved to file folder.")