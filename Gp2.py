import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def calculate_zero_crossing_rate(frame):
    return sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))


def calculate_short_time_energy(frame):
    return sum(frame ** 2) / len(frame)


def classify_frame(frame, threshold_unvoiced=0.001, threshold_voiced=1):
    zero_crossing_rate = calculate_zero_crossing_rate(frame)
    short_time_energy = calculate_short_time_energy(frame)

    if short_time_energy < threshold_unvoiced:
        return "Unvoiced"
    elif zero_crossing_rate < threshold_voiced:
        return "Voiced"
    elif short_time_energy < 0.05:
        return "Silence"


def analyze_speech_signal(file_path):
    samples, sr = librosa.load(file_path)

    # Convert window size to samples for a 20 ms window
    window_size_ms = 20
    frame_size = int((window_size_ms / 1000) * sr)

    # Hop size (Step Size)
    hop_size = 512

    # Initialize labels vector
    labels = []

    # Initialize zero-crossing rate and short-time energy vectors
    zero_crossing_rates = []
    short_time_energies = []

    # Iterate through frames
    for i in range(0, len(samples) - frame_size, hop_size):
        frame = samples[i:i + frame_size]

        # Calculate zero-crossing rate and short-time energy
        zero_crossing_rate = calculate_zero_crossing_rate(frame)
        short_time_energy = calculate_short_time_energy(frame)

        # Classify frame
        label = classify_frame(frame)

        # Append values to vectors
        zero_crossing_rates.append(zero_crossing_rate)
        short_time_energies.append(short_time_energy)

        # Append label to the vector
        labels.append(label)

    return labels, zero_crossing_rates, short_time_energies


def visualize_results(file_path, labels, zero_crossing_rates, short_time_energies):
    samples, sr = librosa.load(file_path, sr=None)

    # Plot the time-domain signal
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(samples, sr=sr)
    plt.title('Time-Domain Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    # Plot Zero Crossing Rate
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(0, len(zero_crossing_rates)), zero_crossing_rates, color='b')
    plt.title('Zero Crossing Rate Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('Zero Crossing Rate')
    plt.show()

    # Plot Short Time Energy
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(0, len(short_time_energies)), short_time_energies, color='r')
    plt.title('Short Time Energy Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('Short Time Energy')
    plt.show()

    # Plot the classified frames
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(0, len(labels)), labels, marker='o', linestyle='-', color='g')
    plt.title('Speech Classification Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('Classification')
    plt.yticks(["Silence", "Voiced", "Unvoiced"])
    plt.show()


if __name__ == "__main__":
    file_path = r''
    labels, zero_crossing_rates, short_time_energies = analyze_speech_signal(file_path)
    visualize_results(file_path, labels, zero_crossing_rates, short_time_energies)
