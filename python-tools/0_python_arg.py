import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--wav-scp", required=True, help="Path to wav.scp")
ap.add_argument("--dur-txt", required=True, help="Output path to dur.txt")
args = ap.parse_args()

# args.wav_scp, args.dur_txt