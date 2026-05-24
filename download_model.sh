#!/bin/sh
set -e

REPO="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
MODEL_FILE="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
Qwen3-Coder-30B-A3B-Instruct GGUF downloader

Usage:
  ./download_model.sh [--token TOKEN]

Target:
  $MODEL_FILE
      Q4_K_M GGUF, about 17.3 GiB.
      This is the only supported model target for this repository.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache is used if present.

Environment:
  DS4_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf

After download the script updates:
  ./qwen3-coder.gguf -> <download directory>/$MODEL_FILE
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

download_one() {
    file=$1
    out="$OUT_DIR/$file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$REPO/resolve/main/$file"

    mkdir -p "$OUT_DIR"

    if [ -e "$aria2_part" ]; then
        echo "Found incomplete aria2 download sidecar: $aria2_part" >&2
        echo "Finish or remove that partial download before using this curl downloader." >&2
        exit 1
    fi

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

download_one "$MODEL_FILE"

cd "$ROOT"
ln -sfn "$OUT_DIR/$MODEL_FILE" qwen3-coder.gguf
echo "Linked ./qwen3-coder.gguf -> $OUT_DIR/$MODEL_FILE"
echo
echo "Done."
