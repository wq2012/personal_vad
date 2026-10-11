"""CLI script to download and organize Multilingual LibriSpeech (MLS) + En data.

Downloads a configurable number of speakers per language across all 7
non-English languages of Multilingual LibriSpeech
(`facebook/multilingual_librispeech`: German, French, Spanish, Italian, Dutch,
Portuguese, Polish) from Hugging Face Hub (`train`, `dev`, and `test` splits)
and combines them with English LibriSpeech (`train-clean-100`, `dev-clean`, and
`test-clean`, OpenSLR 12) to produce speaker-disjoint 8-language training,
development (validation), and evaluation manifests.
"""

import argparse
from collections.abc import Sequence
import json
import os
import tarfile
from typing import Any, Optional
import huggingface_hub
import numpy as np
import soundfile as sf


MLS_LANGUAGES: dict[str, str] = {
    "de": "mls_german",
    "fr": "mls_french",
    "es": "mls_spanish",
    "it": "mls_italian",
    "nl": "mls_dutch",
    "pt": "mls_portuguese",
    "pl": "mls_polish",
}


def load_mls_transcripts(
    repo_id: str,
    mls_subdir: str,
    split: str,
    token: Optional[str] = None,
) -> dict[str, str]:
  """Downloads and parses `transcripts.txt` for an MLS language split."""
  rel_path = f"data/{mls_subdir}/{split}/transcripts.txt"
  local_path = huggingface_hub.hf_hub_download(
      repo_id=repo_id,
      filename=rel_path,
      repo_type="dataset",
      token=token,
  )
  transcripts: dict[str, str] = {}
  with open(local_path, "r", encoding="utf-8") as f:
    for line in f:
      parts = line.strip().split("\t", 1)
      if len(parts) == 2:
        transcripts[parts[0].strip()] = parts[1].strip()
  return transcripts


def extract_mls_speaker_utts(
    tar_path: str,
    output_audio_dir: str,
    lang_code: str,
    speaker_id: str,
    transcripts: dict[str, str],
    max_utts: int = 12,
    min_sec: float = 1.8,
    max_sec: float = 5.5,
) -> list[dict[str, Any]]:
  """Extracts up to `max_utts` 16 kHz `.flac` files from an MLS speaker tar."""
  os.makedirs(output_audio_dir, exist_ok=True)
  records: list[dict[str, Any]] = []

  with tarfile.open(tar_path, "r:gz") as tf:
    members = sorted(
        [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".flac")],
        key=lambda m: m.name,
    )
    for member in members:
      if len(records) >= max_utts:
        break
      base = os.path.splitext(os.path.basename(member.name))[0]
      fobj = tf.extractfile(member)
      if fobj is None:
        continue
      try:
        audio, sr = sf.read(fobj, dtype="float32")
      except Exception:  # pylint: disable=broad-except
        continue
      if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
      if sr != 16000 or audio.size < int(min_sec * sr):
        continue
      # Trim overly long audiobook clips to max_sec while preserving natural
      # speech and trailing silence.
      if audio.size > int(max_sec * sr):
        audio = audio[: int(max_sec * sr)]

      utt_id = f"{lang_code}_{base}"
      out_flac = os.path.join(output_audio_dir, f"{utt_id}.flac")
      sf.write(out_flac, audio, 16000)
      duration = float(audio.size) / 16000.0
      records.append({
          "utt_id": utt_id,
          "speaker_id": f"{lang_code}_{speaker_id}",
          "language": lang_code,
          "audio_path": out_flac,
          "transcript": transcripts.get(base, ""),
          "duration_sec": round(duration, 3),
      })
  return records


def collect_cached_mls_speaker_utts(
    output_audio_dir: str,
    lang_code: str,
    speaker_id: str,
    transcripts: dict[str, str],
    max_utts: int = 12,
) -> list[dict[str, Any]]:
  """Loads already-extracted speaker utterances from disk if present."""
  if not os.path.isdir(output_audio_dir):
    return []
  prefix = f"{lang_code}_{speaker_id}_"
  records: list[dict[str, Any]] = []
  for fname in sorted(os.listdir(output_audio_dir)):
    if not (fname.startswith(prefix) and fname.endswith(".flac")):
      continue
    if len(records) >= max_utts:
      break
    out_flac = os.path.join(output_audio_dir, fname)
    try:
      info = sf.info(out_flac)
    except Exception:  # pylint: disable=broad-except
      continue
    utt_id = os.path.splitext(fname)[0]
    base = utt_id[len(lang_code) + 1:]
    records.append({
        "utt_id": utt_id,
        "speaker_id": f"{lang_code}_{speaker_id}",
        "language": lang_code,
        "audio_path": out_flac,
        "transcript": transcripts.get(base, ""),
        "duration_sec": round(float(info.duration), 3),
    })
  return records


def collect_english_librispeech(
    librispeech_root: str,
    output_audio_dir: str,
    split_subdirs: Sequence[str],
    num_speakers: int,
    max_utts_per_speaker: int = 12,
    min_sec: float = 1.8,
    max_sec: float = 5.5,
) -> list[dict[str, Any]]:
  """Collects English LibriSpeech utterances and transcripts."""
  os.makedirs(output_audio_dir, exist_ok=True)
  speaker_dirs: dict[str, list[str]] = {}
  for sub in split_subdirs:
    sub_dir = os.path.join(librispeech_root, sub)
    if not os.path.isdir(sub_dir):
      continue
    for spk in sorted(os.listdir(sub_dir)):
      spk_path = os.path.join(sub_dir, spk)
      if os.path.isdir(spk_path):
        speaker_dirs.setdefault(spk, []).append(spk_path)

  all_records: list[dict[str, Any]] = []
  selected_speakers = 0
  sorted_spks = sorted(
      speaker_dirs.keys(), key=lambda x: int(x) if x.isdigit() else x
  )
  for spk in sorted_spks:
    if selected_speakers >= num_speakers:
      break
    spk_records: list[dict[str, Any]] = []
    for spk_path in speaker_dirs[spk]:
      for chapter in sorted(os.listdir(spk_path)):
        chap_dir = os.path.join(spk_path, chapter)
        if not os.path.isdir(chap_dir):
          continue
        trans_file = os.path.join(chap_dir, f"{spk}-{chapter}.trans.txt")
        trans_map: dict[str, str] = {}
        if os.path.exists(trans_file):
          with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
              parts = line.strip().split(" ", 1)
              if len(parts) == 2:
                trans_map[parts[0]] = parts[1]
        for fname in sorted(os.listdir(chap_dir)):
          if not fname.endswith(".flac"):
            continue
          if len(spk_records) >= max_utts_per_speaker:
            break
          base = os.path.splitext(fname)[0]
          flac_path = os.path.join(chap_dir, fname)
          try:
            audio, sr = sf.read(flac_path, dtype="float32")
          except Exception:  # pylint: disable=broad-except
            continue
          if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
          if sr != 16000 or audio.size < int(min_sec * sr):
            continue
          if audio.size > int(max_sec * sr):
            audio = audio[: int(max_sec * sr)]
          utt_id = f"en_{base}"
          out_flac = os.path.join(output_audio_dir, f"{utt_id}.flac")
          sf.write(out_flac, audio, 16000)
          spk_records.append({
              "utt_id": utt_id,
              "speaker_id": f"en_{spk}",
              "language": "en",
              "audio_path": out_flac,
              "transcript": trans_map.get(base, ""),
              "duration_sec": round(float(audio.size) / 16000.0, 3),
          })
    if len(spk_records) >= 8:
      for idx, rec in enumerate(spk_records):
        rec["role"] = "enroll" if idx < 3 else "utterance"
      all_records.extend(spk_records)
      selected_speakers += 1
  return all_records


def build_multilingual_dataset(
    output_dir: str,
    librispeech_root: str,
    train_speakers_per_mls_lang: int = 12,
    dev_speakers_per_mls_lang: int = 2,
    test_speakers_per_mls_lang: int = 5,
    en_train_speakers: int = 15,
    en_dev_speakers: int = 2,
    en_test_speakers: int = 6,
    max_utts_per_speaker: int = 12,
    repo_id: str = "facebook/multilingual_librispeech",
    token: Optional[str] = None,
) -> dict[str, Any]:
  """Downloads MLS splits and combines with English LibriSpeech."""
  os.makedirs(output_dir, exist_ok=True)
  api = huggingface_hub.HfApi(token=token)
  repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

  train_records: list[dict[str, Any]] = []
  dev_records: list[dict[str, Any]] = []
  test_records: list[dict[str, Any]] = []

  # 1. English LibriSpeech (OpenSLR 12)
  if os.path.isdir(librispeech_root):
    print("Collecting English LibriSpeech (en)...")
    en_train = collect_english_librispeech(
        librispeech_root=librispeech_root,
        output_audio_dir=os.path.join(output_dir, "audio", "train", "en"),
        split_subdirs=["train-clean-100"],
        num_speakers=en_train_speakers,
        max_utts_per_speaker=max_utts_per_speaker,
    )
    en_dev = collect_english_librispeech(
        librispeech_root=librispeech_root,
        output_audio_dir=os.path.join(output_dir, "audio", "dev", "en"),
        split_subdirs=["dev-clean"],
        num_speakers=en_dev_speakers,
        max_utts_per_speaker=max_utts_per_speaker,
    )
    en_test = collect_english_librispeech(
        librispeech_root=librispeech_root,
        output_audio_dir=os.path.join(output_dir, "audio", "test", "en"),
        split_subdirs=["test-clean"],
        num_speakers=en_test_speakers,
        max_utts_per_speaker=max_utts_per_speaker,
    )
    train_records.extend(en_train)
    dev_records.extend(en_dev)
    test_records.extend(en_test)

  # 2. Non-English Multilingual LibriSpeech (de, fr, es, it, nl, pt, pl)
  for lang_code, mls_subdir in MLS_LANGUAGES.items():
    for split, target_spks, dest_list in [
        ("train", train_speakers_per_mls_lang, train_records),
        ("dev", dev_speakers_per_mls_lang, dev_records),
        ("test", test_speakers_per_mls_lang, test_records),
    ]:
      print(f"Downloading {mls_subdir} ({lang_code}) [{split}]...")
      transcripts = load_mls_transcripts(
          repo_id=repo_id, mls_subdir=mls_subdir, split=split, token=token
      )
      prefix = f"data/{mls_subdir}/{split}/audio/"
      tars = sorted(
          [
              f
              for f in repo_files
              if f.startswith(prefix) and f.endswith(".tar.gz")
          ]
      )
      spk_to_tars: dict[str, list[str]] = {}
      for t in tars:
        spk_id = os.path.basename(t).split("_")[0]
        spk_to_tars.setdefault(spk_id, []).append(t)

      added_spks = 0
      out_audio_dir = os.path.join(output_dir, "audio", split, lang_code)
      for spk_id in sorted(spk_to_tars.keys()):
        if added_spks >= target_spks:
          break
        cached = collect_cached_mls_speaker_utts(
            output_audio_dir=out_audio_dir,
            lang_code=lang_code,
            speaker_id=spk_id,
            transcripts=transcripts,
            max_utts=max_utts_per_speaker,
        )
        if len(cached) >= max_utts_per_speaker:
          spk_utts = cached[:max_utts_per_speaker]
        else:
          spk_utts = []
          for tar_rel in spk_to_tars[spk_id]:
            if len(spk_utts) >= max_utts_per_speaker:
              break
            local_tar = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=tar_rel,
                repo_type="dataset",
                token=token,
            )
            extracted = extract_mls_speaker_utts(
                tar_path=local_tar,
                output_audio_dir=out_audio_dir,
                lang_code=lang_code,
                speaker_id=spk_id,
                transcripts=transcripts,
                max_utts=max_utts_per_speaker - len(spk_utts),
            )
            spk_utts.extend(extracted)
            try:
              os.remove(local_tar)
            except OSError:
              pass
        if len(spk_utts) >= 8:
          for idx, rec in enumerate(spk_utts):
            rec["role"] = "enroll" if idx < 3 else "utterance"
          dest_list.extend(spk_utts)
          added_spks += 1

  train_manifest_path = os.path.join(output_dir, "train_manifest.json")
  dev_manifest_path = os.path.join(output_dir, "dev_manifest.json")
  test_manifest_path = os.path.join(output_dir, "test_manifest.json")
  with open(train_manifest_path, "w", encoding="utf-8") as f:
    json.dump(train_records, f, indent=2, ensure_ascii=False)
  with open(dev_manifest_path, "w", encoding="utf-8") as f:
    json.dump(dev_records, f, indent=2, ensure_ascii=False)
  with open(test_manifest_path, "w", encoding="utf-8") as f:
    json.dump(test_records, f, indent=2, ensure_ascii=False)

  train_spks = sorted({r["speaker_id"] for r in train_records})
  dev_spks = sorted({r["speaker_id"] for r in dev_records})
  test_spks = sorted({r["speaker_id"] for r in test_records})
  all_recs = train_records + dev_records + test_records
  langs_present = sorted({r["language"] for r in all_recs})

  disjoint = (
      len(set(train_spks).intersection(dev_spks)) == 0
      and len(set(train_spks).intersection(test_spks)) == 0
      and len(set(dev_spks).intersection(test_spks)) == 0
  )

  summary = {
      "languages": langs_present,
      "num_languages": len(langs_present),
      "source_corpora": {
          "non_english": (
              "facebook/multilingual_librispeech (train, dev, test splits)"
          ),
          "english": (
              "OpenSLR 12 LibriSpeech (train-clean-100, dev-clean, test-clean)"
          ),
      },
      "train_num_speakers": len(train_spks),
      "train_num_utterances": len(train_records),
      "dev_num_speakers": len(dev_spks),
      "dev_num_utterances": len(dev_records),
      "test_num_speakers": len(test_spks),
      "test_num_utterances": len(test_records),
      "total_num_speakers": len(train_spks) + len(dev_spks) + len(test_spks),
      "total_num_utterances": len(all_recs),
      "speaker_disjoint": disjoint,
      "per_language_speakers": {
          lang: {
              "train_speakers": len({
                  r["speaker_id"]
                  for r in train_records
                  if r["language"] == lang
              }),
              "dev_speakers": len({
                  r["speaker_id"]
                  for r in dev_records
                  if r["language"] == lang
              }),
              "test_speakers": len({
                  r["speaker_id"]
                  for r in test_records
                  if r["language"] == lang
              }),
              "train_utterances": sum(
                  1 for r in train_records if r["language"] == lang
              ),
              "dev_utterances": sum(
                  1 for r in dev_records if r["language"] == lang
              ),
              "test_utterances": sum(
                  1 for r in test_records if r["language"] == lang
              ),
          }
          for lang in langs_present
      },
      "train_manifest": train_manifest_path,
      "dev_manifest": dev_manifest_path,
      "test_manifest": test_manifest_path,
  }
  summary_path = os.path.join(output_dir, "dataset_summary.json")
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
  return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses CLI arguments."""
  parser = argparse.ArgumentParser(
      description="Download and prepare Multilingual LibriSpeech (MLS) + En."
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="/usr/local/google/home/quanw/Data/multilingual_librispeech",
      help="Destination directory for extracted MLS + LibriSpeech dataset.",
  )
  parser.add_argument(
      "--librispeech_root",
      type=str,
      default=(
          "/usr/local/google/home/quanw/Data/speaker_datasets/"
          "librispeech/LibriSpeech"
      ),
      help="Path to local English LibriSpeech directory.",
  )
  parser.add_argument(
      "--train_speakers_per_mls_lang",
      "--speakers_per_lang",
      dest="train_speakers_per_mls_lang",
      type=int,
      default=12,
      help="Number of training speakers per non-English MLS language.",
  )
  parser.add_argument(
      "--dev_speakers_per_mls_lang",
      type=int,
      default=2,
      help="Number of development speakers per non-English MLS language.",
  )
  parser.add_argument(
      "--test_speakers_per_mls_lang",
      type=int,
      default=5,
      help="Number of evaluation speakers per non-English MLS language.",
  )
  parser.add_argument(
      "--en_train_speakers",
      type=int,
      default=15,
      help="Number of English LibriSpeech training speakers.",
  )
  parser.add_argument(
      "--en_dev_speakers",
      type=int,
      default=2,
      help="Number of English LibriSpeech development speakers.",
  )
  parser.add_argument(
      "--en_test_speakers",
      type=int,
      default=6,
      help="Number of English LibriSpeech evaluation speakers.",
  )
  parser.add_argument(
      "--max_utts_per_speaker",
      "--utterances_per_speaker",
      dest="max_utts_per_speaker",
      type=int,
      default=12,
      help="Maximum utterances to extract per speaker.",
  )
  return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
  args = parse_args(argv)
  token = os.environ.get("HF_TOKEN")
  summary = build_multilingual_dataset(
      output_dir=args.output_dir,
      librispeech_root=args.librispeech_root,
      train_speakers_per_mls_lang=args.train_speakers_per_mls_lang,
      dev_speakers_per_mls_lang=args.dev_speakers_per_mls_lang,
      test_speakers_per_mls_lang=args.test_speakers_per_mls_lang,
      en_train_speakers=args.en_train_speakers,
      en_dev_speakers=args.en_dev_speakers,
      en_test_speakers=args.en_test_speakers,
      max_utts_per_speaker=args.max_utts_per_speaker,
      token=token,
  )
  print("Dataset summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
