"""
Builds (or incrementally updates) the hybrid index from PDFs in data/:
- Chroma vectorstore (dense/semantic search)
- index_store/bm25.json (sparse/keyword search)
- index_store/parents.json (parent chunks for context expansion)
- index_store/manifest.json (per-file hash, so unchanged PDFs are skipped)

Run locally whenever you add/change PDFs in data/. Pass --force to wipe and
rebuild everything from scratch (needed if you change EMBEDDING_MODEL or
chunk-size settings in .env, since old chunks won't match the new scheme).
"""
import argparse

from app.indexing import build_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Wipe and rebuild the entire index")
    args = parser.parse_args()

    result = build_index(force=args.force)
    print(f"Indexed {result['indexed']} file(s), skipped {result['skipped']} unchanged file(s).")
    for f in result["files"]:
        print(f"  - {f['source']}: {f['chunks']} chunks from {f['pages']} pages ({f.get('policy_name', '?')})")
    if result["indexed"] == 0 and result["skipped"] == 0:
        print("No PDFs found in data/. Add policy PDFs, then re-run this script.")


if __name__ == "__main__":
    main()
