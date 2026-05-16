# Archive

Use this directory for old scripts, imported snapshots, and superseded experiments once they are no longer referenced by active runs.

Do not move active launch scripts here until remote jobs and documentation have been updated.

## Copy-Paste Directory Transfer

These scripts help move a directory/file when direct download is unavailable but text copy still works.

On the source/server machine:

```bash
python3 /path/to/encode_dir_to_text.py /path/to/source_dir -o source_dir_text_bundle
```

This creates:

- `manifest.json`
- `chunk_0001_of_XXXX.b64.txt`
- `chunk_0002_of_XXXX.b64.txt`
- ...

Copy the text content of `manifest.json` and every chunk file into matching files on the destination/local machine.

On the destination/local machine:

```bash
python3 /path/to/restore_from_text.py source_dir_text_bundle -o restored_output
```

The restore script verifies every chunk and the final archive with SHA256 before extracting.

For smaller copy blocks, lower the chunk size:

```bash
python3 /path/to/encode_dir_to_text.py /path/to/source_dir -o source_dir_text_bundle --chunk-chars 200000
```
