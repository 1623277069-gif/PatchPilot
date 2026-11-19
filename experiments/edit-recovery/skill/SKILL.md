---
name: edit-recovery
description: Recover failed edit_file calls using version-checked line replacement. Use only after argument or exact-match failures on an existing source file.
---

Use this fallback for the current failed edit, then resume the normal repair and focused check. Do not use it for already successful edits or change unrelated code.

Call `run_skill_script` with `skill_name="edit-recovery"`, `file_path="scripts/edit_lines.py"`, and an `args` list of strings. The script works in `/workspace`, independently of the skill's temporary directory.

1. Read a short range around the intended change (1-based, inclusive; at most 120 lines):

   `args=["read", "src/example.py", "20", "45"]`

   Read stdout JSON: `sha256`, actual `start`/`end`, and `lines` containing `number` and `text`. Use the returned current text, indentation and line numbers; the read may clamp the end to EOF.

2. Replace only the intended whole lines using that full-file hash:

   `args=["replace", "src/example.py", "27", "29", "<sha256 from read>", "    return result\n"]`

   Supply actual replacement code as the final string, without printed line numbers. Encode newlines normally in JSON; do not double-escape or supply a second old-code block. Empty replacement deletes the selected lines. A missing trailing newline is added when following lines exist. For insertion, replace a nearby line and include it unchanged with the new lines.

Check stdout JSON `status`, not merely the outer tool status. `ok` returns the actual diff and new hash; Python syntax is checked, not behavior. `STALE_FILE` means reread and reconsider the edit. Other errors leave validation-rejected edits unapplied; correct the specific path, range or code rather than repeating identical arguments. The script rejects test/config edits and does not create files. If this fallback also fails without a clear correction, inspect the current diff and finish the best-supported repair within budget.
