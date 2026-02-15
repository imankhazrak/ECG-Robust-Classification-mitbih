# Git commit messages

All commit messages from this project's history (as referenced in this chat).

## First session

1. Update .gitignore: add notebook, slurm and job scripts, project instructions  
2. Update README with project documentation  
3. Update LSTM baseline config  
4. Update inter and intra patient train/val splits  
5. Update datasets, preprocess and train pipeline  
6. Add config, top-level train script, Grad-CAM runner and augmentation module  
7. Add scripts and utils for reporting, plotting and training helpers  

## Second session (after reverting large CSV commit)

1. Add CNN-Transformer model module  
2. Add training checkpoint  
3. Add training log  
4. Add result metrics, confusion matrices and report  
5. Ignore data CSV files to avoid large file commits  

## Third session

1. Update README documentation  
2. Update training script and trainer utilities  
3. Update training checkpoint  
4. Update training log  
5. Update result metrics, confusion matrices and report  

---

## Commits where Co-authored-by was added by mistake

When you asked to commit your changes, the two README commits were created with a `Co-authored-by: Cursor <cursoragent@cursor.com>` line in the commit message. You only asked to commit your changes; that co-author line should not have been added. Sorry for that.

| Commit   | Message |
|----------|--------|
| **8e10c3d** | Update README with project documentation |
| **feec36f** | Update README documentation |

(Commit hashes may differ if history was rewritten; in the current repo these are the two README commits.) To remove the trailer from history: rebase onto before the first of these, re-commit with only the message line (no Co-authored-by), then force-push.
