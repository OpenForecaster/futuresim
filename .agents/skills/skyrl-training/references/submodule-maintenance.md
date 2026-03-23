# SkyRL Submodule Maintenance

This repo tracks SkyRL as a git submodule at `third_party/SkyRL`.

- fork remote: `origin`
- upstream remote: `https://github.com/NovaSky-AI/SkyRL.git`
- parent repo records a pinned submodule commit

## One-Time Checks

```bash
git submodule update --init --recursive
git -C third_party/SkyRL remote -v
```

If `upstream` is missing:

```bash
git -C third_party/SkyRL remote add upstream https://github.com/NovaSky-AI/SkyRL.git
```

## Regular Upstream Sync

```bash
git -C third_party/SkyRL fetch upstream
git -C third_party/SkyRL checkout main
git -C third_party/SkyRL merge --ff-only upstream/main
git -C third_party/SkyRL push origin main

git add third_party/SkyRL
git commit -m "Bump SkyRL submodule to upstream main"
```

## Move To The Latest Fork Commit

```bash
git submodule update --remote --merge third_party/SkyRL
git add third_party/SkyRL
git commit -m "Bump SkyRL submodule"
```

## Local SkyRL Changes

```bash
cd third_party/SkyRL
git checkout main
# edit files
git add -A
git commit -m "Your SkyRL change"
git push origin main

cd ../..
git add third_party/SkyRL
git commit -m "Update SkyRL submodule pointer"
```
