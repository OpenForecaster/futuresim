# SkyRL Submodule Maintenance

This repo tracks SkyRL as a git submodule at `third_party/SkyRL`.

- Fork remote (default submodule URL): `https://github.com/shash42/SkyRL.git`
- Upstream remote: `https://github.com/NovaSky-AI/SkyRL.git`
- Parent repo records a pinned submodule commit.

## One-Time Checks

Run this after a fresh clone:

```bash
git submodule update --init --recursive

git -C third_party/SkyRL remote -v
# expected:
# origin   https://github.com/shash42/SkyRL.git
# upstream https://github.com/NovaSky-AI/SkyRL.git
```

If `upstream` is missing:

```bash
git -C third_party/SkyRL remote add upstream https://github.com/NovaSky-AI/SkyRL.git
```

## Regular Upstream Sync (Recommended)

Use this to pull latest upstream into your fork-backed submodule:

```bash
# 1) Update submodule repo from upstream
git -C third_party/SkyRL fetch upstream
git -C third_party/SkyRL checkout main
git -C third_party/SkyRL merge --ff-only upstream/main

# 2) Push updated main to your fork
git -C third_party/SkyRL push origin main

# 3) Record new submodule pointer in parent repo
git add third_party/SkyRL
git commit -m "Bump SkyRL submodule to upstream main"
```

## Pull Latest From Fork Only

If your fork is already updated and you just want to move the submodule pointer:

```bash
git submodule update --remote --merge third_party/SkyRL
git add third_party/SkyRL
git commit -m "Bump SkyRL submodule"
```

## Making Local SkyRL Changes

Make changes directly inside the submodule, commit there first, then update pointer in parent:

```bash
# in submodule
cd third_party/SkyRL
git checkout main
# edit files...
git add -A
git commit -m "Your SkyRL change"
git push origin main

# back in parent repo
cd ../..
git add third_party/SkyRL
git commit -m "Update SkyRL submodule pointer"
```

## Quick Status Commands

```bash
# Parent repo submodule status
git submodule status third_party/SkyRL

# Submodule branch + divergence
git -C third_party/SkyRL status -sb
git -C third_party/SkyRL rev-list --left-right --count upstream/main...origin/main
```
