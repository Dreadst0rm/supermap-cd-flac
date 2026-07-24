# Packaging SuperMap CD

## Chosen stack (2026)

| Layer | Choice | Why |
|-------|--------|-----|
| GUI | **PySide6** | Already used; freezes well; fit for tabs/threads/dialogs |
| Freezer | **PyInstaller onedir** | Largest ecosystem; reliable for Qt; CI-friendly |
| Platforms | Build **per OS** | No single binary runs on Windows and macOS |

Build from the repo root:

```powershell
python -m pip install -e ".[packaging]"
python packaging/build_executables.py
```

Outputs:

- `dist/supermap-cd/` — CLI (`supermap-cd`)
- `dist/SuperMap-Converter/` — GUI

Torch is **excluded** from default freezes. ffmpeg is **not** bundled; users install it separately for non-native formats. CD rip remains **Windows-only**; convert works on Windows and macOS.

CI: `.github/workflows/build-executables.yml` (tag `v*` or workflow dispatch).

## Public release checklist

1. Prefer **onedir** (current) over onefile for Qt.
2. **Code-sign** Windows executables.
3. **Notarize** the macOS app for Gatekeeper.
4. Keep ffmpeg and optional `[ml]` torch documented as separate installs.

## Escalation (do not chase by default)

- Users want MSI/DMG installers → Briefcase, Inno Setup / WiX, or create-dmg on top of the onedir tree.
- Antivirus false positives or painful cold start → spike Nuitka for the GUI entry only.

Do **not** rewrite the GUI in Flet/Toga/Electron just to get an executable.
