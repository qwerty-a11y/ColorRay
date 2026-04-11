# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run.py'],
    pathex=["H:\\ColorRay\\src"],
    binaries=[
        ('C:\\Users\\kafuchino\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin\\ffmpeg.exe', '.'),
        ('C:\\Users\\kafuchino\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1.1-full_build\\bin\\ffprobe.exe', '.'),
        ('H:\\ColorRay\\src\\receiver\\detector\\warp_engine.dll', '.'),
        ('H:\\ColorRay\\src\\receiver\\detector\\kmeans_core.dll','.')
    ],
    datas=[],
    hiddenimports=['asyncio', 'importlib.resources'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ColorRay',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['H:\\ColorRay\\src\\Noa.ico'],
)
