from cx_Freeze import setup, Executable

include_files = [
    (
        "C:\\Users\\Kristoph\\AppData\\Local\\pypoetry\\Cache\\virtualenvs\\nice_sound_looper-UUVSEo-F-py3.10\\Lib\\site-packages\\_soundfile_data",
        "lib\\_soundfile_data",
    ),
    (
        "C:\\Users\\Kristoph\\AppData\\Local\\pypoetry\\Cache\\virtualenvs\\nice_sound_looper-UUVSEo-F-py3.10\\Lib\\site-packages\\kivy_deps",
        "lib\\kivy_deps",
    ),
    (
        "C:\\Users\\Kristoph\\AppData\\Local\\pypoetry\\Cache\\virtualenvs\\nice_sound_looper-UUVSEo-F-py3.10\\share",
        "share",
    )
]

build_exe_options = {
    "packages": ["scipy", "kivy", "librosa", "soundfile", "sounddevice"],
    "include_files": include_files,
}

# set base="Win32GUI" to remove the console
base = None

setup(
    name="looper",
    version="0.1",
    description="GUI for looping sounds.",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py", base=base)],
)
