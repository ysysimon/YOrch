# Development Guide

This guide collects local development conventions for YOrch.

## Build Presets

Use CMake presets for configure, build, and test workflows:

```bash
cmake --list-presets
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

For benchmark development, configure and build the benchmark preset:

```bash
cmake --preset benchmark
cmake --build --preset benchmark
```

## clangd Compilation Database

The repository-level `.clangd` points clangd at a stable local path:

```yaml
CompileFlags:
  CompilationDatabase: build/clangd
```

Treat `build/clangd` as a local symlink to the active CMake preset build directory.
Run the following commands from the repository **root** so the relative paths resolve
to the intended build directories.

On Linux/macOS, use `ln` to switch the active compilation database.

For regular development:

```bash
ln -sfn dev build/clangd
```

For benchmark development:

```bash
ln -sfn benchmark build/clangd
```

On Windows PowerShell, use `New-Item`:

```powershell
New-Item -ItemType SymbolicLink -Path build\clangd -Target dev -Force
New-Item -ItemType SymbolicLink -Path build\clangd -Target benchmark -Force
```

On Windows `cmd`, use `mklink`:

```cmd
mklink /D build\clangd dev
mklink /D build\clangd benchmark
```

Windows symlink creation may require Developer Mode or an elevated shell.

After switching the symlink, restart clangd or reload the editor window so clangd
re-reads the compilation database.

Make sure the target preset has been configured first, because clangd needs the
generated `compile_commands.json` file:

```bash
cmake --preset dev
cmake --preset benchmark
```

This keeps `.clangd` stable in the repository while allowing each developer to
choose the local build tree that powers editor diagnostics and indexing.
