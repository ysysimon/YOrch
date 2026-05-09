# Doxygen Documentation Notes

This directory contains the Doxygen-specific documentation build inputs for
YOrch.

The generated documentation uses the repository `README.md` as its main page and
includes:

- public headers under `include/yorch/`
- the English and Chinese user wiki pages under `wiki/`

Build the generated documentation with:

```bash
cmake --preset docs
cmake --build --preset docs --target docs
```

After the build completes, open `build/docs/target/html/index.html`.
