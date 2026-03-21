## Branch Naming

### Recommended Format

```text
<type>/<short-description>
```

If a task ID or issue number is available, use:

```text
<type>/<issue-id>-<short-description>
```

### Branch Types

- `master`: Main branch. It should always remain usable, stable, and releasable.
- `feat/*`: New feature development.
- `fix/*`: Bug fixes.
- `refactor/*`: Refactoring work.
- `docs/*`: Documentation updates.
- `test/*`: Test-related changes.
- `chore/*`: General maintenance such as dependencies, scripts, CI, or directory cleanup.
- `perf/*`: Performance improvements.
- `build/*`: Build system changes.
- `ci/*`: Continuous integration configuration changes.

### Examples

```text
feat/123-add-jwt-login
fix/456-cors-preflight
refactor/789-split-cache-layer
docs/update-api-overview
```

## Commit Message Format

### Recommended Format

```text
<type>(<scope>): <subject>
```

`scope` is optional.

### Commit Types

- `feat`: A new feature.
- `fix`: A bug fix.
- `refactor`: Code changes that do not alter external behavior.
- `docs`: Documentation changes.
- `test`: Test-related updates.
- `chore`: General maintenance tasks.
- `perf`: Performance improvements.
- `build`: Changes to the build system or packaging flow.
- `ci`: Changes to CI/CD configuration.

### Examples

```text
feat(runtime): add topological scheduler
fix(ws): handle reconnect after timeout
refactor(store): split cache from registry
docs(build): add Windows setup notes
perf(sim): reduce temporary allocation
test(graph): cover cycle detection cases
```
