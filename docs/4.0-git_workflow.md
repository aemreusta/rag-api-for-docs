# Git Workflow and Conventional Commits

This document outlines our Git branching strategy and commit message conventions to maintain a clean, understandable, and automated Git history.

## Branching Strategy: GitFlow (Simplified)

### Main Branch (`main`)

- Always deployable
- Represents the latest stable, production-ready version
- **Rule**: No direct commits to `main`
- Updates only via Pull Requests

### Feature Branches

- All new work happens on feature branches
- Branch from and merge back to `main`
- **Naming Convention**: `<type>/<description>-#<issue-number>`

  ```bash
  # Examples
  feat/add-answer-caching-#21
  fix/resolve-pdf-parsing-bug
  docs/update-api-docs-#15
  ```

- **Rule**: Keep branches short-lived and focused

## Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages. This adds human and machine-readable meaning to commit messages.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: A new feature

  ```
  feat(api): add conversation history endpoint
  ```

- **fix**: A bug fix

  ```
  fix(ingestion): handle empty PDF files gracefully
  ```

- **docs**: Documentation changes

  ```
  docs(readme): update installation instructions
  ```

- **style**: Code style changes

  ```
  style(core): format with black
  ```

- **refactor**: Code changes that neither fix a bug nor add a feature

  ```
  refactor(query): extract prompt template logic
  ```

- **perf**: Performance improvements

  ```
  perf(cache): optimize Redis connection pooling
  ```

- **test**: Adding or modifying tests

  ```
  test(api): add integration tests for rate limiting
  ```

- **chore**: Changes to build process or tools

  ```
  chore(deps): update FastAPI to 0.109.0
  ```

### Scope

Optional field indicating the section of the codebase:

- **api**: API-related changes
- **core**: Core functionality
- **deps**: Dependencies
- **docs**: Documentation
- **ingestion**: Data ingestion
- **query**: Query processing

## Workflow Example

### 1. Create Feature Branch

```bash
git checkout main
git pull
git checkout -b feat/add-redis-caching
```

### 2. Make Changes and Commit

```bash
# Add Redis client implementation
git commit -m "feat(api): add initial redis cache check to chat endpoint"

# Refactor for better organization
git commit -m "refactor(core): create centralized redis client"

# Add tests
git commit -m "test(api): add integration test for cache hit and miss"

# Update documentation
git commit -m "docs(readme): update tech stack with redis"
```

### 3. Push and Create Pull Request

```bash
git push -u origin feat/add-redis-caching
```

- Create PR against `main`
- Fill out PR template
- Request review

### 4. Review Process

- CI/CD runs automated checks
  - Linting (ruff)
  - Type checking (mypy)
  - Tests (pytest)
- Code review by team member
- Address feedback

### 5. Merge

- Squash and merge to `main`
- Delete feature branch

## Best Practices

1. **Keep Commits Atomic**
   - One logical change per commit
   - Make it easy to revert if needed

2. **Write Clear Commit Messages**
   - Present tense ("add" not "added")
   - Imperative mood
   - Descriptive but concise

3. **Regular Rebasing**
   - Keep feature branches up to date with `main`
   - Resolve conflicts early

4. **Clean History**
   - Squash commits before merging
   - Remove temporary commits

5. **Branch Hygiene**
   - Delete merged branches
   - Regular cleanup of stale branches
