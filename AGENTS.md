# Agent Instructions

This file provides guidance to AI agents working on this project.

### Code Quality
- Write tests for all new features
- Maintain test coverage above 85%
- Run linters before committing
- Document public APIs
- Never add `allow(dead_code)` to suppress dead code warning, just delete the code

### For Worker Agent

When implementing issues:
1. Read the issue carefully and all comments
2. Check existing code for similar patterns
3. Write tests first (TDD approach preferred)
4. Implement the feature following project conventions
5. Run all tests and linters
6. Update documentation if needed
7. **IMPORTANT**: Do NOT create markdown files (*.md) unless explicitly requested by the user in the issue

### For Reviewer Agent

When reviewing merge requests:
1. Verify implementation matches issue requirements
2. Check code quality and style compliance
3. Ensure tests are comprehensive
4. Verify no breaking changes
5. Check for security issues
6. Validate documentation updates

### Testing

- Unit tests: `cargo test`
- Linting:
  ```
  cargo check
  cargo fmt --check
  cargo clippy
  ```

### Security

- Never commit secrets or API keys
- Use environment variables for configuration
- Validate all user input
- Follow OWASP guidelines

### Documentation

- Update README.md for user-facing changes
- Add inline comments for complex logic
- **Do NOT create new markdown documentation files** unless explicitly specified in the issue
- Only modify existing documentation files when necessary for the implementation
