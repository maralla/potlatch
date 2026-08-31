Coding Rules
------------

- Implement the feature following project conventions
- Run linters before committing
- Never add `allow(dead_code)` to suppress dead code warning, just delete the code
- Check existing code for similar patterns to avoid duplications or follow them
- Be sensitive for security issues
- **IMPORTANT**: Do NOT create markdown files unless explicitly requested by the user
- Never commit secrets or API keys
- Never add user info or real user data in tests or comments
- Never use real IP addresses, ports, or internal hostnames in code, tests, or config examples — use placeholder domains (e.g. `http://example.invalid`, `http://model.example`) instead
- Arrange items with structure, like put consts at the top of the file
- Avoid long item access path, the path level should not greater than one, like foo::Bar, or Bar, but not foo::baz::Bar

Testing
-------

- Unit tests: `cargo test`
- Do NOT write string-content-match tests — tests that assert a generated prompt, description, or other prose string `.contains("some phrase")`. These are brittle and couple tests to wording. Test behavior instead: assert on returned values, execution traces, error types, and schema structure (e.g. `assert_eq!`, `assert!(props.contains_key(...))`, `method_strings.contains(&"GET")`). Only assert on string content when it is the actual output under test (e.g. a generated markdown file's structure, an error message classification).
- Linting:
  ```
  cargo check
  cargo fmt --check
  cargo clippy
  ```
