# Major Project - Spec-Driven Development

This project uses [GitHub Spec-Kit](https://github.com/github/spec-kit) for spec-driven development.

## Setup Complete ✅

- ✅ Spec-kit repository cloned
- ✅ Specify CLI installed (`specify` command available)
- ✅ Project structure created

## Project Structure

```
major project/
├── spec-kit/          # Spec-kit repository (reference)
├── src/               # Your project source code
└── README.md          # This file
```

## Quick Start

The `specify` CLI is now available. Here's how to use it:

### 1. Initialize a Spec-Kit Project

Navigate to your project directory and initialize:

```powershell
cd "major project"
specify init . --ai claude
# or
specify init --here --ai claude
```

### 2. Available Commands

- `specify init <PROJECT_NAME>` - Create a new project
- `specify check` - Verify all tools are installed
- `specify version` - Display version info
- `specify extension` - Manage extensions

### 3. Spec-Kit Workflow

Once initialized, you'll use these AI commands in your editor:

1. **`/speckit.constitution`** - Define project principles and guidelines
2. **`/speckit.specify`** - Describe what you want to build (the "what" and "why")
3. **`/speckit.plan`** - Provide tech stack and architecture choices
4. **`/speckit.tasks`** - Break down into actionable tasks
5. **`/speckit.implement`** - Execute the tasks

## Additional Resources

- [Spec-Kit Documentation](https://github.github.io/spec-kit/)
- [Spec-Kit Repository](./spec-kit/)
- [What is Spec-Driven Development?](./spec-kit/spec-driven.md)

## Notes

- The `specify` command is installed using `uv tool install`
- PATH has been temporarily updated for this PowerShell session
- For permanent PATH updates, run: `uv tool update-shell`

## Next Steps

1. Run `specify init --here --ai claude` in the `major project` directory
2. Follow the spec-driven development workflow above
3. Start building with AI-assisted specifications!
