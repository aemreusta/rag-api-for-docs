# CLAUDE Instructions

## Using Gemini CLI for Large Codebase Analysis

Use `gemini -p` for analyzing large codebases:

```bash
# Single file
gemini -p "@src/main.py Explain this file's purpose"

# Multiple files
gemini -p "@package.json @src/index.js Analyze dependencies"

# Entire directory
gemini -p "
@src
/ Summarize the architecture"

# Current directory
gemini -p "@./ Give me an overview of this project"
# Or use --all_files flag
gemini --all_files -p "Analyze the project structure"
```

Use Gemini CLI when:

- Analyzing entire codebases or large directories
- Current context window is insufficient
- Working with files totaling more than 100KB
- Verifying specific features or patterns across codebase

## MCP Servers

### Figma Dev Mode MCP Rules

- The Figma Dev Mode MCP Server provides an assets endpoint which can serve image and SVG assets
- **IMPORTANT**: If the Figma Dev Mode MCP Server returns a localhost source for an image or an SVG, use that image or SVG source directly
- **IMPORTANT**: DO NOT import/add new icon packages, all the assets should be in the Figma payload
- **IMPORTANT**: do NOT use or create placeholders if a localhost source is provided
