# Coding Structure Principles

This document outlines the key principles that guide our code structure and organization. Following these principles ensures our project remains clean, scalable, and maintainable.

## Single Responsibility Principle (SRP)

### What It Means

Every module, class, or function should have responsibility over a single part of the application's functionality.

### How It Applies

- `chat.py` - Responsible for orchestrating API requests, not RAG implementation details
- `query_engine.py` - Handles all LlamaIndex logic, isolated from web endpoints
- `config.py` - Manages environment variables, not database connections

### Why It's Important

- Makes code easier to test and debug
- Simplifies refactoring
- Clear boundaries of responsibility
- When changes are needed, you know exactly which file to modify

## Dependency Inversion Principle (DIP)

### What It Means

High-level modules should not depend on low-level modules. Both should depend on abstractions.

### How It Applies

- **FastAPI Dependencies**: `chat.py` endpoints declare dependencies but don't create them

  ```python
  def get_answer(db: Session = Depends(get_db)):
      # Endpoint depends on abstraction, not implementation
  ```

- **Query Engine**: API endpoints depend on the QueryEngine abstraction, not LlamaIndex details

  ```python
  class QueryEngine:
      # High-level interface that hides implementation complexity
  ```

### Why It's Important

- Decouples code components
- Makes it easier to swap implementations (e.g., changing vector databases)
- Facilitates testing through dependency injection

## Explicit is Better than Implicit

### What It Means

Code should be clear and readable, avoiding "magic" that happens behind the scenes.

### How It Applies

- **Configuration Management**:

  ```python
  class Settings(BaseSettings):
      model_name: str
      api_key: str
  ```

- **Function Signatures**:

  ```python
  # Good
  def get_answer(question: str, history: list[str]) -> tuple[str, float]:
      
  # Avoid
  def get_answer(*args) -> Any:
  ```

### Why It's Important

- Reduces cognitive load
- Makes code self-documenting
- Helps catch errors through type checking

## Configuration Before Code

### What It Means

Avoid hardcoding values that might need to change.

### How It Applies

- Store configuration in `.env` files:

  ```env
  LLM_MODEL=google/gemini-1.5-pro-latest
  RELEVANCY_THRESHOLD=0.75
  ```

- Keep prompt templates in separate files:

  ```
  /app
    /prompts
      system_prompt.txt
      user_prompt.txt
  ```

### Why It's Important

- Enables experimentation without code changes
- Simplifies deployment to different environments
- Makes the application more configurable

## Best Practices Summary

1. **Keep Files Focused**
   - One responsibility per file
   - Clear and descriptive file names
   - Logical directory structure

2. **Use Type Hints**
   - Makes code more maintainable
   - Enables better IDE support
   - Catches errors early

3. **Document Intentionally**
   - Write docstrings for public functions
   - Include examples in docstrings
   - Comment complex logic, not obvious code

4. **Error Handling**
   - Use custom exception classes
   - Provide meaningful error messages
   - Handle edge cases explicitly

5. **Testing Considerations**
   - Write testable code from the start
   - Use dependency injection
   - Keep functions pure when possible
