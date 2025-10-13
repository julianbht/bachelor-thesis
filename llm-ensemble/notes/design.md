### Domain vs Adapters — Clean Architecture / Domain-Driven Design

This structure follows **Clean Architecture** (a.k.a. **Hexagonal Architecture** or **Ports and Adapters pattern**).  
It separates *core logic* (the **Domain**) from *infrastructure and I/O* (the **Adapters**).

---

**Domain layer 🧠 — “the logic itself”**  
Pure, framework-agnostic code defining what the program *does*, not how it interacts with the world.  
- Works only with Python data structures (lists, dicts, DataFrames).  
- No I/O, HTTP, or APIs — easy to test and reason about.  
- Example domain tasks:
  - **ingest:** normalize + validate dataset  
  - **infer:** build prompts, parse model outputs  
  - **aggregate:** compute ensemble votes, disagreement  
  - **evaluate:** compute metrics, generate summaries  

---

**Adapters layer 🔌 — “the outside world”**  
Handles integration and I/O — adapting external systems to domain logic.  
- Knows about HTTP, file paths, APIs, retries, timeouts.  
- Example adapters:
  - **ingest:** dataset loaders (CSV, JSON, HF)  
  - **infer:** model API wrappers (Ollama, HF, LM Studio)  
  - **aggregate:** Parquet readers/writers  
  - **evaluate:** report/metrics exporters  

---

**Benefits 💡**
- *Isolation:* test logic without APIs or GPUs  
- *Swapability:* replace providers easily  
- *Refactor safety:* change logic or I/O independently  
- *Parallel dev:* multiple people can work on different layers  
- *Future-proof:* only adapters change if you move to services or Docker

