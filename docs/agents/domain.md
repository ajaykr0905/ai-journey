# Domain documentation

This repository uses a single-context domain layout.

Before exploring or changing a subsystem:

1. Read `GLOSSARY.md` when it exists.
2. Read relevant decisions under `docs/adr/`.
3. Use established glossary terms in specifications, tickets, tests, and code.
4. Surface conflicts with existing ADRs instead of silently overriding them.
5. Allow domain-modeling work to create glossary entries and ADRs only when actual terminology or architectural decisions are resolved.

Expected layout:

```text
/
├── GLOSSARY.md
├── docs/
│   ├── adr/
│   └── agents/
└── src/
```
